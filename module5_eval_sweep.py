import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, os, math
import matplotlib.pyplot as plt
from tqdm import tqdm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════
COMBINED_DIR = f'{DRIVE_BASE}/module4_combined'
MODULE5_DIR  = f'{DRIVE_BASE}/module5'
CKPT_BEST    = f'{MODULE5_DIR}/diffusion_ckpt_best.pth'
CKPT_LATEST  = f'{MODULE5_DIR}/diffusion_ckpt.pth'

STEP_COUNTS = [5, 10, 20, 30, 50]   # DDIM step counts to sweep, deterministic (eta=0)
EVAL_BATCH  = 32                     # no backward pass here, so memory is not tight --
                                      # this is the same batch size Module 5's own periodic
                                      # validation used successfully; raise it if you want
                                      # this sweep to run faster and don't hit OOM.
SEED = 2026

torch.manual_seed(SEED)
np.random.seed(SEED)

assert os.path.exists(CKPT_BEST) or os.path.exists(CKPT_LATEST), \
    f'No Module 5 checkpoint found in {MODULE5_DIR}'

# ════════════════════════════════════════════════════════════════════
# DATA -- full validation set only, no training data needed for evaluation
# ════════════════════════════════════════════════════════════════════
val_inp   = np.load(f'{COMBINED_DIR}/val_inputs.npy').astype(np.float32)
val_tgt   = np.load(f'{COMBINED_DIR}/val_targets.npy').astype(np.float32)
val_dnorm = np.load(f'{COMBINED_DIR}/val_disp_norm.npy').astype(np.float32)

val_inp_t   = torch.from_numpy(val_inp).to(DEVICE)
val_tgt_t   = torch.from_numpy(val_tgt).to(DEVICE)
val_dnorm_t = torch.from_numpy(val_dnorm).to(DEVICE)
M_va = len(val_inp)

print('='*60)
print('  MODULE 5 EVAL — DDIM Step-Count Sweep (full validation set)')
print('='*60)
print(f'  Device      : {DEVICE}')
print(f'  Val windows : {M_va:,}')
print(f'  Step counts : {STEP_COUNTS}')


# ════════════════════════════════════════════════════════════════════
# MODEL -- identical definitions to module5_diffusion.py (must match the
# checkpoint's saved weights exactly). Dimensions are passed as constructor
# args here (rather than read from module-level globals as in the training
# script) so this file has no dependency on module5_diffusion.py's CONFIG
# block and can't silently drift out of sync with it.
# ════════════════════════════════════════════════════════════════════
class GeomMLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def sinusoidal_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / half)
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ConditionalDenoiser(nn.Module):
    def __init__(self, window_size, latent_dim, hidden_dim, n_heads, n_tf_layers, dropout):
        super().__init__()
        D = latent_dim
        self.context_geo = GeomMLP(3, hidden_dim, latent_dim, dropout)
        self.target_geo  = GeomMLP(3, hidden_dim, latent_dim, dropout)
        self.temporal_pe = nn.Parameter(torch.randn(window_size, D) * 0.02)

        tf_layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=n_heads, dim_feedforward=hidden_dim,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.temporal_transformer = nn.TransformerEncoder(
            tf_layer, num_layers=n_tf_layers, enable_nested_tensor=False)

        self.time_mlp = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))

        self.cross_t2c = nn.MultiheadAttention(D, n_heads, dropout=dropout, batch_first=True)
        self.cross_c2t = nn.MultiheadAttention(D, n_heads, dropout=dropout, batch_first=True)
        self.norm_t = nn.LayerNorm(D)
        self.norm_c = nn.LayerNorm(D)

        self.output_mlp = nn.Sequential(
            nn.Linear(2 * D, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, 3),
        )

    def encode_context(self, ctx):
        B, Wn, N_A, _ = ctx.shape
        h = self.context_geo(ctx)
        h = h + self.temporal_pe[None, :, None, :]
        h = h.permute(0, 2, 1, 3).reshape(B * N_A, Wn, -1)
        h = self.temporal_transformer(h)
        h = h[:, -1, :]                      # last-token pooling -- matches module5_diffusion.py
        return h.reshape(B, N_A, -1)

    def forward(self, ctx, x_noisy, t):
        ctx_latent = self.encode_context(ctx)
        tgt_latent = self.target_geo(x_noisy)
        tgt_latent = tgt_latent + self.time_mlp(sinusoidal_embedding(t, tgt_latent.shape[-1]))[:, None, :]

        attn_t, _ = self.cross_t2c(tgt_latent, ctx_latent, ctx_latent)
        tgt_upd = self.norm_t(tgt_latent + attn_t)
        attn_c, _ = self.cross_c2t(ctx_latent, tgt_latent, tgt_latent)
        ctx_upd = self.norm_c(ctx_latent + attn_c)

        fused = torch.cat([tgt_upd, ctx_upd], dim=-1)
        return self.output_mlp(fused)


# ════════════════════════════════════════════════════════════════════
# DIFFUSION MATHS -- identical formulas to module5_diffusion.py, parameterised
# per-checkpoint instead of module-global (this script can evaluate two
# checkpoints that, in principle, could have different T_DIFF).
# ════════════════════════════════════════════════════════════════════
def cosine_beta_schedule(T, s=0.008):
    steps = T + 1
    x = torch.linspace(0, T, steps)
    ac = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    ac = ac / ac[0]
    betas = 1 - (ac[1:] / ac[:-1])
    return torch.clip(betas, 1e-4, 0.999)


def pred_x0_from_v(x_t, v_pred, t, sqrt_ac, sqrt_1m_ac):
    a = sqrt_ac[t][:, None, None]
    b = sqrt_1m_ac[t][:, None, None]
    return a * x_t - b * v_pred


def pred_eps_from_v(x_t, v_pred, t, sqrt_ac, sqrt_1m_ac):
    a = sqrt_ac[t][:, None, None]
    b = sqrt_1m_ac[t][:, None, None]
    return b * x_t + a * v_pred


@torch.no_grad()
def ddim_sample(model, ctx, T_diff, alphas_cumprod, sqrt_ac, sqrt_1m_ac, steps, eta=0.0, generator=None):
    B, N_A = ctx.shape[0], ctx.shape[2]
    ts = torch.linspace(T_diff - 1, 0, steps, device=DEVICE).long()
    x = torch.randn(B, N_A, 3, device=DEVICE, generator=generator)
    for i, t in enumerate(ts):
        t_batch = t.expand(B)
        v_pred = model(ctx, x, t_batch)
        x0_pred = pred_x0_from_v(x, v_pred, t_batch, sqrt_ac, sqrt_1m_ac)
        eps_pred = pred_eps_from_v(x, v_pred, t_batch, sqrt_ac, sqrt_1m_ac)
        if i == len(ts) - 1:
            x = x0_pred
            break
        t_next = ts[i + 1]
        ac_t = alphas_cumprod[t]
        ac_next = alphas_cumprod[t_next]
        sigma = eta * torch.sqrt((1 - ac_next) / (1 - ac_t)) * torch.sqrt(1 - ac_t / ac_next)
        noise = torch.randn(B, N_A, 3, device=DEVICE, generator=generator) if eta > 0 else 0.0
        x = torch.sqrt(ac_next) * x0_pred + torch.sqrt(torch.clamp(1 - ac_next - sigma ** 2, min=0.0)) * eps_pred
        if eta > 0:
            x = x + sigma * noise
    return x


def chamfer_distance(pred, gt):
    d = torch.cdist(pred, gt)
    d1 = d.min(dim=2)[0].mean(dim=1)
    d2 = d.min(dim=1)[0].mean(dim=1)
    return (d1 + d2).mean()


def to_physical(delta_scaled, dnorm, data_scale):
    return (delta_scaled / data_scale) / dnorm[:, None, None]


# ════════════════════════════════════════════════════════════════════
# LOAD CHECKPOINT -- architecture dims + DATA_SCALE are read from the
# checkpoint's own saved 'arch_cfg'/'config', not re-typed here, so this
# script can't silently mismatch whatever module5_diffusion.py actually
# trained with.
# ════════════════════════════════════════════════════════════════════
def load_checkpoint(ckpt_path):
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    arch = ck['arch_cfg']
    data_scale = ck['config']['DATA_SCALE']
    model = ConditionalDenoiser(
        window_size=arch['WINDOW_SIZE'], latent_dim=arch['LATENT_DIM'],
        hidden_dim=arch['HIDDEN_DIM'], n_heads=arch['N_HEADS'],
        n_tf_layers=arch['N_TF_LAYERS'], dropout=0.1,
    ).to(DEVICE)
    model.load_state_dict(ck['ema'])
    model.eval()
    betas = cosine_beta_schedule(arch['T_DIFF']).to(DEVICE)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    return {
        'model': model, 'T_diff': arch['T_DIFF'], 'alphas_cumprod': alphas_cumprod,
        'sqrt_ac': torch.sqrt(alphas_cumprod), 'sqrt_1m_ac': torch.sqrt(1.0 - alphas_cumprod),
        'data_scale': data_scale, 'epoch': ck.get('epoch', '?'),
    }


@torch.no_grad()
def evaluate_full_set(bundle, steps, eta=0.0):
    total_cd, n = 0.0, 0
    for bi in range(0, M_va, EVAL_BATCH):
        sl = slice(bi, bi + EVAL_BATCH)
        ctx, tgt, dnorm = val_inp_t[sl], val_tgt_t[sl], val_dnorm_t[sl]
        x0_scaled = ddim_sample(bundle['model'], ctx, bundle['T_diff'], bundle['alphas_cumprod'],
                                 bundle['sqrt_ac'], bundle['sqrt_1m_ac'], steps=steps, eta=eta)
        delta_phys = to_physical(x0_scaled, dnorm, bundle['data_scale'])
        pred_pos = ctx[:, -1] + delta_phys
        cd = chamfer_distance(pred_pos, tgt)
        total_cd += cd.item() * len(ctx)
        n += len(ctx)
    return total_cd / n


@torch.no_grad()
def naive_cd_full_set():
    total_cd, n = 0.0, 0
    for bi in range(0, M_va, EVAL_BATCH):
        sl = slice(bi, bi + EVAL_BATCH)
        ctx, tgt = val_inp_t[sl], val_tgt_t[sl]
        cd = chamfer_distance(ctx[:, -1], tgt)
        total_cd += cd.item() * len(ctx)
        n += len(ctx)
    return total_cd / n


# ════════════════════════════════════════════════════════════════════
# RUN SWEEP
# ════════════════════════════════════════════════════════════════════
naive_cd = naive_cd_full_set()
print(f'\nNaive baseline CD (full val set, {M_va:,} windows): {naive_cd:.6f}\n')

checkpoints_to_eval = []
if os.path.exists(CKPT_BEST):
    checkpoints_to_eval.append(('best', CKPT_BEST))
if os.path.exists(CKPT_LATEST):
    checkpoints_to_eval.append(('latest', CKPT_LATEST))

results = {}   # {label: {steps: cd}}
for label, path in checkpoints_to_eval:
    bundle = load_checkpoint(path)
    print(f'--- {label} checkpoint (epoch {bundle["epoch"]}) ---')
    results[label] = {}
    for steps in tqdm(STEP_COUNTS, desc=f'{label} sweep'):
        cd = evaluate_full_set(bundle, steps, eta=0.0)
        results[label][steps] = cd
        pct = (naive_cd - cd) / naive_cd * 100
        tag = '✅ beats naive' if cd < naive_cd else '⚠️  worse than naive'
        print(f'  steps={steps:>3d}  CD={cd:.6f}  ({pct:+.1f}% vs naive)  {tag}')
    del bundle
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    print()

# ════════════════════════════════════════════════════════════════════
# VISUALISATION
# ════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))
colors = {'best': 'steelblue', 'latest': 'crimson'}
for label, per_step in results.items():
    xs = sorted(per_step.keys())
    ys = [per_step[s] for s in xs]
    ax.plot(xs, ys, marker='o', color=colors.get(label, 'gray'), label=f'{label} checkpoint')
ax.axhline(naive_cd, color='black', linestyle='--', linewidth=1, label='naive baseline')
ax.set_xlabel('DDIM steps'); ax.set_ylabel('Chamfer Distance (full val set)')
ax.set_title('Module 5 — DDIM Step Count vs Chamfer Distance', fontweight='bold')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{MODULE5_DIR}/module5_step_sweep.png', dpi=120, bbox_inches='tight')
plt.show()

print('='*60)
print('  SWEEP COMPLETE')
print('='*60)
for label, per_step in results.items():
    best_steps = min(per_step, key=per_step.get)
    verdict = 'beats' if per_step[best_steps] < naive_cd else 'does not beat'
    print(f'  {label:<8} best step count: {best_steps:>3d}  '
          f'(CD={per_step[best_steps]:.6f} vs naive={naive_cd:.6f}, {verdict} naive)')
print(f'  Saved figure: {MODULE5_DIR}/module5_step_sweep.png')
