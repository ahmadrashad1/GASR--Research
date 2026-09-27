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

STEP_COUNTS  = [5, 10, 20, 30, 50]   # DDIM step counts to sweep, deterministic (eta=0)
# Second sweep axis: how many independent trajectories are averaged into one prediction.
# Chamfer Distance scores a single point prediction and is minimised by the conditional
# MEAN, not by a draw from the conditional -- so n_samples is at least as strong a lever on
# the reported CD as the step count is, and sweeping only steps (as this script originally
# did) hides it. n_samples=1 reproduces the old single-trajectory numbers.
SAMPLE_COUNTS = [1, 2, 4, 8]
EVAL_ROWS   = 32                     # windows * n_samples resident at once; no backward
                                      # pass here, so this is far below the training ceiling
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
val_ps    = np.load(f'{COMBINED_DIR}/val_pred_step.npy').astype(np.int64)   # prediction horizon

val_inp_t   = torch.from_numpy(val_inp).to(DEVICE)
val_tgt_t   = torch.from_numpy(val_tgt).to(DEVICE)
val_dnorm_t = torch.from_numpy(val_dnorm).to(DEVICE)
val_ps_t    = torch.from_numpy(val_ps).to(DEVICE)
M_va = len(val_inp)

print('='*60)
print('  MODULE 5 EVAL — DDIM Step-Count Sweep (full validation set)')
print('='*60)
print(f'  Device      : {DEVICE}')
print(f'  Val windows : {M_va:,}')
print(f'  Step counts : {STEP_COUNTS}')
print(f'  Sample counts: {SAMPLE_COUNTS}  (trajectories averaged per window)')


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
    def __init__(self, window_size, latent_dim, hidden_dim, n_heads, n_tf_layers, dropout,
                 max_pred_step):
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
        # Horizon conditioning -- must mirror module5_diffusion.py exactly or the state_dict
        # will not load. See MAX_PRED_STEP there for why it exists.
        self.horizon_emb = nn.Embedding(max_pred_step + 1, D)
        self.horizon_mlp = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))

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

    def forward(self, ctx, x_noisy, t, hstep):
        h_emb = self.horizon_mlp(self.horizon_emb(hstep))[:, None, :]
        ctx_latent = self.encode_context(ctx) + h_emb
        tgt_latent = self.target_geo(x_noisy)
        tgt_latent = tgt_latent + self.time_mlp(sinusoidal_embedding(t, tgt_latent.shape[-1]))[:, None, :]
        tgt_latent = tgt_latent + h_emb

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
def ddim_sample(model, ctx, hstep, T_diff, alphas_cumprod, sqrt_ac, sqrt_1m_ac, steps,
                 eta=0.0, generator=None, n_samples=1):
    """n_samples>1 averages that many independent trajectories into one prediction -- a
    Monte-Carlo estimate of E[x0 | context, horizon]. Mirrors module5_diffusion.py."""
    B, N_A = ctx.shape[0], ctx.shape[2]
    if n_samples > 1:
        ctx = ctx.repeat_interleave(n_samples, dim=0)
        hstep = hstep.repeat_interleave(n_samples, dim=0)
    BK = ctx.shape[0]
    ts = torch.linspace(T_diff - 1, 0, steps, device=DEVICE).long()
    x = torch.randn(BK, N_A, 3, device=DEVICE, generator=generator)
    for i, t in enumerate(ts):
        t_batch = t.expand(BK)
        v_pred = model(ctx, x, t_batch, hstep)
        x0_pred = pred_x0_from_v(x, v_pred, t_batch, sqrt_ac, sqrt_1m_ac)
        eps_pred = pred_eps_from_v(x, v_pred, t_batch, sqrt_ac, sqrt_1m_ac)
        if i == len(ts) - 1:
            x = x0_pred
            break
        t_next = ts[i + 1]
        ac_t = alphas_cumprod[t]
        ac_next = alphas_cumprod[t_next]
        sigma = eta * torch.sqrt((1 - ac_next) / (1 - ac_t)) * torch.sqrt(1 - ac_t / ac_next)
        noise = torch.randn(BK, N_A, 3, device=DEVICE, generator=generator) if eta > 0 else 0.0
        x = torch.sqrt(ac_next) * x0_pred + torch.sqrt(torch.clamp(1 - ac_next - sigma ** 2, min=0.0)) * eps_pred
        if eta > 0:
            x = x + sigma * noise
    if n_samples > 1:
        x = x.view(B, n_samples, N_A, 3).mean(dim=1)
    return x


def chamfer_distance(pred, gt):
    d = torch.cdist(pred, gt)
    d1 = d.min(dim=2)[0].mean(dim=1)
    d2 = d.min(dim=1)[0].mean(dim=1)
    return (d1 + d2).mean()


def to_physical(delta_scaled, dnorm, data_scale):
    return (delta_scaled / data_scale) / dnorm[:, None, None]


def base_pred(ctx, hstep, target_mode):
    """What the network's output is a correction to: the last observed frame under
    target_mode='delta', constant-velocity extrapolation under 'cv_residual'. Read from the
    checkpoint rather than assumed -- reconstructing a cv_residual model as a delta one
    silently throws away the constant-velocity term and reports nonsense."""
    if target_mode == 'cv_residual':
        return const_velocity_pred(ctx, hstep)
    return ctx[:, -1]


def const_velocity_pred(ctx, hstep):
    """last + (last - previous) * horizon -- the reference bar from Module 4's QT7."""
    last, prev = ctx[:, -1], ctx[:, -2]
    return last + (last - prev) * hstep.to(last.dtype)[:, None, None]


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
        # Older checkpoints (pre horizon-conditioning) have no MAX_PRED_STEP; they also
        # have no horizon_emb weights, so load_state_dict below will reject them -- which is
        # correct, they were trained on a different (ambiguous) objective and their numbers
        # are not comparable to a horizon-conditioned run's.
        max_pred_step=arch.get('MAX_PRED_STEP', 8),
    ).to(DEVICE)
    model.load_state_dict(ck['ema'])
    model.eval()
    betas = cosine_beta_schedule(arch['T_DIFF']).to(DEVICE)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    return {
        'model': model, 'T_diff': arch['T_DIFF'], 'alphas_cumprod': alphas_cumprod,
        'sqrt_ac': torch.sqrt(alphas_cumprod), 'sqrt_1m_ac': torch.sqrt(1.0 - alphas_cumprod),
        'data_scale': data_scale, 'epoch': ck.get('epoch', '?'),
        'target_mode': arch.get('TARGET_MODE', 'delta'),
    }


@torch.no_grad()
def evaluate_full_set(bundle, steps, n_samples=1, eta=0.0):
    rows = max(1, EVAL_ROWS // max(1, n_samples))
    total_cd, n = 0.0, 0
    for bi in range(0, M_va, rows):
        sl = slice(bi, bi + rows)
        ctx, tgt, dnorm, hstep = val_inp_t[sl], val_tgt_t[sl], val_dnorm_t[sl], val_ps_t[sl]
        x0_scaled = ddim_sample(bundle['model'], ctx, hstep, bundle['T_diff'],
                                 bundle['alphas_cumprod'], bundle['sqrt_ac'],
                                 bundle['sqrt_1m_ac'], steps=steps, eta=eta,
                                 n_samples=n_samples)
        delta_phys = to_physical(x0_scaled, dnorm, bundle['data_scale'])
        pred_pos = base_pred(ctx, hstep, bundle['target_mode']) + delta_phys
        cd = chamfer_distance(pred_pos, tgt)
        total_cd += cd.item() * len(ctx)
        n += len(ctx)
    return total_cd / n


@torch.no_grad()
def baseline_cds_full_set():
    """Naive (zero motion) and constant-velocity references, on the same windows."""
    tot_naive, tot_cv, n = 0.0, 0.0, 0
    for bi in range(0, M_va, EVAL_ROWS):
        sl = slice(bi, bi + EVAL_ROWS)
        ctx, tgt, hstep = val_inp_t[sl], val_tgt_t[sl], val_ps_t[sl]
        tot_naive += chamfer_distance(ctx[:, -1], tgt).item() * len(ctx)
        tot_cv += chamfer_distance(const_velocity_pred(ctx, hstep), tgt).item() * len(ctx)
        n += len(ctx)
    return tot_naive / n, tot_cv / n


# ════════════════════════════════════════════════════════════════════
# RUN SWEEP
# ════════════════════════════════════════════════════════════════════
naive_cd, cv_cd = baseline_cds_full_set()
print(f'\nBaselines (full val set, {M_va:,} windows):')
print(f'  Naive (zero motion)            : {naive_cd:.6f}')
print(f'  Constant velocity (last + v*k) : {cv_cd:.6f}  ← the bar worth clearing\n')

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
    combos = [(st, ns) for ns in SAMPLE_COUNTS for st in STEP_COUNTS]
    for steps, ns in tqdm(combos, desc=f'{label} sweep'):
        cd = evaluate_full_set(bundle, steps, n_samples=ns, eta=0.0)
        results[label].setdefault(ns, {})[steps] = cd
        pct = (naive_cd - cd) / naive_cd * 100
        pct_cv = (cv_cd - cd) / cv_cd * 100
        tag = ('✅ beats const-vel' if cd < cv_cd
               else '➖ beats naive only' if cd < naive_cd else '⚠️  worse than naive')
        print(f'  steps={steps:>3d}  samples={ns:>2d}  CD={cd:.6f}  '
              f'({pct:+.1f}% vs naive, {pct_cv:+.1f}% vs const-vel)  {tag}')
    del bundle
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    print()

# ════════════════════════════════════════════════════════════════════
# VISUALISATION
# ════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, max(1, len(results)), figsize=(7 * max(1, len(results)), 5),
                          squeeze=False)
shades = ['#c6dbef', '#6baed6', '#2171b5', '#08306b']
for ai, (label, per_ns) in enumerate(results.items()):
    ax = axes[0][ai]
    for j, ns in enumerate(sorted(per_ns)):
        xs = sorted(per_ns[ns])
        ax.plot(xs, [per_ns[ns][st] for st in xs], marker='o',
                color=shades[j % len(shades)], label=f'{ns} sample{"s" if ns > 1 else ""}')
    ax.axhline(naive_cd, color='black', linestyle='--', linewidth=1, label='naive')
    ax.axhline(cv_cd, color='darkorange', linestyle=':', linewidth=1.5, label='const-velocity')
    ax.set_xlabel('DDIM steps'); ax.set_ylabel('Chamfer Distance (full val set)')
    ax.set_title(f'{label} checkpoint', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.suptitle('Module 5 — DDIM steps x averaged samples vs Chamfer Distance', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{MODULE5_DIR}/module5_step_sweep.png', dpi=120, bbox_inches='tight')
plt.show()

print('='*60)
print('  SWEEP COMPLETE')
print('='*60)
for label, per_ns in results.items():
    flat = {(st, ns): cd for ns, d in per_ns.items() for st, cd in d.items()}
    (b_st, b_ns) = min(flat, key=flat.get)
    cd = flat[(b_st, b_ns)]
    verdict = ('beats const-velocity' if cd < cv_cd
               else 'beats naive but not const-velocity' if cd < naive_cd else 'beats neither')
    print(f'  {label:<8} best config: steps={b_st}, samples={b_ns}  '
          f'(CD={cd:.6f} vs naive={naive_cd:.6f}, const-vel={cv_cd:.6f} — {verdict})')
print(f'  Saved figure: {MODULE5_DIR}/module5_step_sweep.png')
