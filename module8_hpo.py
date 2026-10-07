import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, os, math, json, time, copy, itertools
import matplotlib.pyplot as plt
from tqdm import tqdm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ════════════════════════════════════════════════════════════════════
# MODULE 8 — HYPERPARAMETER AND OPTIMISATION STUDY
#
# Four studies, ordered by cost. Each answers a question a reviewer will ask about a
# result that currently does not beat its baselines: did you actually tune this, or did
# you train one configuration and report it?
#
#   A. INFERENCE COST vs ACCURACY   no retraining      ~20 min
#      How many denoising steps and averaged samples does the trained model actually
#      need? Produces a Pareto curve, which is the honest way to report a generative
#      model's accuracy: a single number hides that accuracy is purchased with compute.
#
#   B. CONTEXT LENGTH               cheap retraining   ~10 min
#      Does showing the model more past frames help? Tested by truncating the existing
#      windows to the last W' frames, so no Module 4 re-run is needed. If accuracy is
#      flat from 2 to 4 frames, widening the window further is not where the limit is.
#
#   C. OPTIMISATION SENSITIVITY     cheap retraining   ~30 min
#      One-factor-at-a-time sweeps around a reference configuration: optimiser, learning
#      rate, schedule, weight decay, batch size. Run on the deterministic Transformer
#      backbone, which trains in about a minute, so the grid is affordable. OFAT rather
#      than a full grid because the deliverable is a sensitivity ranking -- which knobs
#      matter -- not a single best cell.
#
#   D. DIFFUSION ABLATIONS          expensive          hours, opt-in
#      The choices specific to the generative formulation, which cannot be transferred
#      from a deterministic backbone: prediction parameterisation, Min-SNR weighting,
#      target formulation, horizon conditioning, capacity. Run at reduced budget on a
#      fraction of the data, which is valid here because Module 5's best checkpoint
#      occurred near epoch 60 -- well inside a short proxy run.
#
# EVERY study reports Mean Point Error on the same held-out windows as Modules 6 and 7,
# so all three modules' numbers sit in one table.
# ════════════════════════════════════════════════════════════════════

COMBINED_DIR = f'{DRIVE_BASE}/module4_combined'
MODULE5_DIR  = f'{DRIVE_BASE}/module5'
MODULE8_DIR  = f'{DRIVE_BASE}/module8'
CKPT_BEST    = f'{MODULE5_DIR}/diffusion_ckpt_best.pth'
CKPT_LATEST  = f'{MODULE5_DIR}/diffusion_ckpt.pth'
os.makedirs(MODULE8_DIR, exist_ok=True)

RUN_A, RUN_B, RUN_C = True, True, True
RUN_D = False            # opt-in: retrains the diffusion model once per ablation. The
                          # budget is printed before anything starts -- read it first.

# ── Study A: inference cost/accuracy ─────────────────────────────────
A_STEPS   = [2, 5, 10, 20, 50]
A_SAMPLES = [1, 2, 4, 8]
A_SUBSET  = 480          # held-out windows per cell. The full set x 20 cells would be
                          # hours of sampling for a curve whose shape is already clear at
                          # a third of the data; the final table uses the full set

# ── Study B/C: cheap backbone ────────────────────────────────────────
BACKBONE_EPOCHS = 40
BACKBONE_HIDDEN = 256
BACKBONE_BATCH  = 32
BACKBONE_LR     = 1e-3
B_WINDOWS = [2, 3, 4]    # truncations of the existing 4-frame context

# ── Study D: diffusion ablations ─────────────────────────────────────
D_EPOCHS        = 25
D_TRAIN_FRAC    = 0.40
D_BATCH         = 16
D_LR            = 5e-4
D_PROMOTE_TOP   = 2      # best configs are re-run at full budget, a two-rung successive
D_PROMOTE_EPOCH = 75     # halving: cheap screening, then a fair comparison of finalists

SEED = 11
torch.manual_seed(SEED); np.random.seed(SEED)

print('=' * 72)
print('  MODULE 8 — Hyperparameter and Optimisation Study')
print('=' * 72)

# ════════════════════════════════════════════════════════════════════
# DATA — identical windows and split to Modules 5-7
# ════════════════════════════════════════════════════════════════════
def _load(split):
    g = lambda n: np.load(f'{COMBINED_DIR}/{split}_{n}.npy')
    return {'inp': g('inputs').astype(np.float32), 'tgt': g('targets').astype(np.float32),
            'delta': g('deltas').astype(np.float32), 'dnorm': g('disp_norm').astype(np.float32),
            'ps': g('pred_step').astype(np.int64)}

tr, va = _load('train'), _load('val')
M_tr, W_FULL, N_A, _ = tr['inp'].shape
M_va = len(va['inp'])
PRED_STEPS = sorted(set(np.unique(tr['ps']).tolist()))
BASE_SCALE = float(1.0 / (tr['delta'].std() + 1e-8))
T = {k: torch.from_numpy(v).to(DEVICE) for k, v in tr.items()}
V = {k: torch.from_numpy(v).to(DEVICE) for k, v in va.items()}

print(f'  Device {DEVICE} · train {M_tr:,} · val {M_va:,} · points {N_A} · context {W_FULL}')


def mean_point_error(pred, gt):
    return (pred - gt).norm(dim=-1).mean(dim=-1)


def chamfer_distance(pred, gt):
    d = torch.cdist(pred, gt)
    return d.min(dim=2)[0].mean(dim=1) + d.min(dim=1)[0].mean(dim=1)


def to_physical(delta_eq, dnorm):
    return delta_eq / dnorm[:, None, None]


def const_velocity_delta(ctx, k):
    return (ctx[:, -1] - ctx[:, -2]) * k[:, None, None].to(ctx.dtype)


# ════════════════════════════════════════════════════════════════════
# DETERMINISTIC BACKBONE — the vehicle for studies B and C
# Same shape as the diffusion model's temporal encoder, minus the generative part, so
# conclusions about optimisation transfer while each run costs about a minute.
# ════════════════════════════════════════════════════════════════════
class Backbone(nn.Module):
    def __init__(self, hidden=BACKBONE_HIDDEN, layers=3, heads=8, window=W_FULL):
        super().__init__()
        self.window = window
        self.inp = nn.Linear(3, hidden)
        self.pe = nn.Parameter(torch.randn(window, hidden) * 0.02)
        lay = nn.TransformerEncoderLayer(hidden, heads, hidden * 2, dropout=0.1,
                                         activation='gelu', batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(lay, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(hidden + 1, hidden), nn.GELU(),
                                  nn.Linear(hidden, 3))

    def forward(self, ctx, k):
        ctx = ctx[:, -self.window:]                       # truncation = shorter context
        rel = (ctx - ctx[:, -1:]).permute(0, 2, 1, 3)
        B, N, Wn, _ = rel.shape
        h = self.inp(rel.reshape(B * N, Wn, 3)) + self.pe[None]
        h = self.tf(h)[:, -1].reshape(B, N, -1)
        kk = k.to(ctx.dtype)[:, None, None].expand(-1, N, 1)
        return self.head(torch.cat([h, kk], dim=-1))


@torch.no_grad()
def eval_backbone(model, batch=64):
    model.eval()
    tot, n = 0.0, 0
    for bi in range(0, M_va, batch):
        sl = slice(bi, min(bi + batch, M_va))
        pred = V['inp'][sl][:, -1] + to_physical(model(V['inp'][sl], V['ps'][sl]) / BASE_SCALE,
                                                 V['dnorm'][sl])
        e = mean_point_error(pred, V['tgt'][sl])
        tot += e.sum().item(); n += len(e)
    return tot / max(n, 1)


def train_backbone(window=W_FULL, hidden=BACKBONE_HIDDEN, layers=3, epochs=BACKBONE_EPOCHS,
                   lr=BACKBONE_LR, wd=1e-2, batch=BACKBONE_BATCH, optim='adamw',
                   schedule='cosine'):
    model = Backbone(hidden=hidden, layers=layers, window=window).to(DEVICE)
    if optim == 'adamw':
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif optim == 'adam':
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    if schedule == 'cosine':
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=1e-6)
    elif schedule == 'step':
        sch = torch.optim.lr_scheduler.StepLR(opt, max(1, epochs // 3), 0.3)
    else:
        sch = torch.optim.lr_scheduler.ConstantLR(opt, 1.0, total_iters=1)
    rng = np.random.default_rng(SEED)
    best = float('inf')
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(M_tr)
        for bi in range(0, M_tr - batch + 1, batch):
            sl = perm[bi:bi + batch]
            loss = F.mse_loss(model(T['inp'][sl], T['ps'][sl]), T['delta'][sl] * BASE_SCALE)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            best = min(best, eval_backbone(model))
    return best, time.time() - t0, sum(p.numel() for p in model.parameters())


# ════════════════════════════════════════════════════════════════════
# DIFFUSION MODEL — needed by study A (loading) and study D (training)
# ════════════════════════════════════════════════════════════════════
class GeomMLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim))

    def forward(self, x):
        return self.net(x)


def sinusoidal_embedding(t, dim):
    half = dim // 2
    f = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / half)
    a = t.float()[:, None] * f[None]
    e = torch.cat([torch.sin(a), torch.cos(a)], dim=-1)
    return F.pad(e, (0, 1)) if dim % 2 else e


class ConditionalDenoiser(nn.Module):
    def __init__(self, window_size=4, latent_dim=512, hidden_dim=1024, n_heads=8,
                 n_tf_layers=3, dropout=0.1, max_pred_step=8, use_horizon=True):
        super().__init__()
        D = latent_dim
        self.use_horizon = use_horizon
        self.context_geo = GeomMLP(3, hidden_dim, D, dropout)
        self.target_geo = GeomMLP(3, hidden_dim, D, dropout)
        self.temporal_pe = nn.Parameter(torch.randn(window_size, D) * 0.02)
        lay = nn.TransformerEncoderLayer(D, n_heads, hidden_dim, dropout=dropout,
                                         activation='gelu', batch_first=True, norm_first=True)
        self.temporal_transformer = nn.TransformerEncoder(lay, n_tf_layers,
                                                          enable_nested_tensor=False)
        self.time_mlp = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))
        self.horizon_emb = nn.Embedding(max_pred_step + 1, D)
        self.horizon_mlp = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))
        self.cross_t2c = nn.MultiheadAttention(D, n_heads, dropout=dropout, batch_first=True)
        self.cross_c2t = nn.MultiheadAttention(D, n_heads, dropout=dropout, batch_first=True)
        self.norm_t, self.norm_c = nn.LayerNorm(D), nn.LayerNorm(D)
        self.output_mlp = nn.Sequential(
            nn.Linear(2 * D, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, 3))

    def forward(self, ctx, x_noisy, t, hstep):
        B, Wn, N, _ = ctx.shape
        h = self.context_geo(ctx) + self.temporal_pe[None, :, None, :]
        h = h.permute(0, 2, 1, 3).reshape(B * N, Wn, -1)
        c = self.temporal_transformer(h)[:, -1, :].reshape(B, N, -1)
        g = self.target_geo(x_noisy)
        g = g + self.time_mlp(sinusoidal_embedding(t, g.shape[-1]))[:, None, :]
        if self.use_horizon:
            # the ablation that removes horizon conditioning removes it from BOTH branches,
            # restoring the ambiguous objective Module 5 originally had
            he = self.horizon_mlp(self.horizon_emb(hstep))[:, None, :]
            c, g = c + he, g + he
        a, _ = self.cross_t2c(g, c, c)
        g2 = self.norm_t(g + a)
        b, _ = self.cross_c2t(c, g, g)
        c2 = self.norm_c(c + b)
        return self.output_mlp(torch.cat([g2, c2], dim=-1))


def cosine_beta_schedule(T_, s=0.008):
    x = torch.linspace(0, T_, T_ + 1)
    ac = torch.cos(((x / T_) + s) / (1 + s) * math.pi * 0.5) ** 2
    return torch.clip(1 - ((ac / ac[0])[1:] / (ac / ac[0])[:-1]), 1e-4, 0.999)


class Diffusion:
    """Schedule plus the three prediction parameterisations, so study D can switch between
    them without touching the model: eps predicts the noise, x0 the clean target, and v the
    diffusion velocity (Salimans & Ho). Which one is best is empirical and the reason the
    ablation exists."""
    def __init__(self, T_=500, param='v', min_snr=5.0):
        self.T, self.param, self.min_snr = T_, param, min_snr
        ac = torch.cumprod(1.0 - cosine_beta_schedule(T_), 0).to(DEVICE)
        self.ac, self.sq, self.sq1 = ac, torch.sqrt(ac), torch.sqrt(1 - ac)
        snr = ac / (1 - ac)
        self.w = (torch.minimum(snr, torch.full_like(snr, min_snr)) / (snr + 1)
                  if math.isfinite(min_snr) else torch.ones_like(snr))

    def target(self, x0, noise, t):
        a, b = self.sq[t][:, None, None], self.sq1[t][:, None, None]
        return {'eps': noise, 'x0': x0, 'v': a * noise - b * x0}[self.param]

    def to_x0(self, x_t, out, t):
        a, b = self.sq[t][:, None, None], self.sq1[t][:, None, None]
        if self.param == 'x0':
            return out
        if self.param == 'eps':
            return (x_t - b * out) / a.clamp_min(1e-8)
        return a * x_t - b * out

    def to_eps(self, x_t, out, t):
        a, b = self.sq[t][:, None, None], self.sq1[t][:, None, None]
        if self.param == 'eps':
            return out
        if self.param == 'x0':
            return (x_t - a * out) / b.clamp_min(1e-8)
        return b * x_t + a * out


@torch.no_grad()
def ddim(model, diff, ctx, k, steps, n_samples=1, eta=0.0):
    B, N = ctx.shape[0], ctx.shape[2]
    c = ctx.repeat_interleave(n_samples, 0) if n_samples > 1 else ctx
    h = k.repeat_interleave(n_samples, 0) if n_samples > 1 else k
    rows = c.shape[0]
    ts = torch.linspace(diff.T - 1, 0, steps, device=DEVICE).long()
    x = torch.randn(rows, N, 3, device=DEVICE)
    for i, t in enumerate(ts):
        tb = t.expand(rows)
        out = model(c, x, tb, h)
        x0 = diff.to_x0(x, out, tb)
        if i == len(ts) - 1:
            x = x0
            break
        eps = diff.to_eps(x, out, tb)
        a_t, a_n = diff.ac[t], diff.ac[ts[i + 1]]
        sig = eta * torch.sqrt((1 - a_n) / (1 - a_t)) * torch.sqrt(1 - a_t / a_n)
        x = (torch.sqrt(a_n) * x0
             + torch.sqrt(torch.clamp(1 - a_n - sig ** 2, min=0.0)) * eps)
        if eta > 0:
            x = x + sig * torch.randn_like(x)
    return x.view(B, n_samples, N, 3).mean(1) if n_samples > 1 else x


results = {}

# ════════════════════════════════════════════════════════════════════
# STUDY A — inference cost vs accuracy, on the trained model
# ════════════════════════════════════════════════════════════════════
if RUN_A:
    path = CKPT_BEST if os.path.exists(CKPT_BEST) else CKPT_LATEST
    if not os.path.exists(path):
        print('\n  ⚠️  Study A skipped: no Module 5 checkpoint.')
        RUN_A = False
    else:
        ck = torch.load(path, map_location=DEVICE, weights_only=False)
        arch = ck['arch_cfg']
        m = ConditionalDenoiser(arch['WINDOW_SIZE'], arch['LATENT_DIM'], arch['HIDDEN_DIM'],
                                arch['N_HEADS'], arch['N_TF_LAYERS'], 0.1,
                                arch.get('MAX_PRED_STEP', 8)).to(DEVICE)
        m.load_state_dict(ck['ema']); m.eval()
        diff = Diffusion(arch['T_DIFF'], 'v', 5.0)
        scale, mode = ck['config']['DATA_SCALE'], arch.get('TARGET_MODE', 'delta')
        idx = np.linspace(0, M_va - 1, min(A_SUBSET, M_va)).astype(int)
        print(f'\n  STUDY A — inference cost vs accuracy '
              f'({len(idx)} windows, checkpoint epoch {ck.get("epoch", "?")})')
        grid = {}
        for steps, ns in itertools.product(A_STEPS, A_SAMPLES):
            rows = max(1, 64 // ns)
            tot, n, t0 = 0.0, 0, time.time()
            for bi in range(0, len(idx), rows):
                sl = idx[bi:bi + rows]
                ctx, k, dn = V['inp'][sl], V['ps'][sl], V['dnorm'][sl]
                d_eq = ddim(m, diff, ctx, k, steps, ns) / scale
                if mode == 'cv_residual':
                    d_eq = d_eq + const_velocity_delta(ctx, k) * dn[:, None, None]
                pred = ctx[:, -1] + to_physical(d_eq, dn)
                e = mean_point_error(pred, V['tgt'][sl])
                tot += e.sum().item(); n += len(e)
            grid[f'{steps}x{ns}'] = {'steps': steps, 'samples': ns, 'mpe': tot / n,
                                     'forward_passes': steps * ns,
                                     'seconds': time.time() - t0}
        results['A_inference'] = grid
        print(f'    {"steps":>6} {"samples":>8} {"fwd":>6} {"MPE":>11} {"sec":>7}')
        for key, g in sorted(grid.items(), key=lambda kv: kv[1]['mpe']):
            print(f'    {g["steps"]:>6} {g["samples"]:>8} {g["forward_passes"]:>6} '
                  f'{g["mpe"]:>11.6f} {g["seconds"]:>7.1f}')
        bestA = min(grid.values(), key=lambda g: g['mpe'])
        # the cheapest cell within 1% of the best is the operating point worth reporting:
        # anything beyond it buys accuracy below the noise floor at real compute cost
        knee = min((g for g in grid.values() if g['mpe'] <= bestA['mpe'] * 1.01),
                   key=lambda g: g['forward_passes'])
        results['A_best'], results['A_knee'] = bestA, knee
        print(f'    best {bestA["steps"]}x{bestA["samples"]} MPE {bestA["mpe"]:.6f}; '
              f'knee (within 1%) {knee["steps"]}x{knee["samples"]} at '
              f'{knee["forward_passes"]} forward passes '
              f'({bestA["forward_passes"] / knee["forward_passes"]:.1f}x cheaper)')

# ════════════════════════════════════════════════════════════════════
# STUDY B — context length
# ════════════════════════════════════════════════════════════════════
if RUN_B:
    print(f'\n  STUDY B — context length (truncating the {W_FULL}-frame window)')
    out = {}
    for w in B_WINDOWS:
        mpe, secs, par = train_backbone(window=w)
        out[w] = {'mpe': mpe, 'seconds': secs, 'parameters': par}
        print(f'    W={w}  MPE {mpe:.6f}  ({secs:.0f}s)')
    results['B_context'] = out
    ws = sorted(out)
    gain = (out[ws[0]]['mpe'] - out[ws[-1]]['mpe']) / out[ws[0]]['mpe'] * 100
    print(f'    going from W={ws[0]} to W={ws[-1]} changes MPE by {gain:+.1f}%')
    if abs(gain) < 2:
        print('    → essentially flat: the model is not limited by how much past it sees,'
              '\n      so widening the window further is unlikely to be where the gain is.')
    else:
        print('    → context length matters measurably; a Module 4 re-run at larger W is'
              '\n      worth testing.')

# ════════════════════════════════════════════════════════════════════
# STUDY C — optimisation sensitivity, one factor at a time
# ════════════════════════════════════════════════════════════════════
if RUN_C:
    print('\n  STUDY C — optimisation sensitivity (one factor at a time)')
    ref = dict(optim='adamw', lr=BACKBONE_LR, schedule='cosine', wd=1e-2,
               batch=BACKBONE_BATCH, hidden=BACKBONE_HIDDEN, layers=3)
    ref_mpe, ref_s, _ = train_backbone(**ref)
    print(f'    reference {ref} -> MPE {ref_mpe:.6f} ({ref_s:.0f}s)')
    factors = {
        'optimiser':     ('optim', ['adam', 'sgd']),
        'learning rate': ('lr', [1e-4, 3e-4, 3e-3]),
        'schedule':      ('schedule', ['constant', 'step']),
        'weight decay':  ('wd', [0.0, 5e-2]),
        'batch size':    ('batch', [16, 64]),
        'width':         ('hidden', [128, 512]),
        'depth':         ('layers', [1, 6]),
    }
    sens = {}
    for label, (key, values) in factors.items():
        rows = [{'value': ref[key], 'mpe': ref_mpe, 'reference': True}]
        for v in values:
            cfg = dict(ref); cfg[key] = v
            mpe, secs, _ = train_backbone(**cfg)
            rows.append({'value': v, 'mpe': mpe, 'seconds': secs, 'reference': False})
            print(f'    {label:<15} {key}={v!r:<10} MPE {mpe:.6f}')
        spread = (max(r['mpe'] for r in rows) - min(r['mpe'] for r in rows)) / ref_mpe * 100
        sens[label] = {'key': key, 'rows': rows, 'spread_pct': spread}
    results['C_optimisation'] = {'reference': {**ref, 'mpe': ref_mpe}, 'factors': sens}
    print('\n    sensitivity ranking (spread in MPE across the values tried):')
    for label, d in sorted(sens.items(), key=lambda kv: -kv[1]['spread_pct']):
        print(f'      {label:<15} {d["spread_pct"]:>6.1f}%')

# ════════════════════════════════════════════════════════════════════
# STUDY D — diffusion ablations (opt-in)
# ════════════════════════════════════════════════════════════════════
D_CONFIGS = [
    {'name': 'reference (v-pred, min-SNR 5, cv-residual)', 'param': 'v',  'min_snr': 5.0,  'target': 'cv_residual'},
    {'name': 'epsilon-prediction',                          'param': 'eps','min_snr': 5.0,  'target': 'cv_residual'},
    {'name': 'x0-prediction',                               'param': 'x0', 'min_snr': 5.0,  'target': 'cv_residual'},
    {'name': 'no Min-SNR weighting',                        'param': 'v',  'min_snr': math.inf, 'target': 'cv_residual'},
    {'name': 'Min-SNR gamma = 1',                           'param': 'v',  'min_snr': 1.0,  'target': 'cv_residual'},
    {'name': 'raw-displacement target',                     'param': 'v',  'min_snr': 5.0,  'target': 'delta'},
    {'name': 'no horizon conditioning',                     'param': 'v',  'min_snr': 5.0,  'target': 'cv_residual', 'use_horizon': False},
    {'name': 'half width (latent 256)',                     'param': 'v',  'min_snr': 5.0,  'target': 'cv_residual', 'latent': 256, 'hidden': 512},
    {'name': 'six transformer layers',                      'param': 'v',  'min_snr': 5.0,  'target': 'cv_residual', 'layers': 6},
]


def build_targets(target_mode):
    """x0 in the model's scaled space, for whichever target formulation is being ablated."""
    cv_tr = const_velocity_delta(T['inp'], T['ps']) * T['dnorm'][:, None, None]
    cv_va = const_velocity_delta(V['inp'], V['ps']) * V['dnorm'][:, None, None]
    if target_mode == 'cv_residual':
        a, b = T['delta'] - cv_tr, V['delta'] - cv_va
    else:
        a, b = T['delta'], V['delta']
    s = float(1.0 / (a.std().item() + 1e-8))
    return a * s, b * s, s, cv_va


def train_diffusion(cfg, epochs, frac):
    tgt_tr, _, scale, cv_va = build_targets(cfg.get('target', 'cv_residual'))
    model = ConditionalDenoiser(W_FULL, cfg.get('latent', 512), cfg.get('hidden', 1024),
                                8, cfg.get('layers', 3), 0.1, 8,
                                cfg.get('use_horizon', True)).to(DEVICE)
    diff = Diffusion(500, cfg['param'], cfg['min_snr'])
    opt = torch.optim.AdamW(model.parameters(), lr=D_LR, weight_decay=0.05)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler(DEVICE, enabled=(DEVICE == 'cuda'))
    rng = np.random.default_rng(SEED)
    n_use = int(M_tr * frac)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(M_tr)[:n_use]
        for bi in range(0, n_use - D_BATCH + 1, D_BATCH):
            sl = perm[bi:bi + D_BATCH]
            x0 = tgt_tr[sl]
            t = torch.randint(0, diff.T, (len(sl),), device=DEVICE)
            noise = torch.randn_like(x0)
            x_t = diff.sq[t][:, None, None] * x0 + diff.sq1[t][:, None, None] * noise
            with torch.autocast(device_type=DEVICE, dtype=torch.float16,
                                enabled=(DEVICE == 'cuda')):
                out = model(T['inp'][sl], x_t, t, T['ps'][sl])
                loss = (diff.w[t][:, None, None] *
                        (out - diff.target(x0, noise, t)) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        sch.step()
    # evaluate at the operating point study A identified, or a sensible default
    steps = results.get('A_knee', {}).get('steps', 20)
    ns = results.get('A_knee', {}).get('samples', 4)
    model.eval()
    idx = np.linspace(0, M_va - 1, min(A_SUBSET, M_va)).astype(int)
    tot, n = 0.0, 0
    rows = max(1, 64 // ns)
    for bi in range(0, len(idx), rows):
        sl = idx[bi:bi + rows]
        ctx, k, dn = V['inp'][sl], V['ps'][sl], V['dnorm'][sl]
        d_eq = ddim(model, diff, ctx, k, steps, ns) / scale
        if cfg.get('target', 'cv_residual') == 'cv_residual':
            d_eq = d_eq + const_velocity_delta(ctx, k) * dn[:, None, None]
        e = mean_point_error(ctx[:, -1] + to_physical(d_eq, dn), V['tgt'][sl])
        tot += e.sum().item(); n += len(e)
    return tot / n, time.time() - t0, sum(p.numel() for p in model.parameters())


if RUN_D:
    est = len(D_CONFIGS) * D_EPOCHS * D_TRAIN_FRAC + D_PROMOTE_TOP * D_PROMOTE_EPOCH
    print(f'\n  STUDY D — diffusion ablations')
    print(f'    {len(D_CONFIGS)} configs x {D_EPOCHS} epochs at {D_TRAIN_FRAC:.0%} data, '
          f'then top {D_PROMOTE_TOP} at {D_PROMOTE_EPOCH} epochs')
    print(f'    ≈ {est:.0f} full-data-epoch equivalents; at ~100 s/epoch that is '
          f'≈ {est * 100 / 3600:.1f} h. Interrupt now if that is too long.')
    rung1 = {}
    for cfg in D_CONFIGS:
        mpe, secs, par = train_diffusion(cfg, D_EPOCHS, D_TRAIN_FRAC)
        rung1[cfg['name']] = {'mpe': mpe, 'seconds': secs, 'parameters': par, **{
            k: (str(v) if not isinstance(v, (int, float, bool, str)) else v)
            for k, v in cfg.items() if k != 'name'}}
        print(f'    {cfg["name"]:<44} MPE {mpe:.6f}  ({secs / 60:.1f} min)')
    promote = sorted(rung1, key=lambda n: rung1[n]['mpe'])[:D_PROMOTE_TOP]
    print(f'    promoting to {D_PROMOTE_EPOCH} epochs: {promote}')
    rung2 = {}
    for name in promote:
        cfg = next(c for c in D_CONFIGS if c['name'] == name)
        mpe, secs, par = train_diffusion(cfg, D_PROMOTE_EPOCH, 1.0)
        rung2[name] = {'mpe': mpe, 'seconds': secs, 'parameters': par}
        print(f'    {name:<44} MPE {mpe:.6f}  ({secs / 60:.1f} min, full budget)')
    results['D_ablations'] = {'screening': rung1, 'promoted': rung2,
                              'screening_epochs': D_EPOCHS, 'screening_fraction': D_TRAIN_FRAC,
                              'promoted_epochs': D_PROMOTE_EPOCH}
else:
    print('\n  STUDY D — skipped (RUN_D = False). Set it True to run the diffusion '
          'ablations;\n    the budget is printed before training starts.')

# ════════════════════════════════════════════════════════════════════
# OUTPUT
# ════════════════════════════════════════════════════════════════════
json.dump({'config': {'backbone_epochs': BACKBONE_EPOCHS, 'seed': SEED,
                      'a_subset': A_SUBSET, 'a_steps': A_STEPS, 'a_samples': A_SAMPLES,
                      'b_windows': B_WINDOWS, 'd_epochs': D_EPOCHS,
                      'd_fraction': D_TRAIN_FRAC, 'val_windows': M_va},
           'results': results},
          open(f'{MODULE8_DIR}/module8_hpo.json', 'w'), indent=2)
print(f'\n  Results written : {MODULE8_DIR}/module8_hpo.json')

n_panels = sum([RUN_A, RUN_B, RUN_C])
if n_panels:
    fig, axes = plt.subplots(1, max(n_panels, 2), figsize=(6.5 * max(n_panels, 2), 5))
    ax_i = 0
    if RUN_A:
        ax = axes[ax_i]; ax_i += 1
        g = results['A_inference']
        for ns in A_SAMPLES:
            pts = sorted((v for v in g.values() if v['samples'] == ns),
                         key=lambda v: v['forward_passes'])
            ax.plot([p['forward_passes'] for p in pts], [p['mpe'] for p in pts],
                    marker='o', lw=1.3, label=f'{ns} sample(s)')
        kn = results['A_knee']
        ax.scatter([kn['forward_passes']], [kn['mpe']], s=140, facecolors='none',
                   edgecolors='crimson', lw=2, zorder=5, label='knee (within 1% of best)')
        ax.set_xscale('log'); ax.set_xlabel('forward passes per prediction (log)')
        ax.set_ylabel('Mean Point Error')
        ax.set_title('A — inference cost vs accuracy', fontweight='bold')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    if RUN_B:
        ax = axes[ax_i]; ax_i += 1
        ws = sorted(results['B_context'])
        ax.plot(ws, [results['B_context'][w]['mpe'] for w in ws], marker='o', color='#1F6F6B')
        ax.set_xticks(ws); ax.set_xlabel('context frames W'); ax.set_ylabel('Mean Point Error')
        ax.set_title('B — does more past help?', fontweight='bold'); ax.grid(alpha=0.3)
    if RUN_C:
        ax = axes[ax_i]; ax_i += 1
        s = results['C_optimisation']['factors']
        labs = sorted(s, key=lambda l: s[l]['spread_pct'])
        ax.barh(labs, [s[l]['spread_pct'] for l in labs], color='#9A3B2E')
        ax.set_xlabel('MPE spread across values tried (%)')
        ax.set_title('C — which knobs matter', fontweight='bold'); ax.grid(alpha=0.3, axis='x')
    for j in range(ax_i, len(axes)):
        axes[j].axis('off')
    plt.suptitle('Module 8: hyperparameter and optimisation study', fontsize=13,
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{MODULE8_DIR}/module8_hpo.png', dpi=120, bbox_inches='tight')
    plt.show()
    print(f'  Figure written  : {MODULE8_DIR}/module8_hpo.png')
print('=' * 72)
