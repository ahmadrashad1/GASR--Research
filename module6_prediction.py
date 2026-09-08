import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, os, math, json, time
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from tqdm import tqdm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ════════════════════════════════════════════════════════════════════
# MODULE 6 — FUTURE TISSUE GEOMETRY PREDICTION
#
#   Observed geometry (up to frame t)
#            |
#            v
#   Condition the trained diffusion model  (Module 5, EMA weights)
#            |
#            +--> A) DIRECT prediction   G_{t+k} = G_t + delta(context, k)
#            |         one shot, horizon-conditioned, k in the trained PRED_STEPS
#            |
#            +--> B) AUTOREGRESSIVE ROLLOUT
#            |         predict k=1, append the prediction to the context, slide, repeat
#            |         -- the only mechanism that reaches horizons beyond training
#            |
#            v
#   Multiple plausible predictions (stochastic DDIM, eta > 0)
#            |
#            v
#   Per-point uncertainty + PLY/NPY export + metrics
#
# This module trains NOTHING. It is the inference/application stage: it loads the
# Module 5 checkpoint and answers the question the project was posed around --
# "given what we have seen, what does the tissue do next, and how sure are we".
#
# ── Why this module re-derives the trajectories instead of reusing Module 4's windows ──
# Module 4 emits FIXED (context, target) pairs at horizons 1-3. That is exactly the right
# shape for training, and exactly the wrong shape for rollout: a rollout has to feed its
# OWN prediction back in as context, which means it needs the continuous per-frame
# trajectory, not pre-cut windows. So this module replays Module 4's per-video pipeline
# (same seed -> same FPS anchors, same adaptive smoothing window, same frame-0
# normalisation) to rebuild the identical normalised trajectory, then rolls out on it.
# Everything needed to replay it deterministically is recorded in module4_metadata.json;
# nothing is re-tuned here, so a mismatch is impossible by construction (and QT0 checks it
# against Module 4's own saved windows anyway).
#
# ── Honesty note, read this before reading any number below ──
# Module 5's QT2/QT5 currently do NOT beat the naive zero-motion baseline. Module 6 does
# not paper over that: every horizon reports model, naive AND constant-velocity side by
# side on identical windows, and the pass criteria are stated against both baselines. If
# the model loses, these tables are what say so, and QT4's rollout-vs-direct comparison is
# what says whether error accumulation or the single-step predictor is the cause.
# ════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════
COMBINED_DIR = f'{DRIVE_BASE}/module4_combined'
MODULE5_DIR  = f'{DRIVE_BASE}/module5'
MODULE6_DIR  = f'{DRIVE_BASE}/module6'
CKPT_BEST    = f'{MODULE5_DIR}/diffusion_ckpt_best.pth'
CKPT_LATEST  = f'{MODULE5_DIR}/diffusion_ckpt.pth'
os.makedirs(MODULE6_DIR, exist_ok=True)
os.makedirs(f'{MODULE6_DIR}/exports', exist_ok=True)

USE_BEST_CKPT = True     # CKPT_BEST is the best-validating EMA snapshot; CKPT_LATEST is
                          # the final-epoch one. Prediction quality is what matters here, so
                          # prefer BEST. Set False to characterise the end-of-run weights.

# ── Sampling budget ──────────────────────────────────────────────────
# Cost is (number of predictions) x n_samples x DDIM steps forward passes. Module 5's
# CONFIG measured ~18,000 window-evaluations ~= one training epoch of wall-clock (~100s on
# a T4), and that is the unit to budget in. The three studies below are sized so the whole
# module lands near ~20 minutes rather than "leave it overnight":
#   direct  : n_origins x len(PRED_STEPS) x SAMPLES x STEPS
#   rollout : n_rollout_origins x ROLLOUT_HORIZON x SAMPLES x STEPS
#   hypoth. : n_hyp_origins x N_HYPOTHESES x STEPS
DDIM_STEPS    = 20       # matches Module 5's FINAL_STEPS, so QT1 here is directly
                          # comparable to Module 5's QT2 rather than a different operating
                          # point. Module 5's QT4 sweep is what justifies 20.
DDIM_SAMPLES  = 4        # trajectories averaged into ONE point prediction. Chamfer Distance
                          # scores a single prediction and is minimised by the conditional
                          # MEAN, so averaging is not optional polish -- a lone draw pays a
                          # ~sqrt(2)x error penalty. Same reasoning as Module 5's VAL_SAMPLES.
EVAL_ROWS     = 64       # (origins x samples) resident at once. No backward pass, so this
                          # is far under the training memory ceiling. Drop to 32 on an OOM.

# ── A) Direct prediction study ───────────────────────────────────────
DIRECT_STRIDE = 2        # take every Nth valid origin in each video's held-out tail. 1 =
                          # every frame. 2 halves the cost with no loss of signal, since
                          # consecutive origins share 3 of their 4 context frames and are
                          # therefore highly correlated samples of the same quantity.
# Horizons come from module4_metadata.json ('pred_steps', normally [1,2,3]) -- NOT invented
# here. Deliberately not extended past them: MAX_PRED_STEP sizes the horizon embedding at 8,
# but rows 4..8 were never touched by a gradient, so a "direct k=5 prediction" would be
# reading a randomly-initialised embedding and reporting noise. Reaching beyond the trained
# horizons is precisely what the rollout below is for.

# ── B) Autoregressive rollout study ──────────────────────────────────
ROLLOUT_HORIZON  = 10    # frames ahead to roll out. Past ~3 this is genuine extrapolation
                          # beyond anything Module 5 was trained on, which is the point:
                          # QT2 measures how fast error accumulates when the model is fed
                          # its own output.
ROLLOUT_PER_VIDEO = 8    # rollout origins per video, evenly spread across the held-out
                          # tail. Rollouts are ROLLOUT_HORIZON x more expensive per origin
                          # than a direct prediction, hence far fewer of them.

# ── C) Multi-hypothesis study ────────────────────────────────────────
# "Multiple plausible predictions" is the deliverable the project brief names, and eta > 0
# is what produces it: with eta = 0 DDIM is deterministic and every draw is identical.
N_HYPOTHESES  = 5        # 5 plausible futures per origin, as specified in the M4/M5 doc
HYP_ETA       = 1.0      # 1.0 = ancestral/DDPM-like sampling, maximum diversity
HYP_PER_VIDEO = 6        # origins per video for the hypothesis study

# ── Export ───────────────────────────────────────────────────────────
EXPORT_PLY      = True   # per-frame predicted geometry as ASCII PLY, with the per-point
                          # hypothesis spread written as a scalar property so the
                          # uncertainty is visible in any point-cloud viewer
EXPORT_VIDEO_ID = 0      # which video's rollout to export in full (index into video_order)
SEED            = 2026

torch.manual_seed(SEED)
np.random.seed(SEED)

assert os.path.exists(f'{COMBINED_DIR}/module4_metadata.json'), (
    f'module4_metadata.json not found in {COMBINED_DIR} -- run Module 4 first.')
_ckpt_path = CKPT_BEST if (USE_BEST_CKPT and os.path.exists(CKPT_BEST)) else CKPT_LATEST
assert os.path.exists(_ckpt_path), (
    f'No Module 5 checkpoint found ({CKPT_BEST} / {CKPT_LATEST}) -- train Module 5 first.')

meta4 = json.load(open(f'{COMBINED_DIR}/module4_metadata.json'))
N_ANCHOR    = meta4['n_anchor']
WINDOW_SIZE = meta4['window_size']
PRED_STEPS  = meta4['pred_steps']
M4_SEED     = meta4['seed']
POLYORDER   = meta4['smooth_polyorder']
video_order = meta4['video_order']

print('=' * 64)
print('  MODULE 6 — Future Tissue Geometry Prediction')
print('=' * 64)
print(f'  Device          : {DEVICE}')
print(f'  Checkpoint      : {os.path.basename(_ckpt_path)}')
print(f'  Videos          : {len(video_order)}')
print(f'  Anchors / frame : {N_ANCHOR}   |  context window W = {WINDOW_SIZE}')
print(f'  Trained horizons: {PRED_STEPS}  (direct prediction is limited to these)')
print(f'  Rollout horizon : {ROLLOUT_HORIZON} frames  (autoregressive, exceeds training)')
print(f'  Hypotheses      : {N_HYPOTHESES} per origin at eta={HYP_ETA}')


# ════════════════════════════════════════════════════════════════════
# MODEL — identical definitions to module5_diffusion.py. Dimensions are
# constructor arguments read from the checkpoint's own 'arch_cfg' rather
# than re-typed constants, so this module cannot silently disagree with
# whatever Module 5 actually trained.
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
    def __init__(self, window_size, latent_dim, hidden_dim, n_heads, n_tf_layers,
                 dropout, max_pred_step):
        super().__init__()
        D = latent_dim
        self.context_geo = GeomMLP(3, hidden_dim, D, dropout)
        self.target_geo  = GeomMLP(3, hidden_dim, D, dropout)
        self.temporal_pe = nn.Parameter(torch.randn(window_size, D) * 0.02)
        tf_layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=n_heads, dim_feedforward=hidden_dim,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.temporal_transformer = nn.TransformerEncoder(
            tf_layer, num_layers=n_tf_layers, enable_nested_tensor=False)
        self.time_mlp = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))
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
        h = h[:, -1, :]
        return h.reshape(B, N_A, -1)

    def forward(self, ctx, x_noisy, t, hstep):
        h_emb = self.horizon_mlp(self.horizon_emb(hstep))[:, None, :]
        ctx_latent = self.encode_context(ctx) + h_emb
        tgt_latent = self.target_geo(x_noisy)
        tgt_latent = tgt_latent + self.time_mlp(
            sinusoidal_embedding(t, tgt_latent.shape[-1]))[:, None, :]
        tgt_latent = tgt_latent + h_emb
        attn_t, _ = self.cross_t2c(tgt_latent, ctx_latent, ctx_latent)
        tgt_upd = self.norm_t(tgt_latent + attn_t)
        attn_c, _ = self.cross_c2t(ctx_latent, tgt_latent, tgt_latent)
        ctx_upd = self.norm_c(ctx_latent + attn_c)
        return self.output_mlp(torch.cat([tgt_upd, ctx_upd], dim=-1))


# ════════════════════════════════════════════════════════════════════
# DIFFUSION MATHS — same formulas as Module 5, parameterised per-checkpoint
# ════════════════════════════════════════════════════════════════════
def cosine_beta_schedule(T, s=0.008):
    steps = T + 1
    x = torch.linspace(0, T, steps)
    ac = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    ac = ac / ac[0]
    betas = 1 - (ac[1:] / ac[:-1])
    return torch.clip(betas, 1e-4, 0.999)


def pred_x0_from_v(x_t, v_pred, t, sqrt_ac, sqrt_1m_ac):
    return sqrt_ac[t][:, None, None] * x_t - sqrt_1m_ac[t][:, None, None] * v_pred


def pred_eps_from_v(x_t, v_pred, t, sqrt_ac, sqrt_1m_ac):
    return sqrt_1m_ac[t][:, None, None] * x_t + sqrt_ac[t][:, None, None] * v_pred


@torch.no_grad()
def ddim_sample(bundle, ctx, hstep, steps, eta=0.0, generator=None, n_samples=1):
    """Generalised DDIM. Returns predicted x0 (the displacement residual) in the model's
    SCALED space -- still needs /DATA_SCALE and /disp_norm to become a physical delta.

    n_samples > 1 runs that many independent trajectories per origin and returns their
    MEAN, a Monte-Carlo estimate of E[x0 | context, horizon]. With eta = 0 every trajectory
    from the same noise seed is identical, so averaging only helps because the INITIAL
    noise differs; that is exactly the variance being averaged out.
    """
    model = bundle['model']
    n_pts = ctx.shape[2]
    if n_samples > 1:
        ctx = ctx.repeat_interleave(n_samples, dim=0)
        hstep = hstep.repeat_interleave(n_samples, dim=0)
    n_rows = ctx.shape[0]
    ts = torch.linspace(bundle['T_diff'] - 1, 0, steps, device=DEVICE).long()
    x = torch.randn(n_rows, n_pts, 3, device=DEVICE, generator=generator)
    for i, t in enumerate(ts):
        t_batch = t.expand(n_rows)
        v_pred = model(ctx, x, t_batch, hstep)
        x0_pred = pred_x0_from_v(x, v_pred, t_batch, bundle['sqrt_ac'], bundle['sqrt_1m_ac'])
        if i == len(ts) - 1:
            x = x0_pred
            break
        eps_pred = pred_eps_from_v(x, v_pred, t_batch, bundle['sqrt_ac'], bundle['sqrt_1m_ac'])
        ac_t, ac_next = bundle['alphas_cumprod'][t], bundle['alphas_cumprod'][ts[i + 1]]
        sigma = eta * torch.sqrt((1 - ac_next) / (1 - ac_t)) * torch.sqrt(1 - ac_t / ac_next)
        x = (torch.sqrt(ac_next) * x0_pred
             + torch.sqrt(torch.clamp(1 - ac_next - sigma ** 2, min=0.0)) * eps_pred)
        if eta > 0:
            x = x + sigma * torch.randn(n_rows, n_pts, 3, device=DEVICE, generator=generator)
    if n_samples > 1:
        x = x.view(-1, n_samples, n_pts, 3).mean(dim=1)
    return x


def chamfer_distance(pred, gt):
    """Symmetric Chamfer Distance. pred, gt: (B, N, 3) -> (B,) per-window, NOT reduced.
    Per-window rather than Module 5's batch-mean because Module 6 needs to weight by
    horizon, video and rollout step separately, and a pre-averaged scalar cannot do that."""
    d = torch.cdist(pred, gt)
    return d.min(dim=2)[0].mean(dim=1) + d.min(dim=1)[0].mean(dim=1)


def load_bundle(ckpt_path):
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    arch = ck['arch_cfg']
    model = ConditionalDenoiser(
        window_size=arch['WINDOW_SIZE'], latent_dim=arch['LATENT_DIM'],
        hidden_dim=arch['HIDDEN_DIM'], n_heads=arch['N_HEADS'],
        n_tf_layers=arch['N_TF_LAYERS'], dropout=0.1,
        max_pred_step=arch.get('MAX_PRED_STEP', 8),
    ).to(DEVICE)
    # EMA weights, never the raw ones: EMA is the inference model throughout this project.
    model.load_state_dict(ck['ema'])
    model.eval()
    betas = cosine_beta_schedule(arch['T_DIFF']).to(DEVICE)
    ac = torch.cumprod(1.0 - betas, dim=0)
    return {
        'model': model, 'T_diff': arch['T_DIFF'], 'alphas_cumprod': ac,
        'sqrt_ac': torch.sqrt(ac), 'sqrt_1m_ac': torch.sqrt(1.0 - ac),
        'data_scale': ck['config']['DATA_SCALE'], 'epoch': ck.get('epoch', '?'),
        'max_pred_step': arch.get('MAX_PRED_STEP', 8), 'arch': arch,
        'best_cd': ck.get('best_cd', float('nan')),
    }


BUNDLE = load_bundle(_ckpt_path)
n_params = sum(p.numel() for p in BUNDLE['model'].parameters())
print(f'  Model           : {n_params:,} params  |  trained to epoch {BUNDLE["epoch"]}  '
      f'|  best val CD {BUNDLE["best_cd"]:.6f}')
print(f'  DATA_SCALE      : {BUNDLE["data_scale"]:.3f}   (inverted before every metric)')


# ════════════════════════════════════════════════════════════════════
# TRAJECTORY RECONSTRUCTION — replay Module 4's per-video pipeline
#
# Reproduces, bit for bit, the normalised anchor trajectory Module 4 built:
#   1. same anchors  -- FPS seeded with default_rng(M4_SEED + video_id), opacity-gated
#   2. same smoothing -- savgol at the per-video window Module 4 CHOSE adaptively and
#                        recorded in module4_metadata.json (not re-selected here: re-running
#                        the selection could pick a different window and silently shift the
#                        data under the model)
#   3. same normalisation -- centre/scale from frame 0 of the SMOOTHED trajectory
# QT0 below verifies the replay against Module 4's own saved windows before anything is
# predicted, so a drift in any of these three steps is caught immediately rather than
# showing up later as an unexplained accuracy loss.
# ════════════════════════════════════════════════════════════════════
def farthest_point_sampling(points, n_samples, rng):
    """Byte-identical to Module 4's implementation -- same greedy order, same rng draw."""
    n = len(points)
    sel = np.zeros(n_samples, dtype=np.int64)
    dists = np.full(n, np.inf, dtype=np.float32)
    sel[0] = rng.integers(0, n)
    for i in range(1, n_samples):
        d = np.sum((points - points[sel[i - 1]]) ** 2, axis=1)
        dists = np.minimum(dists, d)
        sel[i] = np.argmax(dists)
    return sel


def select_anchors(traj_full, ckpt_path, n_anchor, rng):
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        opa = torch.sigmoid(ck['opa']).squeeze(-1).numpy()
        vis = opa > 0.05
        pts_v = traj_full[0][vis]
        if len(pts_v) >= n_anchor:
            return np.where(vis)[0][farthest_point_sampling(pts_v, n_anchor, rng)]
    return farthest_point_sampling(traj_full[0], n_anchor, rng)


def rebuild_video(v):
    """v: one entry of meta4['videos']. Returns the replayed per-video state, or None if
    that video's trajectory is missing from Drive."""
    ws = v['workspace']
    traj_path = f'{DRIVE_BASE}/{ws}/exports/gaussian_trajectory.npy'
    if not os.path.exists(traj_path):
        return None
    traj_full = np.load(traj_path)
    rng = np.random.default_rng(M4_SEED + v['video_id'])
    anc = select_anchors(traj_full, f'{DRIVE_BASE}/{ws}/checkpoints/4dgs_v4.pth',
                         N_ANCHOR, rng)
    traj_raw = traj_full[:, anc, :]
    traj = savgol_filter(traj_raw, window_length=v['smooth_window'],
                         polyorder=POLYORDER, axis=0)
    ctr = traj[0].mean(axis=0)
    scl = np.abs(traj[0] - ctr).max() + 1e-8
    return {
        'video_id': v['video_id'], 'workspace': ws,
        'traj_n': torch.from_numpy(((traj - ctr) / scl).astype(np.float32)).to(DEVICE),
        'ctr': ctr, 'scl': float(scl), 'n_frames': traj.shape[0],
        # Physical delta = (model delta / DATA_SCALE) / disp_norm_factor. Module 4 scaled
        # each video's target delta by this factor to stop high-motion videos dominating
        # training; undoing it is what puts predictions back into this video's own space.
        'disp_norm': float(v['disp_norm_factor']),
        # Everything from here on is the held-out tail: Module 4 split each video temporally
        # 80/20, so no origin at or past this frame contributed a single gradient step.
        'split_frame': int(v['split_frame_pred1']),
        'psnr': v.get('psnr', 'n/a'),
    }


print('\nReplaying Module 4 per-video trajectories (anchors, smoothing, normalisation)...')
videos = []
for v in meta4['videos']:
    rb = rebuild_video(v)
    if rb is None:
        print(f'  ⚠️  {v["workspace"]}: gaussian_trajectory.npy missing — skipped')
        continue
    videos.append(rb)
    n_eval = max(0, rb['n_frames'] - max(PRED_STEPS) - rb['split_frame'] + 1)
    print(f'  ✅ {rb["workspace"]:<24} frames={rb["n_frames"]:<5} '
          f'held-out origins={n_eval:<5} smooth_win={v["smooth_window"]:<3} '
          f'disp_norm={rb["disp_norm"]:.3f}  PSNR={rb["psnr"]}')
assert videos, 'No video trajectories could be rebuilt -- check DRIVE_BASE.'


def to_physical(delta_scaled, disp_norm):
    """Model-space displacement -> this video's normalised-position space. Both of Module
    4/5's scale transforms are undone here, so EVERY metric below is computed in physical
    (normalised-position) units and is directly comparable to Module 5's QT2."""
    return (delta_scaled / BUNDLE['data_scale']) / disp_norm


def const_velocity_pred(traj_n, t, k):
    """last + (last - previous) * s for s = 1..k. Module 4's QT7 measured that after
    adaptive smoothing this two-line estimator beats naive by 30-57% on this exact data,
    which makes it -- not naive -- the bar a 15.8M-parameter diffusion model has to clear
    to have earned its place in the pipeline."""
    last, prev = traj_n[t - 1], traj_n[t - 2]
    steps = torch.arange(1, k + 1, device=traj_n.device, dtype=last.dtype)
    return last[None] + (last - prev)[None] * steps[:, None, None]


# ════════════════════════════════════════════════════════════════════
# QT0 — REPLAY VERIFICATION (runs before any prediction)
# Compares the rebuilt trajectory against Module 4's own saved val windows. If the anchor
# selection, smoothing window or normalisation had drifted by even one step, the contexts
# the model is about to be conditioned on would not be the contexts it was trained on, and
# every number in this module would be quietly measuring the wrong thing.
# ════════════════════════════════════════════════════════════════════
print('\n' + '=' * 64)
print('  QT0 — Trajectory replay verification')
print('=' * 64)
_m4_inp = np.load(f'{COMBINED_DIR}/val_inputs.npy')
_m4_vid = np.load(f'{COMBINED_DIR}/val_video_id.npy')
_m4_ps  = np.load(f'{COMBINED_DIR}/val_pred_step.npy')
_m4_tgt = np.load(f'{COMBINED_DIR}/val_targets.npy')

_worst = 0.0
_checked = 0
for vid in videos:
    sel = np.where((_m4_vid == vid['video_id']) & (_m4_ps == 1))[0]
    if len(sel) == 0:
        continue
    # Module 4's val windows for pred_step=1 start at frame `split_frame` and run forward,
    # so window j has origin t = split_frame + j. Check the first, middle and last.
    for j in (0, len(sel) // 2, len(sel) - 1):
        t = vid['split_frame'] + j
        if t + 1 > vid['n_frames']:
            continue
        mine = vid['traj_n'][t - WINDOW_SIZE:t].cpu().numpy()
        theirs = _m4_inp[sel[j]]
        _worst = max(_worst, float(np.abs(mine - theirs).max()))
        _checked += 1
qt0_pass = _checked > 0 and _worst < 1e-5
print(f'  Windows cross-checked      : {_checked}')
print(f'  Max abs deviation vs M4    : {_worst:.3e}')
if qt0_pass:
    print('  ✅ PASS — the replayed trajectory is identical to the one Module 5 trained on.')
else:
    print('  ⚠️  MISMATCH — the rebuilt contexts differ from Module 4 saved windows. Every '
          'number below would be measuring a different dataset than the model was trained '
          'on. Check that module4_metadata.json matches the .npy files in COMBINED_DIR.')


# ════════════════════════════════════════════════════════════════════
# PREDICTION ENGINE
# ════════════════════════════════════════════════════════════════════
@torch.no_grad()
def predict_direct(ctx, k, steps=DDIM_STEPS, n_samples=DDIM_SAMPLES, disp_norm=None,
                   eta=0.0, generator=None):
    """One-shot horizon-conditioned prediction. ctx: (Bn, W, N_A, 3) normalised positions,
    k: scalar horizon. Returns absolute predicted geometry (Bn, N_A, 3).

    This is the 'Condition Diffusion Model -> Sample Future Geometry G_{t+delta}' arrow of
    the project flowchart, and it is only valid for k in the trained PRED_STEPS -- see the
    DIRECT_STRIDE comment for why untrained horizons are not probed here."""
    hstep = torch.full((ctx.shape[0],), k, device=DEVICE, dtype=torch.long)
    x0 = ddim_sample(BUNDLE, ctx, hstep, steps=steps, eta=eta, generator=generator,
                     n_samples=n_samples)
    return ctx[:, -1] + to_physical(x0, disp_norm)


@torch.no_grad()
def rollout(ctx, horizon, steps=DDIM_STEPS, n_samples=DDIM_SAMPLES, disp_norm=None,
            eta=0.0, generator=None):
    """Autoregressive rollout. Repeatedly predicts ONE frame ahead (k=1, the horizon the
    model saw most often) and feeds the prediction back in as the newest context frame.

    Returns (Bn, horizon, N_A, 3). This is the only route to horizons past PRED_STEPS, and
    the reason it is reported separately from direct prediction is that it compounds: every
    step after the first is conditioned on the model's own output, so any per-step bias is
    integrated rather than averaged away. QT2 measures exactly that compounding."""
    work = ctx.clone()
    hstep = torch.full((ctx.shape[0],), 1, device=DEVICE, dtype=torch.long)
    out = []
    for _ in range(horizon):
        x0 = ddim_sample(BUNDLE, work, hstep, steps=steps, eta=eta, generator=generator,
                         n_samples=n_samples)
        nxt = work[:, -1] + to_physical(x0, disp_norm)
        out.append(nxt)
        work = torch.cat([work[:, 1:], nxt[:, None]], dim=1)
    return torch.stack(out, dim=1)


def batched(items, rows):
    for i in range(0, len(items), rows):
        yield items[i:i + rows]


def mean_of(d, key):
    """d[key] / d['n'], or NaN when no origin contributed. A video whose held-out tail is
    shorter than the horizon being measured legitimately contributes zero windows; that
    must read as 'not measured' in the table rather than crashing the report."""
    return d[key] / d['n'] if d['n'] else float('nan')


# ════════════════════════════════════════════════════════════════════
# QT1 — DIRECT SHORT-HORIZON PREDICTION ACCURACY
# "Short-horizon prediction accuracy" from the project's evaluation plan, on held-out
# frames only, with both baselines on identical windows.
# ════════════════════════════════════════════════════════════════════
print('\n' + '=' * 64)
print('  QT1 — Direct prediction accuracy (held-out frames, per horizon)')
print('=' * 64)

rows_per_batch = max(1, EVAL_ROWS // max(1, DDIM_SAMPLES))
direct = {k: {'model': 0.0, 'naive': 0.0, 'cv': 0.0, 'n': 0} for k in PRED_STEPS}
direct_by_video = {}

_t0 = time.time()
for vid in videos:
    tn, dn = vid['traj_n'], vid['disp_norm']
    per_v = {k: {'model': 0.0, 'naive': 0.0, 'cv': 0.0, 'n': 0} for k in PRED_STEPS}
    for k in PRED_STEPS:
        origins = list(range(max(WINDOW_SIZE, vid['split_frame']),
                             vid['n_frames'] - k + 1, DIRECT_STRIDE))
        for chunk in batched(origins, rows_per_batch):
            ctx = torch.stack([tn[t - WINDOW_SIZE:t] for t in chunk])
            gt  = torch.stack([tn[t + k - 1] for t in chunk])
            pred = predict_direct(ctx, k, disp_norm=dn)
            nv = torch.stack([tn[t - 1] for t in chunk])
            cv = torch.stack([const_velocity_pred(tn, t, k)[-1] for t in chunk])
            per_v[k]['model'] += chamfer_distance(pred, gt).sum().item()
            per_v[k]['naive'] += chamfer_distance(nv, gt).sum().item()
            per_v[k]['cv']    += chamfer_distance(cv, gt).sum().item()
            per_v[k]['n']     += len(chunk)
        for key in ('model', 'naive', 'cv', 'n'):
            direct[k][key] += per_v[k][key]
    direct_by_video[vid['workspace']] = per_v

print(f'  ({time.time() - _t0:.0f}s, {DDIM_SAMPLES}-sample mean at {DDIM_STEPS} DDIM steps, '
      f'eta=0)\n')
print(f'  {"k":>3} {"windows":>8} {"model CD":>11} {"naive CD":>11} {"const-vel":>11} '
      f'{"vs naive":>10} {"vs c-vel":>10}')
qt1_beats_naive, qt1_beats_cv = True, True
PS_OK = [k for k in PRED_STEPS if direct[k]['n'] > 0]
for k in PRED_STEPS:
    d = direct[k]
    if d['n'] == 0:
        print(f'  {k:>3} {0:>8}   (no held-out origin reaches this horizon — not measured)')
        continue
    m, nv, cv = mean_of(d, 'model'), mean_of(d, 'naive'), mean_of(d, 'cv')
    qt1_beats_naive = qt1_beats_naive and m < nv
    qt1_beats_cv = qt1_beats_cv and m < cv
    print(f'  {k:>3} {d["n"]:>8} {m:>11.6f} {nv:>11.6f} {cv:>11.6f} '
          f'{(nv - m) / max(nv, 1e-12) * 100:>9.1f}% {(cv - m) / max(cv, 1e-12) * 100:>9.1f}%')
qt1_pass = qt1_beats_naive and bool(PS_OK)
assert PS_OK, ('No held-out window reached any trained horizon -- every video tail is '
                'shorter than min(PRED_STEPS). Nothing can be evaluated.')
print()
if qt1_beats_cv:
    print('  ✅ PASS — beats BOTH the naive and constant-velocity baselines at every horizon.')
elif qt1_beats_naive:
    print('  ⚠️  PARTIAL — beats naive but not constant-velocity. The diffusion model is not '
          'yet earning its parameters over a two-line linear extrapolator.')
else:
    print('  ⚠️  FAIL — does not beat the naive zero-motion baseline at every horizon. '
          'Predictions below are still produced and exported, but they should be reported '
          'as a negative result: on this data the model has not learned a displacement '
          'that is better than assuming the tissue does not move.')


# ════════════════════════════════════════════════════════════════════
# QT2 — MULTI-STEP ROLLOUT ERROR ACCUMULATION
# "Multi-step rollout error accumulation" from the evaluation plan. The question is not
# only how large the error is, but how it GROWS: a model whose error grows linearly is
# integrating an unbiased per-step error, whereas super-linear growth means each step is
# biased in a consistent direction and the rollout is drifting off the manifold.
# ════════════════════════════════════════════════════════════════════
print('\n' + '=' * 64)
print(f'  QT2 — Autoregressive rollout to {ROLLOUT_HORIZON} frames (error accumulation)')
print('=' * 64)

roll = {s: {'model': 0.0, 'naive': 0.0, 'cv': 0.0, 'n': 0}
        for s in range(1, ROLLOUT_HORIZON + 1)}

_t0 = time.time()
for vid in videos:
    tn, dn = vid['traj_n'], vid['disp_norm']
    lo = max(WINDOW_SIZE, vid['split_frame'])
    hi = vid['n_frames'] - ROLLOUT_HORIZON
    if hi <= lo:
        print(f'  ⚠️  {vid["workspace"]}: held-out tail shorter than the rollout horizon — skipped')
        continue
    origins = np.linspace(lo, hi, min(ROLLOUT_PER_VIDEO, hi - lo + 1)).astype(int).tolist()
    for chunk in batched(origins, rows_per_batch):
        ctx = torch.stack([tn[t - WINDOW_SIZE:t] for t in chunk])
        seq = rollout(ctx, ROLLOUT_HORIZON, disp_norm=dn)           # (Bn, H, N_A, 3)
        for s in range(1, ROLLOUT_HORIZON + 1):
            gt = torch.stack([tn[t + s - 1] for t in chunk])
            nv = torch.stack([tn[t - 1] for t in chunk])
            cv = torch.stack([const_velocity_pred(tn, t, s)[-1] for t in chunk])
            roll[s]['model'] += chamfer_distance(seq[:, s - 1], gt).sum().item()
            roll[s]['naive'] += chamfer_distance(nv, gt).sum().item()
            roll[s]['cv']    += chamfer_distance(cv, gt).sum().item()
            roll[s]['n']     += len(chunk)

print(f'  ({time.time() - _t0:.0f}s, {ROLLOUT_PER_VIDEO} origins/video)\n')
print(f'  {"step":>5} {"windows":>8} {"rollout CD":>12} {"naive CD":>11} {"const-vel":>11} '
      f'{"vs naive":>10}')
for s in range(1, ROLLOUT_HORIZON + 1):
    d = roll[s]
    if d['n'] == 0:
        continue
    m, nv, cv = mean_of(d, 'model'), mean_of(d, 'naive'), mean_of(d, 'cv')
    mark = '  <- trained horizon' if s in PRED_STEPS else ''
    print(f'  {s:>5} {d["n"]:>8} {m:>12.6f} {nv:>11.6f} {cv:>11.6f} '
          f'{(nv - m) / max(nv, 1e-12) * 100:>9.1f}%{mark}')

# Growth exponent: fit log(CD) = a + b*log(step). b ~ 0.5 is diffusive (unbiased random
# walk), b ~ 1.0 is a constant directional bias being integrated, b > 1 is compounding
# drift. This single number says more about rollout stability than any absolute CD does.
_ss = [s for s in range(1, ROLLOUT_HORIZON + 1) if roll[s]['n'] > 0]
_cd = np.array([mean_of(roll[s], 'model') for s in _ss])   # _ss is pre-filtered to n > 0
_growth = float(np.polyfit(np.log(_ss), np.log(np.maximum(_cd, 1e-12)), 1)[0]) if len(_ss) > 2 else float('nan')
qt2_pass = bool(np.all(np.diff(_cd) >= -1e-9)) and _growth < 1.5
print(f'\n  Error growth exponent b (CD ~ step^b) : {_growth:.2f}')
print(f'    b~0.5 diffusive/unbiased | b~1.0 integrated constant bias | b>1.5 compounding drift')
if qt2_pass:
    print('  ✅ PASS — rollout error grows monotonically and sub-critically; no runaway drift.')
else:
    print('  ⚠️  Rollout is unstable (non-monotonic or b >= 1.5) — the model is being pulled '
          'off the data manifold by its own predictions. Prefer direct prediction at the '
          'trained horizons over long rollouts.')

# Direct vs rollout at the horizons where both are defined -- this is the diagnostic that
# separates "the single-step predictor is weak" from "the feedback loop is what hurts".
print('\n  Direct vs rollout at overlapping horizons:')
print(f'  {"k":>3} {"direct CD":>12} {"rollout CD":>12}  interpretation')
for k in PS_OK:
    if roll.get(k, {}).get('n', 0) == 0:
        continue
    dcd = mean_of(direct[k], 'model')
    rcd = mean_of(roll[k], 'model')
    note = ('rollout worse — feedback compounds error' if rcd > dcd * 1.05
            else 'rollout better — horizon conditioning is weaker than iterating k=1'
            if rcd < dcd * 0.95 else 'equivalent')
    print(f'  {k:>3} {dcd:>12.6f} {rcd:>12.6f}  {note}')


# ════════════════════════════════════════════════════════════════════
# QT3 — MULTI-HYPOTHESIS PREDICTION
# "Sample multiple futures / check diversity vs realism" and the project's stated
# risk-aware-planning motivation. eta > 0 makes DDIM stochastic, so N draws from the same
# context give N plausible futures; their spread is the model's own uncertainty estimate.
# ════════════════════════════════════════════════════════════════════
print('\n' + '=' * 64)
print(f'  QT3 — Multi-hypothesis prediction ({N_HYPOTHESES} futures/origin, eta={HYP_ETA})')
print('=' * 64)

hyp_spread, hyp_mean_cd, hyp_best_cd, hyp_cd_of_mean, hyp_naive = [], [], [], [], []
per_window_spread, per_window_err = [], []

_t0 = time.time()
for vid in videos:
    tn, dn = vid['traj_n'], vid['disp_norm']
    lo = max(WINDOW_SIZE, vid['split_frame'])
    hi = vid['n_frames'] - max(PRED_STEPS)
    if hi <= lo:
        continue
    origins = np.linspace(lo, hi, min(HYP_PER_VIDEO, hi - lo + 1)).astype(int).tolist()
    for chunk in batched(origins, rows_per_batch):
        ctx = torch.stack([tn[t - WINDOW_SIZE:t] for t in chunk])
        gt  = torch.stack([tn[t] for t in chunk])                       # horizon k = 1
        # Each hypothesis is a single stochastic trajectory (n_samples=1): averaging would
        # destroy the very diversity being measured.
        hyps = torch.stack([
            predict_direct(ctx, 1, n_samples=1, disp_norm=dn, eta=HYP_ETA,
                           generator=torch.Generator(device=DEVICE).manual_seed(SEED + h))
            for h in range(N_HYPOTHESES)], dim=0)                        # (H, Bn, N_A, 3)
        spread = hyps.std(dim=0).norm(dim=-1).mean(dim=-1)               # (Bn,) per window
        cds = torch.stack([chamfer_distance(hyps[h], gt) for h in range(N_HYPOTHESES)])
        hyp_spread.append(spread.cpu())
        hyp_mean_cd.append(cds.mean(dim=0).cpu())
        hyp_best_cd.append(cds.min(dim=0)[0].cpu())
        hyp_cd_of_mean.append(chamfer_distance(hyps.mean(dim=0), gt).cpu())
        hyp_naive.append(chamfer_distance(torch.stack([tn[t - 1] for t in chunk]), gt).cpu())
        per_window_spread.append(spread.cpu())
        per_window_err.append(cds.mean(dim=0).cpu())

HYP_RAN = len(hyp_spread) > 0
if HYP_RAN:
    hyp_spread = torch.cat(hyp_spread); hyp_mean_cd = torch.cat(hyp_mean_cd)
    hyp_best_cd = torch.cat(hyp_best_cd); hyp_cd_of_mean = torch.cat(hyp_cd_of_mean)
    hyp_naive = torch.cat(hyp_naive)
    _sp = torch.cat(per_window_spread).numpy(); _er = torch.cat(per_window_err).numpy()
    # Calibration: does the model know when it is wrong? A useful uncertainty estimate has
    # the spread rising where the error rises. Near-zero correlation means the spread is
    # decorative and must NOT be presented as a confidence signal for surgical planning.
    calib = (float(np.corrcoef(_sp, _er)[0, 1])
             if len(_sp) > 2 and _sp.std() > 0 else float('nan'))
    _mean_cd = hyp_mean_cd.mean().item()
    _best_cd = hyp_best_cd.mean().item()

    print(f'  ({time.time() - _t0:.0f}s, {len(hyp_spread)} origins)\n')
    print(f'  Mean hypothesis spread (per-point std) : {hyp_spread.mean():.6f}')
    print(f'  Mean per-hypothesis CD                 : {_mean_cd:.6f}')
    print(f'  Best-of-{N_HYPOTHESES} CD                          : {_best_cd:.6f}   '
          f'({(_mean_cd - _best_cd) / max(_mean_cd, 1e-12) * 100:+.1f}% vs an average single '
          f'hypothesis — the value of sampling more than one future)')
    print(f'  CD of the hypothesis MEAN              : {hyp_cd_of_mean.mean():.6f}')
    print(f'  Naive on the same origins              : {hyp_naive.mean():.6f}')
    print(f'  Spread-vs-error correlation            : {calib:.3f}')
    qt3_pass = hyp_spread.mean().item() > 1e-6
    if not qt3_pass:
        print('\n  ⚠️  Hypotheses are identical — stochastic sampling is not active; '
              'check that HYP_ETA > 0.')
    elif calib > 0.3:
        print('\n  ✅ PASS — hypotheses are diverse AND the spread tracks the error '
              '(usable as a confidence signal).')
    else:
        print('\n  ✅ PASS (diversity) / ⚠️  the spread does NOT track the error, so it is a '
              'measure of sampling noise rather than of predictive confidence. Report it as '
              'diversity, not as calibrated uncertainty.')
else:
    # No video had a held-out tail long enough. Reported as "not run" rather than silently
    # scoring zeros, which would otherwise read as "hypotheses are identical".
    print('  ⚠️  SKIPPED — no held-out origin was long enough for the hypothesis study.')
    hyp_spread = hyp_mean_cd = hyp_best_cd = hyp_cd_of_mean = hyp_naive = torch.zeros(1)
    _sp = _er = np.zeros(1)
    calib = float('nan')
    qt3_pass = False


# ════════════════════════════════════════════════════════════════════
# QT4 — TEMPORAL SMOOTHNESS & STABILITY OF THE PREDICTED SEQUENCE
# "Ensure no sudden geometry tearing" / "smoothness and stability metrics". Measured as
# mean discrete acceleration ||G_{s+1} - 2G_s + G_{s-1}|| along the rolled-out sequence,
# against the same quantity on the ground-truth sequence. A rollout far ABOVE the
# ground-truth value is jittering; far BELOW it is over-smoothed and has collapsed toward
# a static prediction -- both are failure modes and CD alone distinguishes neither.
# ════════════════════════════════════════════════════════════════════
print('\n' + '=' * 64)
print('  QT4 — Temporal smoothness / stability of predicted sequences')
print('=' * 64)


def mean_accel(seq):
    """seq: (Bn, S, N_A, 3) -> scalar mean |second difference| over time."""
    if seq.shape[1] < 3:
        return float('nan')
    return (seq[:, 2:] - 2 * seq[:, 1:-1] + seq[:, :-2]).norm(dim=-1).mean().item()


acc_pred, acc_gt, acc_cv, n_acc = 0.0, 0.0, 0.0, 0
for vid in videos:
    tn, dn = vid['traj_n'], vid['disp_norm']
    lo = max(WINDOW_SIZE, vid['split_frame'])
    hi = vid['n_frames'] - ROLLOUT_HORIZON
    if hi <= lo:
        continue
    origins = np.linspace(lo, hi, min(4, hi - lo + 1)).astype(int).tolist()
    ctx = torch.stack([tn[t - WINDOW_SIZE:t] for t in origins])
    seq = rollout(ctx, ROLLOUT_HORIZON, disp_norm=dn)
    gts = torch.stack([torch.stack([tn[t + s - 1] for s in range(1, ROLLOUT_HORIZON + 1)])
                       for t in origins])
    cvs = torch.stack([const_velocity_pred(tn, t, ROLLOUT_HORIZON) for t in origins])
    acc_pred += mean_accel(seq) * len(origins)
    acc_gt   += mean_accel(gts) * len(origins)
    acc_cv   += mean_accel(cvs) * len(origins)
    n_acc    += len(origins)

acc_pred /= max(n_acc, 1); acc_gt /= max(n_acc, 1); acc_cv /= max(n_acc, 1)
_ratio = acc_pred / max(acc_gt, 1e-12)
print(f'  Mean |acceleration|, model rollout   : {acc_pred:.6e}')
print(f'  Mean |acceleration|, ground truth    : {acc_gt:.6e}')
print(f'  Mean |acceleration|, constant velocity: {acc_cv:.6e}   (exactly 0 by construction)')
print(f'  Model / ground-truth ratio           : {_ratio:.2f}')
qt4_pass = 0.2 <= _ratio <= 5.0
if qt4_pass:
    print('  ✅ PASS — predicted motion is of a physically comparable smoothness to the '
          'reconstructed tissue; no tearing or popping.')
elif _ratio > 5.0:
    print('  ⚠️  Predicted sequences are far jerkier than the real tissue — geometry is '
          'tearing between rollout steps.')
else:
    print('  ⚠️  Predicted sequences are far smoother than the real tissue — the rollout has '
          'collapsed toward a near-static prediction (which is also why it can look '
          'competitive with the naive baseline on CD).')


# ════════════════════════════════════════════════════════════════════
# QT5 — PER-VIDEO BREAKDOWN (cross-scene generalisation)
# The project's global validation strategy asks for cross-scene validation. A single
# pooled number hides the case where the model works on two easy videos and fails on eight.
# ════════════════════════════════════════════════════════════════════
print('\n' + '=' * 64)
print('  QT5 — Per-video breakdown (direct prediction, all horizons pooled)')
print('=' * 64)
print(f'  {"video":<24} {"n":>6} {"model":>10} {"naive":>10} {"const-vel":>10} {"vs naive":>10}')
qt5_wins = 0
qt5_total = 0
for ws, per_v in direct_by_video.items():
    n = sum(per_v[k]['n'] for k in PRED_STEPS)
    if n == 0:
        continue
    m  = sum(per_v[k]['model'] for k in PRED_STEPS) / n
    nv = sum(per_v[k]['naive'] for k in PRED_STEPS) / n
    cv = sum(per_v[k]['cv'] for k in PRED_STEPS) / n
    qt5_total += 1
    qt5_wins += int(m < nv)
    print(f'  {ws:<24} {n:>6} {m:>10.6f} {nv:>10.6f} {cv:>10.6f} '
          f'{(nv - m) / max(nv, 1e-12) * 100:>9.1f}%')
qt5_pass = qt5_total > 0 and qt5_wins == qt5_total
print(f'\n  Videos where the model beats naive: {qt5_wins}/{qt5_total}')
print(f'  {"✅ PASS — generalises across every scene" if qt5_pass else "⚠️  Does not beat naive on every scene — cross-scene generalisation is incomplete"}')


# ════════════════════════════════════════════════════════════════════
# EXPORT — predicted future geometry
# The deliverable of this module is not only a table: it is predicted GEOMETRY that
# Module 7 and the qualitative side-by-side figures consume. Everything is written back in
# the ORIGINAL trajectory coordinate frame (undoing the per-video frame-0 centre/scale) so
# the exports overlay directly on Module 3's point clouds.
# ════════════════════════════════════════════════════════════════════
print('\n' + '=' * 64)
print('  EXPORT — predicted geometry')
print('=' * 64)


def write_ply(path, pts, scalar=None, scalar_name='uncertainty'):
    """ASCII PLY. `scalar` (per point) is written as an extra float property so the
    hypothesis spread can be colour-mapped in MeshLab/CloudCompare without a side file."""
    n = len(pts)
    head = ['ply', 'format ascii 1.0', f'element vertex {n}',
            'property float x', 'property float y', 'property float z']
    if scalar is not None:
        head.append(f'property float {scalar_name}')
    head += ['end_header']
    with open(path, 'w') as f:
        f.write('\n'.join(head) + '\n')
        for i in range(n):
            row = f'{pts[i, 0]:.6f} {pts[i, 1]:.6f} {pts[i, 2]:.6f}'
            if scalar is not None:
                row += f' {scalar[i]:.6f}'
            f.write(row + '\n')


export_summary = None
_exp = [v for v in videos if v['video_id'] == EXPORT_VIDEO_ID]
if _exp:
    vid = _exp[0]
    tn, dn = vid['traj_n'], vid['disp_norm']
    t0 = max(WINDOW_SIZE, vid['split_frame'])
    if t0 + ROLLOUT_HORIZON <= vid['n_frames']:
        ctx = tn[t0 - WINDOW_SIZE:t0][None]
        seq = rollout(ctx, ROLLOUT_HORIZON, disp_norm=dn)[0]            # (H, N_A, 3)
        gt = torch.stack([tn[t0 + s - 1] for s in range(1, ROLLOUT_HORIZON + 1)])
        # Per-point uncertainty from an independent hypothesis fan at the same origin.
        hyps = torch.stack([
            rollout(ctx, ROLLOUT_HORIZON, n_samples=1, disp_norm=dn, eta=HYP_ETA,
                    generator=torch.Generator(device=DEVICE).manual_seed(SEED + 100 + h))[0]
            for h in range(N_HYPOTHESES)], dim=0)                        # (H_hyp, H, N_A, 3)
        unc = hyps.std(dim=0).norm(dim=-1)                               # (H, N_A)

        # Back to the original (pre-normalisation) trajectory frame.
        def denorm(x):
            return (x.cpu().numpy() * vid['scl']) + vid['ctr']

        np.save(f'{MODULE6_DIR}/exports/pred_rollout_{vid["workspace"]}.npy', denorm(seq))
        np.save(f'{MODULE6_DIR}/exports/gt_rollout_{vid["workspace"]}.npy', denorm(gt))
        np.save(f'{MODULE6_DIR}/exports/uncertainty_{vid["workspace"]}.npy', unc.cpu().numpy())
        np.save(f'{MODULE6_DIR}/exports/hypotheses_{vid["workspace"]}.npy',
                np.stack([denorm(hyps[h]) for h in range(N_HYPOTHESES)]))
        if EXPORT_PLY:
            for s in range(ROLLOUT_HORIZON):
                write_ply(f'{MODULE6_DIR}/exports/pred_{vid["workspace"]}_f{s + 1:03d}.ply',
                          denorm(seq[s]), unc[s].cpu().numpy())
                write_ply(f'{MODULE6_DIR}/exports/gt_{vid["workspace"]}_f{s + 1:03d}.ply',
                          denorm(gt[s]))
        export_summary = {
            'workspace': vid['workspace'], 'origin_frame': int(t0),
            'horizon': ROLLOUT_HORIZON, 'n_hypotheses': N_HYPOTHESES,
            'mean_uncertainty': float(unc.mean()),
        }
        print(f'  Video            : {vid["workspace"]}  (origin frame {t0}, held-out)')
        print(f'  Rollout exported : {ROLLOUT_HORIZON} frames x {N_ANCHOR} points')
        print(f'  Hypotheses       : {N_HYPOTHESES} independent futures')
        print(f'  Mean uncertainty : {unc.mean():.6f}  (per-point hypothesis std)')
        print(f'  Files            : {MODULE6_DIR}/exports/'
              + (f'  ({2 * ROLLOUT_HORIZON} PLY + 4 NPY)' if EXPORT_PLY else '  (4 NPY)'))
    else:
        print(f'  ⚠️  {vid["workspace"]}: held-out tail too short to export a full rollout.')
else:
    print(f'  ⚠️  EXPORT_VIDEO_ID={EXPORT_VIDEO_ID} was not among the rebuilt videos.')


# ════════════════════════════════════════════════════════════════════
# METRICS FILE — everything Module 7 needs, without re-running any of this
# ════════════════════════════════════════════════════════════════════
metrics = {
    'checkpoint': os.path.basename(_ckpt_path),
    'checkpoint_epoch': BUNDLE['epoch'],
    'data_scale': BUNDLE['data_scale'],
    'ddim_steps': DDIM_STEPS, 'ddim_samples': DDIM_SAMPLES,
    'pred_steps': PRED_STEPS, 'rollout_horizon': ROLLOUT_HORIZON,
    'n_hypotheses': N_HYPOTHESES, 'hyp_eta': HYP_ETA,
    'qt0_replay_max_dev': _worst,
    'direct': {str(k): {kk: (mean_of(direct[k], kk) if kk != 'n' else vv)
                        for kk, vv in direct[k].items()} for k in PS_OK},
    'rollout': {str(s): {kk: (mean_of(roll[s], kk) if kk != 'n' else vv)
                         for kk, vv in roll[s].items()}
                for s in range(1, ROLLOUT_HORIZON + 1) if roll[s]['n'] > 0},
    'rollout_growth_exponent': _growth,
    'hypotheses': {
        'mean_spread': float(hyp_spread.mean()),
        'mean_cd': float(hyp_mean_cd.mean()),
        'best_of_n_cd': float(hyp_best_cd.mean()),
        'cd_of_mean': float(hyp_cd_of_mean.mean()),
        'naive_cd': float(hyp_naive.mean()),
        'spread_error_correlation': calib,
    },
    'smoothness': {'model': acc_pred, 'ground_truth': acc_gt,
                   'const_velocity': acc_cv, 'ratio': _ratio},
    'per_video': {ws: {str(k): {kk: (mean_of(pv[k], kk) if kk != 'n' else vv)
                                for kk, vv in pv[k].items()} for k in PRED_STEPS}
                  for ws, pv in direct_by_video.items()},
    'export': export_summary,
    'quality_tests': {'QT0': qt0_pass, 'QT1': qt1_pass, 'QT2': qt2_pass,
                      'QT3': qt3_pass, 'QT4': qt4_pass, 'QT5': qt5_pass},
}
json.dump(metrics, open(f'{MODULE6_DIR}/module6_metrics.json', 'w'), indent=2)
print(f'\n  Metrics written  : {MODULE6_DIR}/module6_metrics.json')


# ════════════════════════════════════════════════════════════════════
# VISUALISATION
# ════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(19, 9))

# (0,0) Direct prediction vs baselines, per horizon
ax = axes[0, 0]
ks = list(PS_OK)
ax.plot(ks, [mean_of(direct[k], 'model') for k in ks], 'o-', color='crimson',
        label='diffusion (direct)')
ax.plot(ks, [mean_of(direct[k], 'naive') for k in ks], 's--', color='gray',
        label='naive (zero motion)')
ax.plot(ks, [mean_of(direct[k], 'cv') for k in ks], '^--', color='steelblue',
        label='constant velocity')
ax.set_title('QT1 — Direct prediction accuracy', fontweight='bold')
ax.set_xlabel('horizon k (frames)'); ax.set_ylabel('Chamfer Distance')
ax.set_xticks(ks); ax.legend(fontsize=8); ax.grid(alpha=0.3)

# (0,1) Rollout error accumulation
ax = axes[0, 1]
ss = [s for s in range(1, ROLLOUT_HORIZON + 1) if roll[s]['n'] > 0]
ax.plot(ss, [mean_of(roll[s], 'model') for s in ss], 'o-', color='crimson',
        label='diffusion (rollout)')
ax.plot(ss, [mean_of(roll[s], 'naive') for s in ss], 's--', color='gray', label='naive')
ax.plot(ss, [mean_of(roll[s], 'cv') for s in ss], '^--', color='steelblue',
        label='constant velocity')
for k in PRED_STEPS:
    ax.axvline(k, color='seagreen', alpha=0.25, linewidth=1)
ax.set_title(f'QT2 — Rollout error accumulation (b={_growth:.2f})', fontweight='bold')
ax.set_xlabel('rollout step (green = trained horizons)'); ax.set_ylabel('Chamfer Distance')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# (0,2) Hypothesis spread vs error (calibration)
ax = axes[0, 2]
ax.scatter(_sp, _er, s=10, alpha=0.5, color='darkorange')
ax.set_title(f'QT3 — Hypothesis spread vs error (r={calib:.2f})', fontweight='bold')
ax.set_xlabel('per-window hypothesis spread'); ax.set_ylabel('mean per-hypothesis CD')
ax.grid(alpha=0.3)

# (1,0) Per-video improvement over naive
ax = axes[1, 0]
names, imps = [], []
for ws, pv in direct_by_video.items():
    n = sum(pv[k]['n'] for k in PRED_STEPS)
    if n == 0:
        continue
    m = sum(pv[k]['model'] for k in PRED_STEPS) / n
    nv = sum(pv[k]['naive'] for k in PRED_STEPS) / n
    names.append(ws.replace('workspace_', '')); imps.append((nv - m) / max(nv, 1e-12) * 100)
ax.barh(names, imps, color=['seagreen' if i > 0 else 'indianred' for i in imps])
ax.axvline(0, color='black', linewidth=1)
ax.set_title('QT5 — Improvement over naive, per video', fontweight='bold')
ax.set_xlabel('% better than naive'); ax.grid(alpha=0.3, axis='x')

# (1,1) Predicted vs ground truth geometry, XY projection
ax = axes[1, 1]
if export_summary is not None:
    _p = seq[0].cpu().numpy(); _g = gt[0].cpu().numpy(); _c = ctx[0, -1].cpu().numpy()
    ax.scatter(_c[:, 0], _c[:, 1], s=5, c='lightgray', alpha=0.7, label='last observed (naive)')
    ax.scatter(_g[:, 0], _g[:, 1], s=8, c='gold', marker='*', label='ground truth t+1')
    ax.scatter(_p[:, 0], _p[:, 1], s=5, c='crimson', alpha=0.7, label='predicted t+1')
    ax.set_aspect('equal'); ax.legend(fontsize=7)
ax.set_title('Predicted vs actual geometry (XY)', fontweight='bold'); ax.grid(alpha=0.3)

# (1,2) Uncertainty growth along the rollout
ax = axes[1, 2]
if export_summary is not None:
    _u = unc.cpu().numpy()
    ax.plot(range(1, ROLLOUT_HORIZON + 1), _u.mean(axis=1), 'o-', color='purple')
    ax.fill_between(range(1, ROLLOUT_HORIZON + 1),
                    np.percentile(_u, 10, axis=1), np.percentile(_u, 90, axis=1),
                    alpha=0.2, color='purple')
    ax.set_xlabel('rollout step'); ax.set_ylabel('per-point hypothesis std')
ax.set_title('Predictive uncertainty vs horizon', fontweight='bold'); ax.grid(alpha=0.3)

plt.suptitle(f'Module 6: Future Tissue Geometry Prediction  '
             f'({len(videos)} videos, checkpoint epoch {BUNDLE["epoch"]})',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{MODULE6_DIR}/module6_qa.png', dpi=120, bbox_inches='tight')
plt.show()


# ════════════════════════════════════════════════════════════════════
# STATUS
# ════════════════════════════════════════════════════════════════════
_qt = metrics['quality_tests']
_passed = sum(1 for v in _qt.values() if v)
print('\n' + '=' * 64)
print('  MODULE 6 STATUS')
print('=' * 64)
print(f'  Quality tests    : {_passed}/{len(_qt)} passed  '
      + '  '.join(f'{k}:{"PASS" if v else "FAIL"}' for k, v in _qt.items()))
print(f'  Direct CD (k={PS_OK[0]})  : {mean_of(direct[PS_OK[0]], "model"):.6f}'
      f'   naive {mean_of(direct[PS_OK[0]], "naive"):.6f}')
print(f'  Rollout growth b : {_growth:.2f}')
print(f'  Exports          : {MODULE6_DIR}/exports/')
print(f'  Metrics          : {MODULE6_DIR}/module6_metrics.json')
print(f'  Figure           : {MODULE6_DIR}/module6_qa.png')
if not qt1_pass:
    print('\n  → Reporting note: the model does not beat the naive zero-motion baseline at '
          'every horizon, so Module 6 is a NEGATIVE result on predictive accuracy. That is '
          'a publishable finding for this dataset, not a bug in this module: QT0 confirms '
          'the evaluation is on the correct held-out data, and QT1/QT5 localise the failure '
          'by horizon and by video. The mechanism, exports and uncertainty estimates are '
          'all in place, so re-running this cell against an improved Module 5 checkpoint '
          'reproduces every number without any code change.')
print('  → Next: Module 7 consumes module6_metrics.json and exports/ for the final '
      'Chamfer/PSNR/SSIM evaluation and the qualitative side-by-side figures.')
