import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, os, math, json, time, glob, shutil
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy import stats
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

# ── Magnitude calibration ────────────────────────────────────────────
CALIBRATE        = True   # fit one scalar per horizon on TRAINING origins (see the
                           # CALIBRATION section for why this is provable, not hopeful)
APPLY_CALIBRATION = True  # use it for the headline numbers. Both calibrated and
                           # uncalibrated are always reported, so nothing is hidden either way
CALIB_PER_VIDEO  = 24     # training origins per video per horizon used for the fit
CALIB_MAX        = 3.0    # clamp, so a degenerate fit cannot blow predictions up

# ── A) Direct prediction study ───────────────────────────────────────
DIRECT_STRIDE = 1        # take every Nth valid origin in each video's held-out tail. Was 2,
                          # which is fine for the AVERAGES (consecutive origins share 3 of 4
                          # context frames). It is not fine for the error timeline below: a
                          # frame dropped between the last observed frame and the target
                          # touches exactly ONE k=1 origin, so stride 2 hides half of those
                          # spikes. QT1 records every origin and the timeline reuses them,
                          # so the model is never run twice for the same window.
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

# ── Error timeline ───────────────────────────────────────────────────
TIMELINE_HORIZON = 1     # horizon plotted along each video (all horizons go to the JSON)
TIMELINE_SPIKE_Z = 3.5   # a spike is an origin more than this many robust SDs (median +
                          # z x 1.4826 x MAD) above that video's own typical error
TIMELINE_TOP_FRAC = 0.10 # "concentration" = share of the model's total excess error over
                          # naive carried by its worst 10% of origins. Evenly spread excess
                          # gives ~10-20%; a handful of bursts pushes it past 50%.

# Module 1's frame selection, needed to recover when each trajectory frame was actually
# filmed. Module 1 saves kept frames as 00000.png, 00001.png... -- a running counter that
# discards the raw video index -- and drops any frame blurrier than the threshold. So two
# consecutive trajectory frames can be 2 raw frames apart or 10. These MUST match Module 1
# (cell 9, and the identical copies embedded in the multi-video loops); the recovery below
# verifies itself pixel-for-pixel against the saved PNGs, so a mismatch is reported rather
# than silently producing wrong timings.
M1_FRAME_STRIDE = 2
M1_TARGET_WH    = (640, 360)
M1_SHARPNESS    = 30.0
TIMING_THUMB    = (64, 36)   # thumbnail used to match decoded frames to the saved PNGs
TIMING_EXACT_TOL = 0.75      # mean |grey diff| on a thumbnail for a replayed frame to count
                              # as identical (absorbs codec rounding across OpenCV builds)
TIMING_MATCH_TOL = 4.0       # looser bound for the image-matching fallback
TIMING_SEARCH    = 40        # fallback search window, in stride-sampled candidate frames
# Workspace -> source video, mirroring the VIDEOS (cell 0) and VIDEOS_EXTRA (videos 7-10)
# registries. When those lists exist in the running notebook they take precedence.
VIDEO_FILES = {
    'workspace_v2': 'Video01.mp4',     'workspace_video2': 'Video02.mp4',
    'workspace_video3': 'Video03.mp4', 'workspace_video4': 'video04.mp4',
    'workspace_video5': 'Video05.mp4', 'workspace_video6': 'Video06.mp4',
    'workspace_video7': 'Video07.mp4', 'workspace_video8': 'Video08.mp4',
    'workspace_video9': 'Video09.mp4', 'workspace_video10': 'Video10.mp4',
}
for _vn, _wn in list(globals().get('VIDEOS', [])) + list(globals().get('VIDEOS_EXTRA', [])):
    VIDEO_FILES[_wn] = _vn

# ── Visual inference (model creating frames + side-by-side with the real video) ──
VIS_ENABLED        = True
VIS_VIDEO_ID       = EXPORT_VIDEO_ID
VIS_RENDER         = True   # render predicted RGB frames through the video's own 4DGS scene.
                             # Needs gsplat + that video's checkpoints/4dgs_v4.pth; without
                             # either, it falls back to motion overlays on the real frames.
VIS_KNN            = 8      # anchors blended per Gaussian when spreading the 512 predicted
                             # anchor displacements to the full scene (inverse-distance weights)
VIS_ARROW_PX       = 18     # target on-screen length of a MEDIAN true-motion arrow. Real motion
                             # is ~1-2 px per frame, invisible at true scale, so arrows are
                             # magnified by one shared gain -- printed on every panel.
VIS_DENOISE_PANELS = 6      # DDIM steps shown in the denoising strip (the MP4 shows all)
VIS_GRID_STEPS     = (1, 2, 3, 5, 10)   # rollout steps shown as rows in the side-by-side PNG
VIS_FPS            = 2      # side-by-side MP4: few frames, so slow enough to actually compare
VIS_DENOISE_FPS    = 4
VIS_ERR_GAIN       = 4.0    # |predicted - actual| is amplified by this in the error panels

# ── Forecast past the end of the video ───────────────────────────────
# The side-by-side above pairs every predicted frame with a real one, which is what makes it
# measurable -- and also what makes it unconvincing to watch: nothing on screen is actually a
# forecast, and at 1-2 px of real motion per frame the two sides look identical.
# This runs the model off the END of the footage instead, producing frames for moments the
# camera never recorded. Those frames cannot be scored (there is no ground truth, by
# definition), so they are a demonstration rather than a measurement, and every panel says so.
FORECAST_FRAMES     = 12    # frames to predict beyond the last real frame
FORECAST_LEADIN     = 12    # real frames replayed first, so the handover is visible
FORECAST_EXAGGERATE = 8.0   # a second panel replays the prediction with motion multiplied by
                             # this, purely so a 1-2 px deformation is visible on screen.
                             # Always labelled; never used for any number.
FORECAST_GRID_STEPS = (1, 2, 4, 8, 12)
VIS_INLINE_VIDEO   = True   # embed both MP4s in the notebook output (Colab / Jupyter)
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
def ddim_sample(bundle, ctx, hstep, steps, eta=0.0, generator=None, n_samples=1, trace=None):
    """Generalised DDIM. Returns predicted x0 (the displacement residual) in the model's
    SCALED space -- still needs /DATA_SCALE and /disp_norm to become a physical delta.

    n_samples > 1 runs that many independent trajectories per origin and returns their
    MEAN, a Monte-Carlo estimate of E[x0 | context, horizon]. With eta = 0 every trajectory
    from the same noise seed is identical, so averaging only helps because the INITIAL
    noise differs; that is exactly the variance being averaged out.

    trace: optional list. When given, one (timestep, x_t, x0_estimate) tuple is appended per
    DDIM step -- this is what the visual-inference section draws. x_t is the FIRST sample's
    noisy state (averaging independent noise would shrink it and misrepresent what the
    network actually sees); x0_estimate is the mean over samples, so the last entry equals
    the returned prediction exactly.
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
        if trace is not None:
            trace.append((int(t),
                          x.view(-1, n_samples, n_pts, 3)[:, 0].clone(),
                          x0_pred.view(-1, n_samples, n_pts, 3).mean(dim=1).clone()))
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
        # 'delta' (model predicts displacement) or 'cv_residual' (model predicts the
        # correction to constant velocity). Read from the checkpoint, never assumed: a
        # cv_residual checkpoint reconstructed as a delta one is wrong in every number.
        'target_mode': arch.get('TARGET_MODE', 'delta'),
        'max_pred_step': arch.get('MAX_PRED_STEP', 8), 'arch': arch,
        'best_cd': ck.get('best_cd', float('nan')),
    }


BUNDLE = load_bundle(_ckpt_path)
n_params = sum(p.numel() for p in BUNDLE['model'].parameters())
print(f'  Model           : {n_params:,} params  |  trained to epoch {BUNDLE["epoch"]}  '
      f'|  best val CD {BUNDLE["best_cd"]:.6f}')
print(f'  DATA_SCALE      : {BUNDLE["data_scale"]:.3f}   (inverted before every metric)')
print(f'  Target mode     : {BUNDLE["target_mode"]}' + ('   (network output is the correction '
      'to constant velocity)' if BUNDLE['target_mode'] == 'cv_residual' else
      '   (network output is the displacement itself)'))


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
        # Indices of the 512 anchors among ALL of this video's Gaussians. Kept so the visual
        # section can spread predicted anchor motion back out to the full 4DGS scene.
        'anc': anc,
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


def base_pred(ctx, hstep):
    """What the scene looks like when the network emits exactly zero -- the prediction its
    output is a correction TO. Mirrors module5_diffusion.base_pred, driven by the mode
    recorded in the checkpoint so the two can never disagree."""
    if BUNDLE['target_mode'] == 'cv_residual':
        last, prev = ctx[:, -1], ctx[:, -2]
        return last + (last - prev) * hstep.to(last.dtype)[:, None, None]
    return ctx[:, -1]


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
                   eta=0.0, generator=None, alpha=1.0):
    """One-shot horizon-conditioned prediction. ctx: (Bn, W, N_A, 3) normalised positions,
    k: scalar horizon. Returns absolute predicted geometry (Bn, N_A, 3).

    This is the 'Condition Diffusion Model -> Sample Future Geometry G_{t+delta}' arrow of
    the project flowchart, and it is only valid for k in the trained PRED_STEPS -- see the
    DIRECT_STRIDE comment for why untrained horizons are not probed here."""
    hstep = torch.full((ctx.shape[0],), k, device=DEVICE, dtype=torch.long)
    x0 = ddim_sample(BUNDLE, ctx, hstep, steps=steps, eta=eta, generator=generator,
                     n_samples=n_samples)
    return base_pred(ctx, hstep) + alpha * to_physical(x0, disp_norm)


@torch.no_grad()
def rollout(ctx, horizon, steps=DDIM_STEPS, n_samples=DDIM_SAMPLES, disp_norm=None,
            eta=0.0, generator=None, alpha=1.0):
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
        nxt = base_pred(work, hstep) + alpha * to_physical(x0, disp_norm)
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
# CALIBRATION — one scalar per horizon, fitted on TRAINING origins
#
# Module 6's own measurements say the model's predicted displacement points roughly the
# right way (cosine +0.40 two frames ahead) while still losing to naive on Chamfer
# Distance. Those two facts together mean the DIRECTION carries signal and the MAGNITUDE
# does not -- the model over- or under-shoots.
#
# That is worth fixing at inference, because it is provable rather than hopeful. Writing
# the model's contribution as m and the true displacement as d, the best scalar multiple
# of m is alpha* = <m, d> / <m, m>, and using it leaves squared error |d|^2 (1 - cos^2).
# Since naive scores |d|^2, ANY non-zero cosine beats naive once the magnitude is
# calibrated. At cos = 0.40 that is roughly an 8% reduction in RMS error.
#
# Fitted on origins BEFORE each video's train/held-out split, never on the held-out tail
# every other number here is measured on. That matters: alpha fitted on the evaluation
# frames would be fitting the answer, and the improvement it bought would be fictional.
# ════════════════════════════════════════════════════════════════════
print('\n' + '=' * 64)
print('  CALIBRATION — magnitude scaling fitted on training origins')
print('=' * 64)

alpha = {k: 1.0 for k in PRED_STEPS}
calib_report = {}
rows_per_batch = max(1, EVAL_ROWS // max(1, DDIM_SAMPLES))
if CALIBRATE:
    _t0 = time.time()
    for k in PRED_STEPS:
        num = den = 0.0
        cos_s = rat_s = n_w = 0.0
        for vid in videos:
            tn, dn = vid['traj_n'], vid['disp_norm']
            hi = vid['split_frame'] - k          # last origin whose TARGET is still train-side
            lo = WINDOW_SIZE
            if hi <= lo:
                continue
            origins = np.linspace(lo, hi, min(CALIB_PER_VIDEO, hi - lo + 1)).astype(int).tolist()
            for chunk in batched(origins, rows_per_batch):
                ctx = torch.stack([tn[t - WINDOW_SIZE:t] for t in chunk])
                gt = torch.stack([tn[t + k - 1] for t in chunk])
                hstep = torch.full((len(chunk),), k, device=DEVICE, dtype=torch.long)
                x0 = ddim_sample(BUNDLE, ctx, hstep, steps=DDIM_STEPS, eta=0.0,
                                 n_samples=DDIM_SAMPLES)
                base = base_pred(ctx, hstep)
                m = to_physical(x0, dn)              # the model's own contribution
                resid = gt - base                    # what that contribution should have been
                num += float((m * resid).sum())
                den += float((m * m).sum())
                d_pred, d_true = base + m - ctx[:, -1], gt - ctx[:, -1]
                cos_s += float(F.cosine_similarity(d_pred, d_true, dim=-1).mean(dim=-1).sum())
                rat_s += float((d_pred.norm(dim=-1).sum(dim=-1)
                                / d_true.norm(dim=-1).sum(dim=-1).clamp_min(1e-12)).sum())
                n_w += len(chunk)
        a = float(num / den) if den > 0 else 1.0
        alpha[k] = float(np.clip(a, 0.0, CALIB_MAX))
        calib_report[k] = {'alpha_raw': a, 'alpha': alpha[k], 'n_windows': int(n_w),
                           'train_cos': cos_s / max(n_w, 1), 'train_ratio': rat_s / max(n_w, 1)}
    print(f'  ({time.time() - _t0:.0f}s on {CALIB_PER_VIDEO} origins per video per horizon)\n')
    print(f'  {"k":>3} {"windows":>8} {"alpha":>7} {"cos (train)":>12} {"|pred|/|true|":>14}')
    for k in PRED_STEPS:
        c = calib_report.get(k)
        if c:
            print(f'  {k:>3} {c["n_windows"]:>8} {c["alpha"]:>7.3f} {c["train_cos"]:>12.3f} '
                  f'{c["train_ratio"]:>14.3f}')
    _a = np.mean([c['alpha'] for c in calib_report.values()]) if calib_report else 1.0
    if _a < 0.9:
        print(f'\n  → alpha < 1 at every horizon: the model OVERSHOOTS. Scaling its output '
              f'down is expected to beat the uncalibrated prediction below.')
    elif _a > 1.1:
        print(f'\n  → alpha > 1: the model UNDERSHOOTS, predicting less motion than occurs.')
    else:
        print(f'\n  → alpha ~ 1: magnitude is already about right, so calibration will change '
              f'little and the error is in the direction, not the scale.')
    print(f'  Applied to the headline number: {APPLY_CALIBRATION}  '
          f'(both are reported either way)')
else:
    print('  (disabled — CALIBRATE = False)')


# ════════════════════════════════════════════════════════════════════
# QT1 — DIRECT SHORT-HORIZON PREDICTION ACCURACY
# "Short-horizon prediction accuracy" from the project's evaluation plan, on held-out
# frames only, with both baselines on identical windows.
# ════════════════════════════════════════════════════════════════════
print('\n' + '=' * 64)
print('  QT1 — Direct prediction accuracy (held-out frames, per horizon)')
print('=' * 64)

rows_per_batch = max(1, EVAL_ROWS // max(1, DDIM_SAMPLES))
# Rollout iterates the k=1 predictor, so k=1's calibration is the one that applies to
# every step of it.
ROLL_ALPHA = alpha.get(1, 1.0) if (CALIBRATE and APPLY_CALIBRATION) else 1.0
direct = {k: {'model': 0.0, 'model_cal': 0.0, 'naive': 0.0, 'cv': 0.0,
              'cos': 0.0, 'ratio': 0.0, 'n': 0} for k in PRED_STEPS}
direct_by_video = {}
# Every origin's individual CD, kept so the timeline section can show WHEN errors happen
# rather than only their average: direct_by_origin[workspace][k] = lists aligned by 't'.
direct_by_origin = {}

_t0 = time.time()
for vid in videos:
    tn, dn = vid['traj_n'], vid['disp_norm']
    per_v = {k: {'model': 0.0, 'model_cal': 0.0, 'naive': 0.0, 'cv': 0.0,
                 'cos': 0.0, 'ratio': 0.0, 'n': 0} for k in PRED_STEPS}
    per_o = {k: {'t': [], 'model': [], 'naive': [], 'cv': []} for k in PRED_STEPS}
    for k in PRED_STEPS:
        origins = list(range(max(WINDOW_SIZE, vid['split_frame']),
                             vid['n_frames'] - k + 1, DIRECT_STRIDE))
        for chunk in batched(origins, rows_per_batch):
            ctx = torch.stack([tn[t - WINDOW_SIZE:t] for t in chunk])
            gt  = torch.stack([tn[t + k - 1] for t in chunk])
            hstep = torch.full((len(chunk),), k, device=DEVICE, dtype=torch.long)
            # One sampler call, then both reconstructions -- calibration only rescales the
            # network's contribution, so it costs no extra forward passes.
            x0 = ddim_sample(BUNDLE, ctx, hstep, steps=DDIM_STEPS, eta=0.0,
                             n_samples=DDIM_SAMPLES)
            base = base_pred(ctx, hstep)
            m_contrib = to_physical(x0, dn)
            pred = base + m_contrib
            pred_cal = base + alpha[k] * m_contrib
            nv = torch.stack([tn[t - 1] for t in chunk])
            cv = torch.stack([const_velocity_pred(tn, t, k)[-1] for t in chunk])
            # Direction and magnitude of the predicted displacement against the true one --
            # the diagnostic that separates "learned nothing" from "right way, wrong size".
            d_pred, d_true = pred - nv, gt - nv
            per_v[k]['cos'] += float(F.cosine_similarity(d_pred, d_true, dim=-1)
                                     .mean(dim=-1).sum())
            per_v[k]['ratio'] += float((d_pred.norm(dim=-1).sum(dim=-1)
                                        / d_true.norm(dim=-1).sum(dim=-1).clamp_min(1e-12)).sum())
            per_v[k]['model_cal'] += chamfer_distance(pred_cal, gt).sum().item()
            cd_m = chamfer_distance(pred, gt)
            cd_n = chamfer_distance(nv, gt)
            cd_c = chamfer_distance(cv, gt)
            per_v[k]['model'] += cd_m.sum().item()
            per_v[k]['naive'] += cd_n.sum().item()
            per_v[k]['cv']    += cd_c.sum().item()
            per_v[k]['n']     += len(chunk)
            per_o[k]['t'].extend(int(t) for t in chunk)
            per_o[k]['model'].extend(cd_m.tolist())
            per_o[k]['naive'].extend(cd_n.tolist())
            per_o[k]['cv'].extend(cd_c.tolist())
        for key in ('model', 'model_cal', 'naive', 'cv', 'cos', 'ratio', 'n'):
            direct[k][key] += per_v[k][key]
    direct_by_video[vid['workspace']] = per_v
    direct_by_origin[vid['workspace']] = per_o

print(f'  ({time.time() - _t0:.0f}s, {DDIM_SAMPLES}-sample mean at {DDIM_STEPS} DDIM steps, '
      f'eta=0)\n')
print(f'  {"k":>3} {"windows":>8} {"model CD":>11} {"calibrated":>11} {"naive CD":>11} '
      f'{"const-vel":>11} {"vs naive":>10} {"cos":>7} {"|p|/|t|":>8}')
qt1_beats_naive, qt1_beats_cv = True, True
PS_OK = [k for k in PRED_STEPS if direct[k]['n'] > 0]
for k in PRED_STEPS:
    d = direct[k]
    if d['n'] == 0:
        print(f'  {k:>3} {0:>8}   (no held-out origin reaches this horizon — not measured)')
        continue
    m, nv, cv = mean_of(d, 'model'), mean_of(d, 'naive'), mean_of(d, 'cv')
    mc = mean_of(d, 'model_cal')
    headline = mc if APPLY_CALIBRATION else m
    qt1_beats_naive = qt1_beats_naive and headline < nv
    qt1_beats_cv = qt1_beats_cv and headline < cv
    print(f'  {k:>3} {d["n"]:>8} {m:>11.6f} {mc:>11.6f} {nv:>11.6f} {cv:>11.6f} '
          f'{(nv - headline) / max(nv, 1e-12) * 100:>9.1f}% {mean_of(d, "cos"):>7.3f} '
          f'{mean_of(d, "ratio"):>8.2f}')
qt1_pass = qt1_beats_naive and bool(PS_OK)
assert PS_OK, ('No held-out window reached any trained horizon -- every video tail is '
                'shorter than min(PRED_STEPS). Nothing can be evaluated.')
print(f'  columns: "model CD" = raw network output · "calibrated" = its magnitude scaled by '
      f'alpha fitted on TRAINING origins')
print(f'  "vs naive" scores the {"calibrated" if APPLY_CALIBRATION else "uncalibrated"} '
      f'prediction · cos and |p|/|t| = direction and size of the predicted displacement')
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
        seq = rollout(ctx, ROLLOUT_HORIZON, disp_norm=dn, alpha=ROLL_ALPHA)
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
            predict_direct(ctx, 1, n_samples=1, disp_norm=dn, eta=HYP_ETA, alpha=ROLL_ALPHA,
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
    seq = rollout(ctx, ROLLOUT_HORIZON, disp_norm=dn, alpha=ROLL_ALPHA)
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
# TIMELINE — WHERE along each video the model fails
#
# Every number above is an average over held-out origins, and an average cannot say WHEN
# the model fails. This section lays each origin's CD out along its video's held-out tail
# (model, naive and constant velocity on identical windows) to answer four questions the
# averages cannot:
#
#   1. SPIKES. Does error jump at particular moments -- instrument contact, a sudden tissue
#      pull, scope motion? A model spike is marked "shared" when naive spikes at the same
#      moment too: the tissue genuinely moved unusually and every predictor struggled.
#      "Model-only" spikes are failures specific to the model.
#   2. CONCENTRATION. Bad everywhere, or bad in a few bursts? Measured as the share of the
#      model's total excess error over naive that its worst 10% of origins carry. Evenly
#      spread points at the model or objective; bursts point at specific events in the
#      data. Those need very different fixes.
#   3. DROPPED FRAMES. Module 1 discards blurry frames, so consecutive trajectory frames are
#      NOT evenly spaced in real time, yet every module downstream treats them as if they
#      were. Frame timing is recovered below and tested against the error. Split two ways,
#      because they mean different things: a drop INSIDE the 4-frame context corrupts the
#      velocity the model infers; a drop between the last observed frame and the TARGET
#      just means more real time -- and therefore more motion -- passed than k implies.
#   4. DRIFT. Does error rise the further an origin sits from the training frames? Tested
#      on log(model CD / naive CD): naive CD tracks how much the tissue is moving at that
#      moment, so the ratio separates "the model got worse" from "the tissue got busier".
#
# Nothing new is predicted: QT1 already scored every origin (DIRECT_STRIDE = 1) and kept the
# per-origin values in direct_by_origin.
# ════════════════════════════════════════════════════════════════════
import cv2

print('\n' + '=' * 64)
print('  TIMELINE — error along each video\'s held-out tail')
print('=' * 64)

if TIMELINE_HORIZON not in PS_OK:
    print(f'  ⚠️  TIMELINE_HORIZON={TIMELINE_HORIZON} was not measured; using k={PS_OK[0]}.')
    TIMELINE_HORIZON = PS_OK[0]

TIMING_DIR = f'{MODULE6_DIR}/frame_timing'
os.makedirs(TIMING_DIR, exist_ok=True)


def recover_frame_timing(ws, n_frames):
    """Which raw video frame every trajectory frame came from, as {'ok': True, 'raw_index':
    [...], 'fps': ...} or {'ok': False, 'reason': ...}. Cached per workspace.

    First replays Module 1's own selection rule on the source video (every M1_FRAME_STRIDE-th
    frame, resized, kept if Laplacian sharpness >= M1_SHARPNESS) and accepts it only if
    every kept frame matches its saved PNG. If a different OpenCV/FFmpeg build flips a
    borderline keep/drop decision, falls back to matching each saved PNG to the nearest
    decoded frame, in order. Either way the result is checked against the pixels actually on
    Drive, so a timing that does not correspond to the saved frames is never returned."""
    cache = f'{TIMING_DIR}/{ws}.json'
    if os.path.exists(cache):
        c = json.load(open(cache))
        if c.get('ok') and c.get('n_frames') == n_frames:
            c['cached'] = True
            return c
    vname = VIDEO_FILES.get(ws)
    if vname is None:
        return {'ok': False, 'reason': f'no source video registered for {ws} (add to VIDEO_FILES)'}
    src = f'{DRIVE_BASE}/{vname}'
    if not os.path.exists(src):
        return {'ok': False, 'reason': f'{vname} not found in {DRIVE_BASE}'}
    frame_paths = sorted(glob.glob(f'{DRIVE_BASE}/{ws}/frames/*.png'))
    if len(frame_paths) != n_frames:
        return {'ok': False,
                'reason': f'{len(frame_paths)} frame PNGs but {n_frames} trajectory frames'}

    def thumb(gray):
        return cv2.resize(gray, TIMING_THUMB, interpolation=cv2.INTER_AREA)

    saved = np.stack([thumb(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY))
                      for p in frame_paths]).astype(np.int16)

    # OpenCV cannot read reliably from the Drive FUSE mount; Module 1 copies to /tmp as well.
    local = f'/tmp/module6_{vname}'
    if not os.path.exists(local):
        shutil.copy2(src, local)
    cap = cv2.VideoCapture(local)
    if not cap.isOpened():
        return {'ok': False, 'reason': f'OpenCV cannot open {local}'}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    raw, replay, cand_idx, cand = 0, [], [], []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if raw % M1_FRAME_STRIDE == 0:
            img = cv2.resize(frame, M1_TARGET_WH, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if cv2.Laplacian(gray, cv2.CV_64F).var() >= M1_SHARPNESS:
                replay.append(len(cand_idx))
            cand_idx.append(raw)
            cand.append(thumb(gray))
        raw += 1
    cap.release()
    if not cand:
        return {'ok': False, 'reason': f'OpenCV decoded 0 frames from {vname}'}
    cand = np.stack(cand).astype(np.int16)

    method, pick, res = None, None, []
    if len(replay) == n_frames:
        res = [float(np.abs(saved[i] - cand[j]).mean()) for i, j in enumerate(replay)]
        if max(res) <= TIMING_EXACT_TOL:
            method, pick = 'replayed Module 1 selection', replay
    if method is None:
        pick, res, j = [], [], 0
        for i in range(n_frames):
            hi = min(len(cand), j + TIMING_SEARCH)
            if j >= hi:
                break
            d = np.abs(cand[j:hi] - saved[i][None]).mean(axis=(1, 2))
            b = int(np.argmin(d))
            pick.append(j + b)
            res.append(float(d[b]))
            j += b + 1
        worst = max(res) if res else float('nan')
        if len(pick) < n_frames or not worst <= TIMING_MATCH_TOL:
            return {'ok': False,
                    'reason': (f'saved frames do not align with {vname} (matched '
                               f'{len(pick)}/{n_frames}, worst residual {worst:.2f})')}
        method = 'matched saved frames to the source video'
    out = {'ok': True, 'workspace': ws, 'video': vname, 'n_frames': int(n_frames), 'fps': fps,
           'raw_index': [int(cand_idx[j]) for j in pick], 'method': method,
           'max_residual': float(max(res)), 'n_raw_frames': int(raw), 'cached': False}
    json.dump(out, open(cache, 'w'), indent=1)
    return out


def dropped_between(raw_index, a, b):
    """Frames Module 1 discarded between trajectory frames a < b (0 when evenly spaced)."""
    return (raw_index[b] - raw_index[a]) // M1_FRAME_STRIDE - (b - a)


def robust_spikes(x, z):
    med = np.median(x)
    mad = 1.4826 * np.median(np.abs(x - med))
    return x > med + z * mad if mad > 0 else np.zeros(len(x), dtype=bool)


def centred_log(x):
    lx = np.log(np.maximum(x, 1e-12))
    return lx - np.median(lx)


def group_test(values, mask):
    """Median of `values` (centred logs) with vs without `mask`, and a Mann-Whitney p-value.
    None when either group is too small to say anything."""
    if mask.sum() < 3 or (~mask).sum() < 3:
        return None
    return {'n_with': int(mask.sum()), 'n_without': int((~mask).sum()),
            'factor': float(np.exp(np.median(values[mask]) - np.median(values[~mask]))),
            'p': float(stats.mannwhitneyu(values[mask], values[~mask],
                                          alternative='two-sided').pvalue)}


timeline, timing_by_ws = {}, {}
pool = {k: [] for k in ('dist', 'lr', 'ln', 'lm', 'lc', 'ctx', 'tgt')}
_t0 = time.time()
for vid in videos:
    ws = vid['workspace']
    o = direct_by_origin.get(ws, {}).get(TIMELINE_HORIZON)
    if o is None or len(o['t']) < 5:
        print(f'  ⚠️  {ws}: fewer than 5 held-out origins at k={TIMELINE_HORIZON} — skipped.')
        continue
    t = np.array(o['t'])
    m, nv, cv = np.array(o['model']), np.array(o['naive']), np.array(o['cv'])

    tim = recover_frame_timing(ws, vid['n_frames'])
    timing_by_ws[ws] = tim

    spike = robust_spikes(m, TIMELINE_SPIKE_Z)
    naive_spike = robust_spikes(nv, TIMELINE_SPIKE_Z)
    shared = spike & (np.convolve(naive_spike.astype(int), [1, 1, 1], mode='same') > 0)
    excess = np.maximum(m - nv, 0.0)
    n_top = max(1, int(math.ceil(TIMELINE_TOP_FRAC * len(m))))
    excess_share = (float(np.sort(excess)[::-1][:n_top].sum() / excess.sum())
                    if excess.sum() > 0 else float('nan'))
    lr = centred_log(m / np.maximum(nv, 1e-12))
    dist = t - vid['split_frame']
    if np.std(dist) > 0 and np.std(lr) > 0:
        rho, p_drift = stats.spearmanr(dist, lr)
        slope = float(np.polyfit(dist, lr, 1)[0])
    else:
        rho, p_drift, slope = float('nan'), float('nan'), float('nan')

    rec = {'t': t.tolist(), 'model': m.tolist(), 'naive': nv.tolist(), 'cv': cv.tolist(),
           'spike': spike.tolist(), 'spike_shared': shared.tolist(),
           'n_spikes': int(spike.sum()), 'n_shared': int(shared.sum()),
           'loss_frac': float((m >= nv).mean()), 'excess_share_top': excess_share,
           'drift_rho': float(rho), 'drift_p': float(p_drift),
           'drift_pct_per_10_frames': float((math.exp(slope * 10) - 1) * 100) if np.isfinite(slope) else float('nan'),
           'split_frame': int(vid['split_frame']), 'timing_ok': bool(tim['ok']),
           'timing_note': tim.get('method') if tim['ok'] else tim['reason']}
    if tim['ok']:
        r = tim['raw_index']
        ctx_d = np.array([dropped_between(r, tt - WINDOW_SIZE, tt - 1) for tt in t])
        tgt_d = np.array([dropped_between(r, tt - 1, tt + TIMELINE_HORIZON - 1) for tt in t])
        rec.update({'ctx_drops': ctx_d.tolist(), 'tgt_drops': tgt_d.tolist(),
                    'fps': tim['fps'], 'timing_residual': tim['max_residual']})
        pool['ctx'].append(ctx_d > 0)
        pool['tgt'].append(tgt_d > 0)
        pool['ln'].append(centred_log(nv))
        pool['lm'].append(centred_log(m))
        pool['lc'].append(centred_log(cv))
    pool['dist'].append(dist)
    pool['lr'].append(lr)
    timeline[ws] = rec

print(f'  ({time.time() - _t0:.0f}s incl. frame-timing recovery; k={TIMELINE_HORIZON}, every held-out origin)\n')
print(f'  {"video":<20} {"origins":>7} {"loses":>6} {"spikes":>9} {"top10%":>7} '
      f'{"drift rho":>10} {"ctx/tgt drops":>14}  frame timing')
for ws, T in timeline.items():
    drops = (f'{int(np.sum(np.array(T["ctx_drops"]) > 0))}/{int(np.sum(np.array(T["tgt_drops"]) > 0))}'
             if T['timing_ok'] else 'n/a')
    print(f'  {ws:<20} {len(T["t"]):>7} {T["loss_frac"] * 100:>5.0f}% '
          f'{T["n_spikes"]:>3} ({T["n_shared"]} sh) {T["excess_share_top"] * 100:>6.0f}% '
          f'{T["drift_rho"]:>+6.2f}{"*" if T["drift_p"] < 0.05 else " ":<4}'
          f'{drops:>14}  {T["timing_note"]}')
print('  loses = origins where model CD >= naive · spikes: "sh" = shared with naive · '
      'top10% = share of excess error in the worst 10% of origins · * p < 0.05')

# ── The four questions, pooled across videos ──
findings = {}
if timeline:
    print('\n  Findings:')
    n_sp = sum(T['n_spikes'] for T in timeline.values())
    n_sh = sum(T['n_shared'] for T in timeline.values())
    findings['spikes'] = {'total': n_sp, 'shared_with_naive': n_sh}
    if n_sp == 0:
        print('  1. Spikes         : none above the robust threshold — error has no sharp moments.')
    else:
        verdict = ('mostly moments where the tissue itself moved unusually (naive spikes too)'
                   if n_sh >= n_sp / 2 else
                   'mostly MODEL-ONLY — moments naive handles but the model does not')
        print(f'  1. Spikes         : {n_sp} across {len(timeline)} videos, {n_sh} shared with '
              f'naive → {verdict}.')

    loss = float(np.median([T['loss_frac'] for T in timeline.values()]))
    conc = float(np.nanmedian([T['excess_share_top'] for T in timeline.values()]))
    findings['concentration'] = {'median_loss_frac': loss, 'median_top_excess_share': conc}
    if loss >= 0.7:
        verdict = ('BAD EVERYWHERE — the model loses to naive at most moments. Points at the '
                   'model or objective, not at specific events in the data.')
    elif conc >= 0.5:
        verdict = ('CONCENTRATED IN BURSTS — a few moments carry most of the excess error. '
                   'Inspect those frames: the fix is in the data, not the model.')
    else:
        verdict = 'MIXED — neither uniformly bad nor dominated by a few bursts.'
    print(f'  2. Concentration  : loses at {loss * 100:.0f}% of origins (median video); worst '
          f'{TIMELINE_TOP_FRAC * 100:.0f}% carry {conc * 100:.0f}% of the excess error → {verdict}')

    if pool['ctx']:
        ctx_any, tgt_any = np.concatenate(pool['ctx']), np.concatenate(pool['tgt'])
        lr_t = np.concatenate([lr_ for lr_, T in zip(pool['lr'], timeline.values()) if T['timing_ok']])
        ln_t = np.concatenate(pool['ln'])
        findings['dropped_frames'] = {
            'origins_with_context_drop': int(ctx_any.sum()),
            'origins_with_target_drop': int(tgt_any.sum()),
            'origins_total': int(len(ctx_any)),
            'context_drop_vs_model_over_naive': group_test(lr_t, ctx_any),
            'target_drop_vs_model_over_naive': group_test(lr_t, tgt_any),
            'target_drop_vs_naive': group_test(ln_t, tgt_any),
            'context_drop_vs_model': group_test(np.concatenate(pool['lm']), ctx_any),
            'context_drop_vs_const_velocity': group_test(np.concatenate(pool['lc']), ctx_any),
        }
        fd = findings['dropped_frames']
        print(f'  3. Dropped frames : {fd["origins_with_context_drop"]} origins have a drop '
              f'inside the context, {fd["origins_with_target_drop"]} between last observed '
              f'frame and target (of {fd["origins_total"]}).')
        for label, key, meaning in (
                ('context drop → model / naive', 'context_drop_vs_model_over_naive',
                 'irregular spacing inside the context corrupts the velocity the MODEL infers'),
                ('context drop → const-velocity', 'context_drop_vs_const_velocity',
                 'the same corruption hits linear extrapolation'),
                ('target drop  → naive', 'target_drop_vs_naive',
                 'more real time passed, so the tissue moved further than k implies'),
                ('target drop  → model / naive', 'target_drop_vs_model_over_naive',
                 'the model copes with the extra elapsed time worse than naive does')):
            g = fd[key]
            if g is None:
                print(f'       {label:<30}: too few origins in one group to test')
                continue
            sig = g['p'] < 0.05 and g['factor'] > 1
            print(f'       {label:<30}: ×{g["factor"]:.2f} with drops (p={g["p"]:.3f}, '
                  f'n={g["n_with"]}/{g["n_without"]})' + (f'  ← {meaning}' if sig else ''))
        c_mn, c_cv = fd['context_drop_vs_model_over_naive'], fd['context_drop_vs_const_velocity']
        if c_mn and c_mn['p'] < 0.05 and c_mn['factor'] > 1:
            print('       → Irregular frame spacing IS corrupting per-frame motion. Fix upstream: '
                  'resample trajectories to uniform time, or feed real time gaps to the model.')
        elif (c_cv and c_cv['p'] < 0.05 and c_cv['factor'] > 1):
            print('       → Drops disrupt velocity (constant velocity suffers) but not the model '
                  'relative to naive — spacing is not what limits the model.')
        else:
            print('       → No measurable link between dropped frames and model error; the spikes '
                  'come from somewhere else.')
    else:
        reasons = sorted({T['timing_note'] for T in timeline.values()})
        findings['dropped_frames'] = {'unavailable': reasons}
        print(f'  3. Dropped frames : not testable — frame timing unavailable ({"; ".join(reasons)}).')

    dist_all, lr_all = np.concatenate(pool['dist']), np.concatenate(pool['lr'])
    if np.std(dist_all) > 0 and len(dist_all) >= 10:
        rho_all, p_all = stats.spearmanr(dist_all, lr_all)
    else:
        rho_all, p_all = float('nan'), float('nan')
    n_up = sum(1 for T in timeline.values() if T['drift_p'] < 0.05 and T['drift_rho'] > 0)
    findings['drift'] = {'pooled_rho': float(rho_all), 'pooled_p': float(p_all),
                         'videos_with_significant_rise': n_up}
    if p_all < 0.05 and rho_all > 0:
        verdict = ('ERROR RISES with distance from the training frames — the model generalises '
                   'worse the further it extrapolates in time.')
    elif p_all < 0.05 and rho_all < 0:
        verdict = 'error FALLS later in the tail — no sign of degradation away from training.'
    else:
        verdict = 'no drift — the model does not get worse further from its training frames.'
    print(f'  4. Drift          : Spearman rho={rho_all:+.2f} (p={p_all:.3f}) on model/naive vs '
          f'frames past the split; {n_up}/{len(timeline)} videos rise significantly → {verdict}')

    # ── Figure 1: one timeline per video ──
    _n = len(timeline)
    fig_t = plt.figure(figsize=(16, 3.4 * _n + 0.6))
    gs_t = fig_t.add_gridspec(2 * _n, 1, height_ratios=[3, 1] * _n, hspace=0.28)
    for row, (ws, T) in enumerate(timeline.items()):
        ax = fig_t.add_subplot(gs_t[2 * row])
        axg = fig_t.add_subplot(gs_t[2 * row + 1], sharex=ax)
        t = np.array(T['t'])
        ms = np.array(T['model'])
        sp, sh = np.array(T['spike']), np.array(T['spike_shared'])
        if T['timing_ok']:
            for tt, cd_, td in zip(t, T['ctx_drops'], T['tgt_drops']):
                if td > 0:
                    ax.axvspan(tt - 0.5, tt + 0.5, color='darkorange', alpha=0.28, lw=0)
                elif cd_ > 0:
                    ax.axvspan(tt - 0.5, tt + 0.5, color='gold', alpha=0.16, lw=0)
        ax.plot(t, T['naive'], color='gray', ls='--', lw=1.1, label='naive (zero motion)')
        ax.plot(t, T['cv'], color='steelblue', lw=1.0, alpha=0.85, label='constant velocity')
        ax.plot(t, ms, color='crimson', lw=1.5, label='diffusion model')
        ax.scatter(t[sp & ~sh], ms[sp & ~sh], marker='v', s=42, color='crimson', zorder=5,
                   label='model-only spike')
        ax.scatter(t[sp & sh], ms[sp & sh], marker='v', s=42, facecolors='none',
                   edgecolors='crimson', zorder=5, label='spike shared with naive')
        ax.set_ylabel('Chamfer Distance')
        ax.grid(alpha=0.3)
        ax.set_title(f'{ws}  ·  loses to naive at {T["loss_frac"] * 100:.0f}% of origins  ·  '
                     f'{T["n_spikes"]} spikes ({T["n_shared"]} shared)  ·  drift rho '
                     f'{T["drift_rho"]:+.2f}', fontsize=10, loc='left')
        plt.setp(ax.get_xticklabels(), visible=False)
        if row == 0:
            from matplotlib.patches import Patch
            h, l = ax.get_legend_handles_labels()
            h += [Patch(color='gold', alpha=0.35), Patch(color='darkorange', alpha=0.5)]
            l += ['frame dropped inside context', 'frame dropped before target']
            ax.legend(h, l, fontsize=8, ncol=4, loc='upper right')
        if T['timing_ok']:
            r = np.array(timing_by_ws[ws]['raw_index'])
            idx = np.arange(max(1, int(t.min()) - WINDOW_SIZE + 1),
                            min(len(r), int(t.max()) + TIMELINE_HORIZON))
            gaps = r[idx] - r[idx - 1]
            axg.bar(idx, gaps, width=0.8,
                    color=np.where(gaps > M1_FRAME_STRIDE, 'darkorange', 'lightgray'))
            axg.axhline(M1_FRAME_STRIDE, color='black', lw=0.8, ls=':')
            axg.set_ylabel('raw frames\nsince previous', fontsize=8)
            if T['fps'] > 0:
                _fps, _fr = T['fps'], np.arange(len(r))
                sec = ax.secondary_xaxis(
                    'top', functions=(lambda x, r=r, f=_fr, s=_fps: np.interp(x, f, r) / s,
                                      lambda y, r=r, f=_fr, s=_fps: np.interp(y * s, r, f)))
                sec.set_xlabel('video time (s)', fontsize=8)
                sec.tick_params(labelsize=8)
        else:
            axg.text(0.5, 0.5, f'frame timing unavailable: {T["timing_note"]}',
                     transform=axg.transAxes, ha='center', va='center', fontsize=9)
            axg.set_yticks([])
        axg.grid(alpha=0.3, axis='y')
        axg.set_xlabel(f'predicted frame t  (context = frames t-{WINDOW_SIZE}…t-1, '
                       f'horizon k={TIMELINE_HORIZON})', fontsize=8)
    fig_t.suptitle('Module 6: error along each held-out tail', fontsize=13, fontweight='bold', y=1.0)
    fig_t.savefig(f'{MODULE6_DIR}/module6_timeline.png', dpi=100, bbox_inches='tight')
    plt.show()

    # ── Figure 2: the pooled evidence behind findings 2-4 ──
    fig_d, axd = plt.subplots(1, 3, figsize=(19, 5))
    ax = axd[0]
    if pool['ctx']:
        ctx_any, tgt_any = np.concatenate(pool['ctx']), np.concatenate(pool['tgt'])
        clean = ~(ctx_any | tgt_any)
        groups = [('no drop', clean), ('drop in context', ctx_any), ('drop before target', tgt_any)]
        data = [lr_t[g] for _, g in groups if g.sum() > 0]
        labels = [f'{name}\nn={int(g.sum())}' for name, g in groups if g.sum() > 0]
        ax.boxplot(data, showfliers=True)
        ax.set_xticks(range(1, len(data) + 1))
        ax.set_xticklabels(labels)
        ax.axhline(0, color='gray', lw=0.8, ls='--')
        ax.set_ylabel('log(model CD / naive CD), centred per video')
    else:
        ax.text(0.5, 0.5, 'frame timing unavailable', ha='center', va='center',
                transform=ax.transAxes)
    ax.set_title('Does the model do worse where frames were dropped?', fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    ax = axd[1]
    names = [ws.replace('workspace_', '') for ws in timeline]
    shares = [T['excess_share_top'] * 100 for T in timeline.values()]
    losses = [T['loss_frac'] * 100 for T in timeline.values()]
    y = np.arange(len(names))
    ax.barh(y - 0.2, shares, height=0.4, color='crimson', alpha=0.8,
            label=f'excess error in worst {TIMELINE_TOP_FRAC * 100:.0f}% of origins')
    ax.barh(y + 0.2, losses, height=0.4, color='gray', alpha=0.6, label='origins losing to naive')
    ax.axvline(50, color='black', lw=0.8, ls=':')
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel('%')
    ax.set_xlim(0, 100)
    ax.set_title('Bad everywhere, or bad in bursts?', fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.3, axis='x')

    ax = axd[2]
    for (ws, T), d_, l_ in zip(timeline.items(), pool['dist'], pool['lr']):
        ax.scatter(d_, l_, s=9, alpha=0.55, label=ws.replace('workspace_', ''))
    if np.std(dist_all) > 0:
        fit = np.polyfit(dist_all, lr_all, 1)
        xs = np.linspace(dist_all.min(), dist_all.max(), 50)
        ax.plot(xs, np.polyval(fit, xs), color='black', lw=1.5)
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.set_xlabel('frames past the train/held-out split')
    ax.set_ylabel('log(model CD / naive CD), centred per video')
    ax.set_title(f'Does error drift away from training?  rho={rho_all:+.2f}, p={p_all:.3f}',
                 fontweight='bold')
    ax.legend(fontsize=7, ncol=2, markerscale=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig_d.savefig(f'{MODULE6_DIR}/module6_timeline_diagnostics.png', dpi=110, bbox_inches='tight')
    plt.show()
    print(f'\n  Figures          : {MODULE6_DIR}/module6_timeline.png')
    print(f'                     {MODULE6_DIR}/module6_timeline_diagnostics.png')
    print(f'  Frame timing     : {TIMING_DIR}/  (cached; delete to recompute)')


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
        seq = rollout(ctx, ROLLOUT_HORIZON, disp_norm=dn, alpha=ROLL_ALPHA)[0]
        gt = torch.stack([tn[t0 + s - 1] for s in range(1, ROLLOUT_HORIZON + 1)])
        # Per-point uncertainty from an independent hypothesis fan at the same origin.
        hyps = torch.stack([
            rollout(ctx, ROLLOUT_HORIZON, n_samples=1, disp_norm=dn, eta=HYP_ETA,
                    alpha=ROLL_ALPHA,
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
    'target_mode': BUNDLE['target_mode'],
    'calibration': {'enabled': CALIBRATE, 'applied': APPLY_CALIBRATION,
                    'alpha': {str(k): alpha[k] for k in PRED_STEPS},
                    'fit': {str(k): v for k, v in calib_report.items()}},
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
    # Every held-out origin, every horizon: per_origin[workspace][k] = {'t', 'model', 'naive', 'cv'}
    'per_origin': {ws: {str(k): v for k, v in po.items()} for ws, po in direct_by_origin.items()},
    'timeline': {
        'horizon': TIMELINE_HORIZON, 'spike_z': TIMELINE_SPIKE_Z, 'top_frac': TIMELINE_TOP_FRAC,
        'videos': timeline, 'findings': findings,
        'frame_timing': {ws: {k: v for k, v in tim.items() if k != 'raw_index'}
                         for ws, tim in timing_by_ws.items()},
    },
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
# VISUAL INFERENCE — the model creating frames, side by side with the real video
#
# Two views, both on one held-out origin of VIS_VIDEO_ID:
#
#   1. DENOISING. Every DDIM step drawn as motion arrows over the real video frame: the
#      noisy sample x_t the network is handed, and its running estimate of the clean
#      displacement x0. Arrows are coloured by agreement with the TRUE motion (green =
#      same direction, grey = unrelated, red = opposite), so the strip shows noise
#      organising into -- or failing to organise into -- the real deformation.
#
#   2. SIDE BY SIDE. For every rollout step, the actual video frame next to a RENDERED
#      predicted frame, the 4DGS reconstruction of that same frame, a naive "nothing
#      moves" render, the anchor motion, and an error map, with PSNR/SSIM.
#
# How the predicted frame is rendered, and what that means for reading it:
#   The model predicts motion for 512 anchors only. To render, each of the video's full
#   set of Gaussians takes an inverse-distance blend of its VIS_KNN nearest anchors'
#   displacements, added to the scene exactly as reconstructed at the last observed frame
#   (Module 2's camera is fixed at identity, so world space IS camera space and nothing
#   else needs aligning). Colour, scale, rotation and opacity stay frozen at that frame.
#   So this render shows the model's predicted GEOMETRY and nothing more -- it cannot show
#   instruments moving, lighting or specular changes, because nothing in the pipeline
#   predicts them. Tissue-only PSNR uses Module 1's tool masks to take instruments out
#   of the comparison, and the 4DGS reconstruction column is the renderer's own ceiling:
#   a perfect geometric predictor could not beat it.
# ════════════════════════════════════════════════════════════════════
import glob, subprocess, base64
import cv2

try:
    from skimage.metrics import structural_similarity as _sk_ssim
except Exception:
    _sk_ssim = None

visual_summary = None


# ── Module 2 scene model, re-declared so this cell runs in a fresh session ──
# Must stay identical to Module 2 (notebook cell 15): the deformation MLP's state_dict is
# loaded from each video's 4dgs_v4.pth, so any architectural drift fails loudly at load.
N_FRAMES_EMBED = 300


class PosEnc(nn.Module):
    def __init__(self, d_in, n_freqs=8):
        super().__init__()
        self.d_out = d_in * (1 + 2 * n_freqs)
        self.register_buffer('freqs', 2. ** torch.linspace(0, n_freqs - 1, n_freqs))

    def forward(self, x):
        enc = [x]
        for f in self.freqs:
            enc += [torch.sin(f * x), torch.cos(f * x)]
        return torch.cat(enc, -1)


class DeformMLP(nn.Module):
    def __init__(self, hidden=128, n_xyz=8, n_t=6, latent_dim=32, n_frames=300):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc_xyz = PosEnc(3, n_xyz)
        self.enc_t = PosEnc(1, n_t)
        self.frame_embed = nn.Embedding(n_frames, latent_dim)
        d = self.enc_xyz.d_out + self.enc_t.d_out + latent_dim
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.h_xyz = nn.Linear(hidden, 3)
        self.h_scl = nn.Linear(hidden, 3)
        self.h_rot = nn.Linear(hidden, 4)

    def forward(self, xyz, t_norm, frame_idx=None):
        N = xyz.shape[0]
        tv = torch.full((N, 1), float(t_norm), device=xyz.device)
        enc = torch.cat([self.enc_xyz(xyz), self.enc_t(tv)], -1)
        if frame_idx is not None:
            lat = self.frame_embed(torch.tensor([frame_idx], device=xyz.device)).expand(N, -1)
            enc = torch.cat([enc, lat], -1)
        else:
            enc = torch.cat([enc, torch.zeros(N, self.latent_dim, device=xyz.device)], -1)
        feat = self.net(enc)
        return self.h_xyz(feat), self.h_scl(feat), self.h_rot(feat)


def eval_sh_deg3(sh, dirs):
    x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]
    C0 = 0.28209479177387814
    C1 = 0.4886025119029199
    C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
          -1.0925484305920792, 0.5462742152960396]
    C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
          -0.4570457994644658, 1.445305721320277, -0.5900435899266435]
    sh1, sh2, sh3 = sh[:, 1:4], sh[:, 4:9], sh[:, 9:16]
    result = C0 * sh[:, 0]
    result = result - C1 * y * sh1[:, 0] + C1 * z * sh1[:, 1] - C1 * x * sh1[:, 2]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    result = (result
              + C2[0] * xy * sh2[:, 0]
              + C2[1] * yz * sh2[:, 1]
              + C2[2] * (2 * zz - xx - yy) * sh2[:, 2]
              + C2[3] * xz * sh2[:, 3]
              + C2[4] * (xx - yy) * sh2[:, 4])
    result = (result
              + C3[0] * y * (3 * xx - yy) * sh3[:, 0]
              + C3[1] * xy * z * sh3[:, 1]
              + C3[2] * y * (4 * zz - xx - yy) * sh3[:, 2]
              + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh3[:, 3]
              + C3[4] * x * (4 * zz - xx - yy) * sh3[:, 4]
              + C3[5] * z * (xx - yy) * sh3[:, 5]
              + C3[6] * x * (xx - 3 * yy) * sh3[:, 6])
    return (result + 0.5).clamp(0, 1)


def load_scene(ws):
    """The video's reconstructed 4DGS scene, or None if its checkpoint is not on Drive."""
    p = f'{DRIVE_BASE}/{ws}/checkpoints/4dgs_v4.pth'
    if not os.path.exists(p):
        return None
    ck = torch.load(p, map_location=DEVICE, weights_only=False)
    deform = DeformMLP(hidden=128, n_frames=N_FRAMES_EMBED).to(DEVICE)
    deform.load_state_dict(ck['deform'])
    deform.eval()
    return {'means': ck['means'].to(DEVICE), 'log_sc': ck['log_sc'].to(DEVICE),
            'quats': ck['quats'].to(DEVICE), 'opa': ck['opa'].to(DEVICE),
            'sh': ck['sh'].to(DEVICE), 'deform': deform}


@torch.no_grad()
def scene_at(scene, fi, n_frames):
    """Gaussian means / log-scales / quaternions at frame fi, exactly as Module 2 renders
    and Module 3 exports them (same t_norm, same capped frame embedding index)."""
    d_xyz, d_sc, d_rot = scene['deform'](scene['means'], fi / max(n_frames - 1, 1),
                                         min(fi, N_FRAMES_EMBED - 1))
    return scene['means'] + d_xyz, scene['log_sc'] + d_sc, scene['quats'] + d_rot


@torch.no_grad()
def render_rgb(means, log_sc, quats, scene, K, H, W):
    """Module 2's render path (identity camera), RGB only. Returns uint8 (H, W, 3)."""
    from gsplat import rasterization
    sh = scene['sh']
    if sh.dim() == 3 and sh.shape[1] == 16:
        # Camera sits at the origin, so the view direction is simply the Gaussian position.
        colors = eval_sh_deg3(sh, F.normalize(means, dim=-1))
    else:
        colors = torch.sigmoid(sh if sh.dim() == 2 else sh[:, 0, :])
    out, _, _ = rasterization(
        means=means, quats=F.normalize(quats, dim=-1),
        scales=torch.exp(log_sc).clamp(1e-6, 3.0),
        opacities=torch.sigmoid(scene['opa']).squeeze(-1), colors=colors,
        viewmats=torch.eye(4, device=DEVICE)[None], Ks=K[None],
        width=W, height=H, render_mode='RGB')
    return (out[0, :, :, :3].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)


@torch.no_grad()
def propagate_motion(anchor_pos, anchor_delta, all_pos, k=VIS_KNN, chunk=16384):
    """Spread anchor displacements to every Gaussian by inverse-squared-distance weighting
    over the k nearest anchors. An anchor's own Gaussian gets its displacement exactly
    (zero distance dominates the weights). Chunked so 60k x 512 distances never
    materialise at once."""
    out = torch.empty_like(all_pos)
    kk = min(k, anchor_pos.shape[0])
    for i in range(0, all_pos.shape[0], chunk):
        dist, idx = torch.cdist(all_pos[i:i + chunk], anchor_pos).topk(kk, dim=1, largest=False)
        w = 1.0 / (dist ** 2 + 1e-10)
        w = w / w.sum(dim=1, keepdim=True)
        out[i:i + chunk] = (w[..., None] * anchor_delta[idx]).sum(dim=1)
    return out


# ── Image-space helpers ──────────────────────────────────────────────
def project(pts, cam):
    """World (= camera) points -> pixel coordinates, pinhole with Module 1's intrinsics."""
    z = np.clip(pts[..., 2], 1e-6, None)
    return np.stack([pts[..., 0] / z * cam['fx'] + cam['cx'],
                     pts[..., 1] / z * cam['fy'] + cam['cy']], -1)


def agreement_color(c):
    """Cosine with the true motion -> RGB. -1 red, 0 grey, +1 green."""
    c = float(np.clip(c, -1.0, 1.0))
    mid = np.array([165, 165, 165]); pos = np.array([70, 190, 105]); neg = np.array([220, 70, 60])
    col = mid + (pos - mid) * c if c >= 0 else mid + (neg - mid) * (-c)
    return tuple(int(v) for v in col)


def draw_motion(base_rgb, start_w, delta_w, gain, cam, cos=None, color=(235, 185, 40),
                gt_delta_w=None, dim=0.55):
    """Arrows from each anchor's current on-screen position along its displacement, with
    the on-screen vector multiplied by `gain`. `gt_delta_w` adds a small gold dot where the
    TRUE motion would put the arrow tip, so misses are visible as tip-to-dot gaps."""
    img = (base_rgb.astype(np.float32) * dim).astype(np.uint8)
    H_, W_ = img.shape[:2]
    p0 = project(start_w, cam)
    tip = p0 + (project(start_w + delta_w, cam) - p0) * gain
    ok = ((start_w[:, 2] > 0) & (p0[:, 0] >= 0) & (p0[:, 0] < W_)
          & (p0[:, 1] >= 0) & (p0[:, 1] < H_))
    if gt_delta_w is not None:
        dots = p0 + (project(start_w + gt_delta_w, cam) - p0) * gain
        for i in np.where(ok)[0]:
            cv2.circle(img, (int(round(dots[i, 0])), int(round(dots[i, 1]))), 2,
                       (235, 185, 40), -1, cv2.LINE_AA)
    for i in np.where(ok)[0]:
        col = color if cos is None else agreement_color(cos[i])
        a = (int(round(p0[i, 0])), int(round(p0[i, 1])))
        b = (int(round(tip[i, 0])), int(round(tip[i, 1])))
        if a == b:
            cv2.circle(img, a, 1, col, -1)
        else:
            cv2.arrowedLine(img, a, b, col, 1, cv2.LINE_AA, tipLength=0.35)
    return img


def auto_gain(start_w, gt_delta_w, cam):
    """One arrow magnification for a whole figure, chosen so the MEDIAN true motion is
    ~VIS_ARROW_PX long, snapped to 1/2/5 x 10^n so the printed gain reads cleanly."""
    p0 = project(start_w, cam)
    L = float(np.median(np.linalg.norm(project(start_w + gt_delta_w, cam) - p0, axis=1)))
    if not np.isfinite(L) or L < 1e-6:
        return 1.0
    g = VIS_ARROW_PX / L
    mag = 10.0 ** np.floor(np.log10(g))
    g = min((m * mag for m in (1, 2, 5, 10)), key=lambda v: abs(v - g))
    return float(np.clip(g, 1.0, 1000.0))


def put_label(img, text, org=(8, 22)):
    """White text on a black plate. OpenCV fonts are ASCII-only, so video labels avoid
    unicode (the PNG figures use matplotlib titles instead)."""
    out = img.copy()
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(out, (org[0] - 4, org[1] - th - 6), (org[0] + tw + 4, org[1] + 6), (0, 0, 0), -1)
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def error_panel(pred, actual):
    e = np.abs(pred.astype(np.float32) - actual.astype(np.float32)).mean(-1) * VIS_ERR_GAIN
    e = np.clip(e, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(e, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)


def psnr_u8(a, b, mask=None):
    se = ((a.astype(np.float32) - b.astype(np.float32)) / 255.0) ** 2
    if mask is not None:
        if not mask.any():
            return float('nan')
        se = se[mask]
    return float(-10.0 * np.log10(se.mean() + 1e-10))


def ssim_u8(a, b):
    return float('nan') if _sk_ssim is None else float(_sk_ssim(a, b, channel_axis=2, data_range=255))


def write_mp4(path, frames_rgb, fps):
    """mp4v via OpenCV, then re-encoded to H.264 so browsers and Colab can play it -- the
    same two-step Module 1's reconstruction video uses. Keeps the mp4v file if ffmpeg
    is unavailable."""
    h, w = frames_rgb[0].shape[:2]
    h, w = h - h % 2, w - w % 2                      # H.264 / yuv420p need even dimensions
    raw = path[:-4] + '_mp4v.mp4'
    wr = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frames_rgb:
        wr.write(cv2.cvtColor(np.ascontiguousarray(f[:h, :w]), cv2.COLOR_RGB2BGR))
    wr.release()
    try:
        r = subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', raw, '-vcodec', 'libx264',
                            '-pix_fmt', 'yuv420p', '-crf', '20', path], capture_output=True)
        encoded = r.returncode == 0 and os.path.exists(path)
    except FileNotFoundError:
        encoded = False
    if encoded:
        os.remove(raw)
    else:
        os.replace(raw, path)
    return path


def show_video(path, title, width=960):
    if not VIS_INLINE_VIDEO:
        return
    try:
        from IPython.display import HTML, display
        b64 = base64.b64encode(open(path, 'rb').read()).decode()
        display(HTML(f'<b>{title}</b><br><video width="{width}" controls loop>'
                     f'<source src="data:video/mp4;base64,{b64}" type="video/mp4"></video>'))
    except Exception:
        pass                                          # not in a notebook -- files are on disk


def save_grid(rows, path, title):
    if not rows:
        print(f'  ⚠️  nothing to draw for {os.path.basename(path)} (check VIS_GRID_STEPS)')
        return
    n_c = max(len(r) for r in rows)
    fig_, axs = plt.subplots(len(rows), n_c, figsize=(3.3 * n_c, 2.05 * len(rows) + 0.7),
                             squeeze=False)
    for r, row in enumerate(rows):
        for c in range(n_c):
            ax = axs[r][c]
            ax.axis('off')
            if c < len(row):
                ax.imshow(row[c][1])
                ax.set_title(row[c][0], fontsize=8)
    fig_.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig_.savefig(path, dpi=110, bbox_inches='tight')
    plt.show()


print('\n' + '=' * 64)
print('  VISUAL INFERENCE — model creating frames vs the real video')
print('=' * 64)

_vis = [v for v in videos if v['video_id'] == VIS_VIDEO_ID]
if not VIS_ENABLED:
    print('  (disabled — VIS_ENABLED = False)')
elif not _vis:
    print(f'  ⚠️  VIS_VIDEO_ID={VIS_VIDEO_ID} was not among the rebuilt videos — skipped.')
else:
    vid = _vis[0]
    ws, tn, dn = vid['workspace'], vid['traj_n'], vid['disp_norm']
    n_frames = vid['n_frames']
    t0 = max(WINDOW_SIZE, vid['split_frame'])          # same held-out origin as the exports
    VIS_DIR = f'{MODULE6_DIR}/visuals'
    os.makedirs(VIS_DIR, exist_ok=True)

    def denorm_np(x):
        return x.detach().cpu().numpy() * vid['scl'] + vid['ctr']

    # ── Real video frames, aligned 1:1 with trajectory indices ──
    # Mirrors Module 2's SurgicalDataset: sorted frames/*.png, index fi == trajectory frame fi.
    frame_paths = sorted(glob.glob(f'{DRIVE_BASE}/{ws}/frames/*.png'))
    frames_ok = len(frame_paths) == n_frames
    if not frames_ok:
        print(f'  ⚠️  {ws}: {len(frame_paths)} frame PNGs but the trajectory has {n_frames} '
              f'frames. Refusing to pair them — a misaligned side-by-side would compare the '
              f'prediction against the wrong moment. Overlays use a blank canvas instead.')

    def read_frame(fi):
        if not frames_ok:
            return None
        img = cv2.imread(frame_paths[fi])
        return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def tissue_mask(fi):
        """True on tissue. Module 1 masks paint instruments white; Module 2 trains on m < 127."""
        if not frames_ok:
            return None
        stem = os.path.splitext(os.path.basename(frame_paths[fi]))[0]
        m = cv2.imread(f'{DRIVE_BASE}/{ws}/masks/{stem}.png', cv2.IMREAD_GRAYSCALE)
        if m is None:
            return None
        if m.shape != (IMG_H, IMG_W):                   # frames are resized by fit(); match them
            m = cv2.resize(m, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
        return m < 127

    _sample = read_frame(0)
    cam_path = f'{DRIVE_BASE}/{ws}/camera_meta.json'
    if os.path.exists(cam_path):
        cam = json.load(open(cam_path))
    else:
        # Module 1's own fallback: 75-degree horizontal FOV laparoscope, principal point centred.
        H_, W_ = _sample.shape[:2] if _sample is not None else (360, 640)
        _fx = (W_ / 2) / math.tan(math.radians(75.0) / 2)
        cam = {'H': H_, 'W': W_, 'fx': _fx, 'fy': _fx, 'cx': W_ / 2.0, 'cy': H_ / 2.0}
        print(f'  ⚠️  camera_meta.json missing — using Module 1 default intrinsics (75° HFOV).')
    IMG_H, IMG_W = int(cam['H']), int(cam['W'])
    blank = np.full((IMG_H, IMG_W, 3), 40, np.uint8)

    def fit(img):
        if img is None:
            return blank.copy()
        return img if img.shape[:2] == (IMG_H, IMG_W) else cv2.resize(img, (IMG_W, IMG_H))

    if t0 + ROLLOUT_HORIZON > n_frames:
        print(f'  ⚠️  {ws}: held-out tail too short for a {ROLLOUT_HORIZON}-step visual rollout.')
    else:
        last_n = tn[t0 - 1]                                            # last observed (normalised)
        last_w = denorm_np(last_n)                                     # (N_A, 3) world
        ctx_v = tn[t0 - WINDOW_SIZE:t0][None]

        # ════════ 1. DENOISING — the model creating the next frame ════════
        trace = []
        x0_final = ddim_sample(BUNDLE, ctx_v, torch.full((1,), 1, device=DEVICE, dtype=torch.long),
                               steps=DDIM_STEPS, eta=0.0, n_samples=DDIM_SAMPLES, trace=trace,
                               generator=torch.Generator(device=DEVICE).manual_seed(SEED + 11))
        gt_delta_n = tn[t0] - last_n                                   # true next-frame motion
        gt_delta_w = gt_delta_n.cpu().numpy() * vid['scl']
        gain_d = auto_gain(last_w, gt_delta_w, cam)
        observed = fit(read_frame(t0 - 1))
        actual_next = fit(read_frame(t0))

        den_steps = []
        for i, (t_step, x_t, x0_est) in enumerate(trace):
            d_xt = to_physical(x_t[0], dn)
            d_x0 = to_physical(x0_est[0], dn)
            cos_xt = F.cosine_similarity(d_xt, gt_delta_n, dim=-1).cpu().numpy()
            cos_x0 = F.cosine_similarity(d_x0, gt_delta_n, dim=-1).cpu().numpy()
            den_steps.append({
                'i': i, 't': t_step,
                'xt_img': draw_motion(observed, last_w, d_xt.cpu().numpy() * vid['scl'],
                                      gain_d, cam, cos=cos_xt),
                'x0_img': draw_motion(observed, last_w, d_x0.cpu().numpy() * vid['scl'],
                                      gain_d, cam, cos=cos_x0, gt_delta_w=gt_delta_w),
                'cos_xt': float(cos_xt.mean()), 'cos_x0': float(cos_x0.mean()),
            })
        gt_img = draw_motion(observed, last_w, gt_delta_w, gain_d, cam, color=(235, 185, 40))

        pick = np.unique(np.linspace(0, len(den_steps) - 1, VIS_DENOISE_PANELS).round().astype(int))
        rows = [
            [(f'step {den_steps[j]["i"] + 1}/{len(den_steps)} · t={den_steps[j]["t"]}\n'
              f'noisy sample x_t · cos {den_steps[j]["cos_xt"]:+.2f}', den_steps[j]['xt_img'])
             for j in pick] + [('true motion (gold)', gt_img)],
            [(f'model estimate of x0 · cos {den_steps[j]["cos_x0"]:+.2f}', den_steps[j]['x0_img'])
             for j in pick] + [(f'actual next frame  ({t0})', actual_next)],
        ]
        den_png = f'{VIS_DIR}/denoising_{ws}.png'
        save_grid(rows, den_png,
                  f'Model creating frame {t0} from frames {t0 - WINDOW_SIZE}–{t0 - 1}  ·  '
                  f'arrows ×{gain_d:g}  ·  green = moves the true way, red = opposite, '
                  f'gold dot = true tip')

        den_frames = []
        for s_ in den_steps:
            den_frames.append(np.hstack([
                put_label(s_['xt_img'], f'noisy sample x_t   t={s_["t"]}'),
                put_label(s_['x0_img'], f'model estimate x0   cos {s_["cos_x0"]:+.2f}'),
                put_label(gt_img, f'true motion   arrows x{gain_d:g}'),
            ]))
        den_frames += [den_frames[-1]] * VIS_DENOISE_FPS * 2           # hold the result 2 s
        den_mp4 = write_mp4(f'{VIS_DIR}/denoising_{ws}.mp4', den_frames, VIS_DENOISE_FPS)
        show_video(den_mp4, f'Denoising — the model creating frame {t0} ({ws})', width=1200)

        # The trace's last estimate must be exactly what the sampler returned; if not, the
        # figure would be showing a different prediction from the one being scored.
        assert torch.allclose(trace[-1][2], x0_final, atol=1e-5), 'denoising trace diverged'
        print(f'  Denoising        : direction agreement with true motion '
              f'{den_steps[0]["cos_x0"]:+.2f} (step 1) → {den_steps[-1]["cos_x0"]:+.2f} '
              f'(step {len(den_steps)})')
        print(f'                     {den_png}')
        print(f'                     {den_mp4}')

        # ════════ 2. SIDE BY SIDE — rendered prediction vs the real video ════════
        seq_v = rollout(ctx_v, ROLLOUT_HORIZON, disp_norm=dn, alpha=ROLL_ALPHA,
                        generator=torch.Generator(device=DEVICE).manual_seed(SEED + 13))[0]

        scene, render_note = None, None
        if VIS_RENDER:
            try:
                import gsplat  # noqa: F401
                scene = load_scene(ws)
                if scene is None:
                    render_note = f'{ws}/checkpoints/4dgs_v4.pth not found'
            except ImportError:
                render_note = ('gsplat is not installed — run the notebook dependency cell, or: pip install gsplat')
        if scene is not None:
            K_t = torch.tensor([[cam['fx'], 0, cam['cx']], [0, cam['fy'], cam['cy']], [0, 0, 1]],
                               dtype=torch.float32, device=DEVICE)
            m_b, s_b, q_b = scene_at(scene, t0 - 1, n_frames)
            # Same check as QT0, one level down: the scene must reproduce the exported
            # trajectory, or the render is not of the geometry the model was trained on.
            _row = np.load(f'{DRIVE_BASE}/{ws}/exports/gaussian_trajectory.npy', mmap_mode='r')[t0 - 1]
            if _row.shape[0] != m_b.shape[0]:
                render_note = (f'4dgs_v4.pth has {m_b.shape[0]:,} Gaussians but the exported '
                               f'trajectory has {_row.shape[0]:,} — the scene was retrained after '
                               f'Module 3; re-run Module 3')
                scene = None
            else:
                _dev = float(np.abs(m_b.cpu().numpy() - _row).max())
                if _dev > 1e-3:
                    print(f'  ⚠️  scene vs exported trajectory max deviation {_dev:.2e} — renders '
                          f'may not match the geometry Module 5 trained on.')
        if scene is None and VIS_RENDER:
            print(f'  ⚠️  Rendering skipped: {render_note}. Showing motion overlays on the '
                  f'real frames only.')

        anc_idx = torch.as_tensor(vid['anc'], device=DEVICE)
        naive_r = render_rgb(m_b, s_b, q_b, scene, K_t, IMG_H, IMG_W) if scene is not None else None
        gain_s = auto_gain(last_w, (tn[t0 - 1 + ROLLOUT_HORIZON] - last_n).cpu().numpy()
                           * vid['scl'], cam)

        per_step, sbs_frames, grid_rows = [], [], []
        # Frame 0 of the video: the last thing the model actually saw.
        if scene is not None:
            sbs_frames.append(np.vstack([
                np.hstack([put_label(observed, f'ACTUAL frame {t0 - 1} (last observed)'),
                           put_label(naive_r, 'RENDER of last observed frame')]),
                np.hstack([put_label(observed, 'model sees frames up to here'),
                           put_label(np.zeros_like(observed), 'error')]),
            ]))
        for s in range(1, ROLLOUT_HORIZON + 1):
            fi = t0 - 1 + s
            actual = fit(read_frame(fi))
            mask = tissue_mask(fi)
            pred_delta_n = seq_v[s - 1] - last_n
            true_delta_n = tn[fi] - last_n
            cos_s = F.cosine_similarity(pred_delta_n, true_delta_n, dim=-1).cpu().numpy()
            motion = draw_motion(actual, last_w, pred_delta_n.cpu().numpy() * vid['scl'], gain_s,
                                 cam, cos=cos_s, gt_delta_w=true_delta_n.cpu().numpy() * vid['scl'])
            rec = {'step': s, 'frame': fi, 'cos': float(cos_s.mean())}

            if scene is not None:
                delta_w = (pred_delta_n * vid['scl']).to(m_b.dtype)
                pred_means = m_b + propagate_motion(m_b[anc_idx], delta_w, m_b)
                pred_r = render_rgb(pred_means, s_b, q_b, scene, K_t, IMG_H, IMG_W)
                m_o, s_o, q_o = scene_at(scene, fi, n_frames)
                oracle_r = render_rgb(m_o, s_o, q_o, scene, K_t, IMG_H, IMG_W)
                err = error_panel(pred_r, actual)
                if frames_ok:
                    rec.update({
                        'psnr_pred': psnr_u8(pred_r, actual), 'psnr_naive': psnr_u8(naive_r, actual),
                        'psnr_recon': psnr_u8(oracle_r, actual),
                        'tissue_psnr_pred': psnr_u8(pred_r, actual, mask),
                        'tissue_psnr_naive': psnr_u8(naive_r, actual, mask),
                        'tissue_psnr_recon': psnr_u8(oracle_r, actual, mask),
                        'ssim_pred': ssim_u8(pred_r, actual), 'ssim_naive': ssim_u8(naive_r, actual),
                        'ssim_recon': ssim_u8(oracle_r, actual),
                    })
                sbs_frames.append(np.vstack([
                    np.hstack([put_label(actual, f'ACTUAL video frame {fi}'),
                               put_label(pred_r, f'PREDICTED frame  (t+{s})')]),
                    np.hstack([put_label(motion, f'predicted motion  arrows x{gain_s:g}'),
                               put_label(err, f'|predicted - actual|  x{VIS_ERR_GAIN:g}')]),
                ]))
                if s in VIS_GRID_STEPS:
                    ps = rec.get('psnr_pred', float('nan'))
                    grid_rows.append([
                        (f'actual video · frame {fi}', actual),
                        (f'4DGS reconstruction · {rec.get("psnr_recon", float("nan")):.2f} dB', oracle_r),
                        (f'predicted t+{s} · {ps:.2f} dB', pred_r),
                        (f'naive (no motion) · {rec.get("psnr_naive", float("nan")):.2f} dB', naive_r),
                        (f'predicted motion ×{gain_s:g} · cos {rec["cos"]:+.2f}', motion),
                        (f'|predicted − actual| ×{VIS_ERR_GAIN:g}', err),
                    ])
            else:
                sbs_frames.append(np.hstack([
                    put_label(actual, f'ACTUAL video frame {fi}'),
                    put_label(motion, f'predicted motion t+{s}  arrows x{gain_s:g}'),
                ]))
                if s in VIS_GRID_STEPS:
                    grid_rows.append([(f'actual video · frame {fi}', actual),
                                      (f'predicted motion t+{s} ×{gain_s:g} · cos {rec["cos"]:+.2f}',
                                       motion)])
            per_step.append(rec)

        sbs_png = f'{VIS_DIR}/side_by_side_{ws}.png'
        save_grid(grid_rows, sbs_png,
                  f'Predicted vs actual — {ws}, rolled out from frame {t0 - 1} '
                  f'(held-out; arrows green = true direction, gold dot = true tip)')
        sbs_frames += [sbs_frames[-1]] * VIS_FPS * 2                   # hold the last step 2 s
        sbs_mp4 = write_mp4(f'{VIS_DIR}/side_by_side_{ws}.mp4', sbs_frames, VIS_FPS)
        show_video(sbs_mp4, f'Actual video vs predicted frames — {ws}')

        print(f'  Side by side     : {sbs_png}')
        print(f'                     {sbs_mp4}')

        # ════════ 3. FORECAST — frames the camera never recorded ════════
        # Everything above predicts a frame that already exists, which is what makes it
        # scoreable. This instead rolls the model off the END of the footage: the context is
        # the last four real frames, and every frame it produces is a moment with no
        # recording to compare against. That is the actual deliverable of a predictive
        # system, and it is the only view here where the predicted stream is LONGER than the
        # video it came from.
        #
        # Nothing here can be scored. Two things therefore carry the honesty: the ACTUAL
        # panel stops dead at the last real frame and says so, and the spread of five
        # independent futures is drawn alongside, so a prediction the model is unsure about
        # looks unsure.
        forecast_summary = None
        if FORECAST_FRAMES > 0 and n_frames >= WINDOW_SIZE + 1:
            f_last = n_frames - 1                       # last frame that actually exists
            f_ctx = tn[n_frames - WINDOW_SIZE:n_frames][None]
            f_seq = rollout(f_ctx, FORECAST_FRAMES, disp_norm=dn, alpha=ROLL_ALPHA,
                            generator=torch.Generator(device=DEVICE).manual_seed(SEED + 21))[0]
            f_hyp = torch.stack([
                rollout(f_ctx, FORECAST_FRAMES, n_samples=1, disp_norm=dn, eta=HYP_ETA,
                        alpha=ROLL_ALPHA,
                        generator=torch.Generator(device=DEVICE).manual_seed(SEED + 300 + h))[0]
                for h in range(N_HYPOTHESES)], dim=0)
            f_unc = f_hyp.std(dim=0).norm(dim=-1)        # (F, N_A) how much the futures differ
            f_last_n = tn[f_last]
            f_last_w = denorm_np(f_last_n)
            f_last_img = fit(read_frame(f_last))
            gain_f = auto_gain(f_last_w, (f_seq[-1] - f_last_n).cpu().numpy() * vid['scl'], cam)
            unc_max = float(f_unc.max()) if float(f_unc.max()) > 0 else 1.0

            if scene is not None:
                m_f, s_f, q_f = scene_at(scene, f_last, n_frames)

            def render_forecast(step, exaggerate=1.0):
                """Predicted frame f_last+step. exaggerate>1 scales the predicted motion so a
                1-2 px deformation is actually visible; it changes the picture, never a number."""
                delta_w = ((f_seq[step - 1] - f_last_n) * vid['scl'] * exaggerate).to(m_f.dtype)
                means = m_f + propagate_motion(m_f[anc_idx], delta_w, m_f)
                return render_rgb(means, s_f, q_f, scene, K_t, IMG_H, IMG_W)

            def uncertainty_panel(step):
                """Where the five futures disagree, drawn on the last real frame."""
                img = (f_last_img.astype(np.float32) * 0.45).astype(np.uint8)
                pts = project(denorm_np(f_seq[step - 1]), cam)
                u = f_unc[step - 1].cpu().numpy() / unc_max
                for i in range(len(pts)):
                    x, y = pts[i]
                    if 0 <= x < IMG_W and 0 <= y < IMG_H:
                        c = agreement_color(1.0 - 2.0 * float(np.clip(u[i], 0, 1)))
                        cv2.circle(img, (int(round(x)), int(round(y))),
                                   1 + int(round(3 * float(np.clip(u[i], 0, 1)))), c, -1, cv2.LINE_AA)
                return img

            # The "video has ended" card: the last real frame, faded, so the contrast with a
            # still-moving prediction beside it is unmistakable.
            end_card = (f_last_img.astype(np.float32) * 0.28).astype(np.uint8)
            end_card = put_label(end_card, f'VIDEO ENDS HERE - last real frame {f_last}')
            end_card = put_label(end_card, 'no footage exists beyond this point', (8, 48))

            f_frames, pred_only, actual_only = [], [], []
            lead = list(range(max(0, f_last - FORECAST_LEADIN + 1), f_last + 1))
            for fi in lead:
                actual = fit(read_frame(fi))
                right = (render_rgb(*scene_at(scene, fi, n_frames), scene, K_t, IMG_H, IMG_W)
                         if scene is not None else actual)
                actual_only.append(put_label(actual, f'ACTUAL VIDEO  frame {fi}'))
                pred_only.append(put_label(right, f'OBSERVED  frame {fi}'))
                f_frames.append(np.vstack([
                    np.hstack([put_label(actual, f'ACTUAL VIDEO  frame {fi}'),
                               put_label(right, 'RECONSTRUCTION  (model is watching)')]),
                    np.hstack([put_label((actual.astype(np.float32) * 0.45).astype(np.uint8),
                                         'observing - no prediction yet'),
                               put_label(np.zeros_like(actual), 'uncertainty: n/a')]),
                ]))

            grid_f = []
            for s in range(1, FORECAST_FRAMES + 1):
                fu = f_last + s
                if scene is not None:
                    pred_img = render_forecast(s)
                    exag_img = render_forecast(s, FORECAST_EXAGGERATE)
                else:
                    pred_img = draw_motion(f_last_img, f_last_w,
                                           (f_seq[s - 1] - f_last_n).cpu().numpy() * vid['scl'],
                                           gain_f, cam, color=(210, 70, 60))
                    exag_img = pred_img
                unc_img = uncertainty_panel(s)
                lbl = f'PREDICTED  frame {fu}  (+{s} beyond the video)'
                pred_only.append(put_label(pred_img, lbl))
                f_frames.append(np.vstack([
                    np.hstack([end_card, put_label(pred_img, lbl)]),
                    np.hstack([put_label(exag_img, f'same prediction, motion x{FORECAST_EXAGGERATE:g} '
                                                   f'(so it is visible)'),
                               put_label(unc_img, 'disagreement between 5 possible futures')]),
                ]))
                if s in FORECAST_GRID_STEPS:
                    grid_f.append([(f'last real frame {f_last}', f_last_img),
                                   (f'predicted +{s}  (frame {fu})', pred_img),
                                   (f'motion ×{FORECAST_EXAGGERATE:g} · exaggerated to be visible', exag_img),
                                   (f'spread of 5 futures · mean {float(f_unc[s - 1].mean()):.5f}', unc_img)])

            fc_png = f'{VIS_DIR}/forecast_{ws}.png'
            save_grid(grid_f, fc_png,
                      f'Forecast past the end of {ws} — frames {f_last + 1}–{f_last + FORECAST_FRAMES} '
                      f'were never recorded, so none of these can be scored')
            f_frames += [f_frames[-1]] * VIS_FPS * 2
            fc_mp4 = write_mp4(f'{VIS_DIR}/forecast_{ws}.mp4', f_frames, VIS_FPS)
            # Two more, deliberately of different lengths: the real clip, and the predicted
            # stream that carries on past where it stops.
            act_mp4 = write_mp4(f'{VIS_DIR}/forecast_actual_{ws}.mp4',
                                actual_only[:len(lead)], VIS_FPS)
            pre_mp4 = write_mp4(f'{VIS_DIR}/forecast_predicted_{ws}.mp4', pred_only, VIS_FPS)
            show_video(fc_mp4, f'Forecast beyond the end of the video — {ws}')

            forecast_summary = {
                'last_real_frame': int(f_last), 'frames_predicted': FORECAST_FRAMES,
                'lead_in': len(lead), 'rendered': scene is not None,
                'exaggeration': FORECAST_EXAGGERATE,
                'mean_uncertainty_by_step': [float(f_unc[i].mean()) for i in range(FORECAST_FRAMES)],
                'png': fc_png, 'mp4': fc_mp4,
                'actual_mp4': act_mp4, 'predicted_mp4': pre_mp4,
                'actual_frames': len(lead), 'predicted_frames': len(pred_only),
            }
            print(f'\n  Forecast         : {FORECAST_FRAMES} frames past the end of the video '
                  f'(last real frame {f_last})')
            print(f'    actual clip    : {len(lead)} frames   {act_mp4}')
            print(f'    predicted clip : {len(pred_only)} frames   {pre_mp4}   '
                  f'← {FORECAST_FRAMES} frames longer')
            print(f'    side by side   : {fc_mp4}')
            print(f'                     {fc_png}')
            _u0, _u1 = float(f_unc[0].mean()), float(f_unc[-1].mean())
            print(f'    uncertainty    : {_u0:.5f} at +1 → {_u1:.5f} at +{FORECAST_FRAMES} '
                  f'({_u1 / max(_u0, 1e-12):.1f}x wider)')
            print('    These frames have no ground truth and are NOT scored anywhere — they '
                  'show what the model does, not how right it is.')

        # ── Image-space table ──
        if scene is not None and frames_ok:
            print(f'\n  Image-space comparison against the actual video '
                  f'(PSNR dB; tissue = instruments masked out):')
            print(f'  {"step":>4} {"frame":>6} {"pred":>7} {"naive":>7} {"recon":>7}   '
                  f'{"tissue pred":>11} {"naive":>7} {"recon":>7}   {"SSIM pred":>9} {"naive":>7}')
            for r in per_step:
                print(f'  {r["step"]:>4} {r["frame"]:>6} {r["psnr_pred"]:>7.2f} {r["psnr_naive"]:>7.2f} '
                      f'{r["psnr_recon"]:>7.2f}   {r["tissue_psnr_pred"]:>11.2f} '
                      f'{r["tissue_psnr_naive"]:>7.2f} {r["tissue_psnr_recon"]:>7.2f}   '
                      f'{r["ssim_pred"]:>9.4f} {r["ssim_naive"]:>7.4f}')
            _tp = np.nanmean([r['tissue_psnr_pred'] for r in per_step])
            _tn = np.nanmean([r['tissue_psnr_naive'] for r in per_step])
            _tr = np.nanmean([r['tissue_psnr_recon'] for r in per_step])
            # How much image-space headroom real motion even creates over "nothing moves".
            # If a perfect geometric predictor (the reconstruction) barely beats naive, the
            # motion is sub-pixel and PSNR cannot rank predictors -- the Chamfer tables above
            # are then the decisive evidence, and saying so beats over-reading a 0.1 dB gap.
            headroom = _tr - _tn
            print(f'\n  Mean tissue PSNR : predicted {_tp:.2f} dB · naive {_tn:.2f} dB · '
                  f'reconstruction {_tr:.2f} dB')
            if headroom < 0.5:
                print(f'  → Motion over this rollout is near sub-pixel: even the reconstruction '
                      f'beats "nothing moves" by only {headroom:+.2f} dB. Image-space scores '
                      f'cannot separate the predictors here; read QT1/QT2 (Chamfer) as the '
                      f'decisive result and use these frames qualitatively.')
            elif _tp > _tn:
                print(f'  → The predicted frames recover {(_tp - _tn) / headroom * 100:.0f}% of '
                      f'the {headroom:.2f} dB that true motion is worth over naive.')
            else:
                print(f'  → The predicted frames are {_tn - _tp:.2f} dB WORSE than naive: the '
                      f'predicted motion moves tissue away from where it actually goes, '
                      f'consistent with QT1.')

        visual_summary = {
            'workspace': ws, 'origin_frame': int(t0),
            'denoising': {'png': den_png, 'mp4': den_mp4, 'arrow_gain': gain_d,
                          'cos_by_step': [s_['cos_x0'] for s_ in den_steps]},
            'side_by_side': {'png': sbs_png, 'mp4': sbs_mp4, 'arrow_gain': gain_s,
                             'rendered': scene is not None, 'render_note': render_note,
                             'frames_aligned': frames_ok, 'steps': per_step},
            'forecast': forecast_summary,
        }
        metrics['visual'] = visual_summary
        json.dump(metrics, open(f'{MODULE6_DIR}/module6_metrics.json', 'w'), indent=2)


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
print(f'  Target mode      : {BUNDLE["target_mode"]}'
      + (f'   ·  alpha {", ".join(f"k{k}={alpha[k]:.2f}" for k in PS_OK)}' if CALIBRATE else ''))
print(f'  Direct CD (k={PS_OK[0]})  : {mean_of(direct[PS_OK[0]], "model"):.6f} raw · '
      f'{mean_of(direct[PS_OK[0]], "model_cal"):.6f} calibrated · '
      f'naive {mean_of(direct[PS_OK[0]], "naive"):.6f}')
print(f'  Rollout growth b : {_growth:.2f}')
print(f'  Exports          : {MODULE6_DIR}/exports/')
print(f'  Metrics          : {MODULE6_DIR}/module6_metrics.json')
print(f'  Figure           : {MODULE6_DIR}/module6_qa.png')
if timeline:
    print(f'  Timeline         : {MODULE6_DIR}/module6_timeline.png  '
          f'(+ _diagnostics.png)')
if visual_summary is not None:
    _fc = visual_summary.get('forecast')
    if _fc:
        print(f'  Forecast         : {_fc["predicted_mp4"]}  '
              f'({_fc["predicted_frames"]} frames vs {_fc["actual_frames"]} real)')
    print(f'  Denoising        : {visual_summary["denoising"]["mp4"]}')
    print(f'  Side by side     : {visual_summary["side_by_side"]["mp4"]}'
          + ('' if visual_summary['side_by_side']['rendered'] else '   (overlays only — not rendered)'))
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
