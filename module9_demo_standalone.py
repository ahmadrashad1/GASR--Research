# ════════════════════════════════════════════════════════════════════
# MODULE 9 — DEMO EXPORT, STANDALONE  (needs nothing but Drive)
#
# GENERATED FILE -- do not edit. Produced by tools/make_demo_cell.py, which lifts the
# model, the sampler, the trajectory reconstruction and the export itself verbatim out
# of module6_prediction.py and module9_demo_export.py. Edit those and re-run the script.
#
# WHY THIS EXISTS
# The normal Module 9 cell reuses Module 6's objects from memory, which is the right
# default: one prediction path, the one every metric was measured with. But memory is
# lost on a runtime disconnect, and re-running Module 6 costs roughly an hour of
# held-out evaluation, rollouts, a hypothesis study and two rendered MP4s -- none of
# which the demo export reads.
#
# This cell rebuilds only what the export needs, from Drive:
#   diffusion_ckpt_best.pth   -> the weights, and the arch/config that built them
#   module4_metadata.json     -> per-video smooth_window, disp_norm, split_frame
#   gaussian_trajectory.npy   -> the raw per-frame Gaussian positions
#   4dgs_v4.pth               -> opacity, for the anchor gate
#
# 'Rebuilt' is the same trajectory, not a similar one: anchor selection is seeded with
# default_rng(M4_SEED + video_id) and the smoothing window is READ from Module 4's
# metadata rather than re-selected. Module 6's QT0 is what verifies that claim against
# Module 4's own saved windows; it is not repeated here, so if the replay has never been
# verified in this project, run Module 6 at least once before trusting this cell.
#
# WHAT IS LOST BY NOT RUNNING MODULE 6
#   - frame timing. Module 6 recovers which raw video frame each trajectory frame came
#     from, well past the point this cell stops. Absent it the payload carries
#     timing_ok: false and the viewer states the horizon in frames only -- which is
#     exact, and the behaviour the design asks for anyway.
#   - nothing else. The magnitude calibration IS fitted below, on training origins, the
#     same scalar Module 6 reports its headline numbers with.
#
# It does not train and it writes no checkpoint: there is no torch.save and no os.rename
# anywhere in this file.
# ════════════════════════════════════════════════════════════════════
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, os, math, json, time
from scipy.signal import savgol_filter

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# DRIVE_BASE and the Drive mount normally come from the setup cell at the top of the
# notebook. Recovered here so this cell runs standalone in a fresh runtime.
if 'DRIVE_BASE' not in globals():
    DRIVE_BASE = '/content/drive/MyDrive/EndoGaussian'
    if not os.path.isdir(DRIVE_BASE):
        from google.colab import drive
        drive.mount('/content/drive')
    assert os.path.isdir(DRIVE_BASE), f'Drive mounted, but {DRIVE_BASE} is absent'
    print(f'DRIVE_BASE recovered locally -> {DRIVE_BASE}')

COMBINED_DIR = f'{DRIVE_BASE}/module4_combined'
MODULE5_DIR  = f'{DRIVE_BASE}/module5'
CKPT_BEST    = f'{MODULE5_DIR}/diffusion_ckpt_best.pth'
CKPT_LATEST  = f'{MODULE5_DIR}/diffusion_ckpt.pth'
USE_BEST_CKPT = True     # the best-validating EMA snapshot, which is what the paper
                          # quotes. Set False to export the final-epoch weights instead.

# Sampling budget. Matches Module 6 exactly, so the numbers this cell prints are
# comparable with Module 6's own -- a different DDIM_STEPS or DDIM_SAMPLES would give a
# different prediction from identical weights.
DDIM_STEPS    = 20
DDIM_SAMPLES  = 4
EVAL_ROWS     = 64

# Calibration. Module 6's defaults, repeated here rather than imported, because this
# cell has to stand on its own -- and flagged in the payload either way.
CALIBRATE         = True
APPLY_CALIBRATION = True
CALIB_PER_VIDEO   = 24
CALIB_MAX         = 3.0

SEED = 2026
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
print('  MODULE 9 — Demo export (standalone: Drive only, no Module 6 needed)')
print('=' * 64)
print(f'  Device          : {DEVICE}')
print(f'  Checkpoint      : {os.path.basename(_ckpt_path)}')
print(f'  Videos          : {len(video_order)}')
print(f'  Anchors / frame : {N_ANCHOR}   |  context window W = {WINDOW_SIZE}')
print(f'  Trained horizons: {PRED_STEPS}')


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
        self.context_geo = GeomMLP(6, hidden_dim, D, dropout)
        # Overwritten by load_state_dict -- see Module 5's encode_context.
        self.register_buffer('ctx_scale', torch.tensor(1.0))
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
        # Modules 6, 7 and 8 load Module 4's full-width window, which may be wider
        # than this model was trained on. Take the most recent frames here, so the
        # checkpoint is always fed the context it actually saw in training and no
        # caller has to remember to slice.
        _w = self.temporal_pe.shape[0]
        if ctx.shape[1] > _w:
            ctx = ctx[:, -_w:]
        B, Wn, N_A, _ = ctx.shape
        # Motion relative to the last frame, plus that frame's absolute position.
        # Must match Module 5's encode_context exactly or the checkpoint is being
        # fed a different input representation than it was trained on.
        last = ctx[:, -1:]
        feat = torch.cat([(ctx - last) * self.ctx_scale,
                          last.expand(-1, Wn, -1, -1)], dim=-1)
        h = self.context_geo(feat)
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

BUNDLE = load_bundle(_ckpt_path)
n_params = sum(p.numel() for p in BUNDLE['model'].parameters())
print(f'  Model           : {n_params:,} params  |  trained to epoch {BUNDLE["epoch"]}  '
      f'|  best val CD {BUNDLE["best_cd"]:.6f}')
print(f'  DATA_SCALE      : {BUNDLE["data_scale"]:.3f}   (inverted before every metric)')
print(f'  Target mode     : {BUNDLE["target_mode"]}' + ('   (network output is the correction '
      'to constant velocity)' if BUNDLE['target_mode'] == 'cv_residual' else
      '   (network output is the displacement itself)'))

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

def batched(items, rows):
    for i in range(0, len(items), rows):
        yield items[i:i + rows]

# ════════════════════════════════════════════════════════════════════
# THE EXPORT — lifted verbatim from module9_demo_export.py
#
# Identical code, so the two cells cannot write different payloads. The DEMO_HORIZONS
# below is the one the calibration block above was narrowed to; it is defined there in
# the normal cell too, which is why this block has to come after the config and before
# the fit in the generated order.
# ════════════════════════════════════════════════════════════════════
DEMO_DIR      = f'{DRIVE_BASE}/module9_demo'
DEMO_POINTS   = 256     # of N_ANCHOR, evenly strided -- a browser canvas cannot usefully
                         # draw 512 overlapping points at 30fps, and the strided subset is
                         # unbiased because Module 3's anchor order carries no spatial sort
DEMO_ORIGINS  = 80      # frames the viewer can step through
DEMO_HORIZONS = [1, 3]  # one short horizon and one long one, so the demonstrator can show
                         # how the comparison and the uncertainty change with distance
DEMO_SAMPLES  = 8       # hypotheses per origin per horizon for the per-point spread. Higher
                         # than the metric path's 4: this feeds a visual, where a noisy
                         # radius reads as a bug, and it costs one origin's sampling time
DEMO_ROUND    = 4
DEMO_FRAME_W  = 320     # exported endoscopic frame width, px
DEMO_FRAME_Q  = 75      # JPEG quality

os.makedirs(DEMO_DIR, exist_ok=True)

# ════════════════════════════════════════════════════════════════════
# CALIBRATION — one scalar per horizon, fitted on TRAINING origins
#
# Lifted verbatim from Module 6, which is what makes the alpha below the same alpha the
# paper's headline numbers use. predict_direct defaults to alpha=1.0, so without this
# the demonstrator would show a different predictor than the write-up describes.
#
# Narrowed to the horizons the demo actually exports. Each horizon's fit is independent
# -- num and den are reset at the top of every k -- so the alpha for a retained horizon
# is bit-identical to the one a full-PRED_STEPS loop would produce. All that changes is
# that horizons nothing will be exported at are not paid for.
# ════════════════════════════════════════════════════════════════════
_unsupported = [k for k in DEMO_HORIZONS if k not in PRED_STEPS]
assert not _unsupported, (
    f'DEMO_HORIZONS asks for k={_unsupported}, which the model was never trained on '
    f'(trained: {PRED_STEPS}). Direct prediction is only valid at trained horizons.')
PRED_STEPS_TRAINED = list(PRED_STEPS)
PRED_STEPS = [k for k in PRED_STEPS_TRAINED if k in DEMO_HORIZONS]
print('\n' + '=' * 64)
print(f'  CALIBRATION — fitted on training origins, for k={PRED_STEPS}')
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

# Module 6 fits one magnitude scalar per horizon on TRAINING origins and reports its
# headline numbers with it applied. predict_direct defaults to alpha=1.0, so an export
# that does not pass it would quietly show a different predictor than the paper
# describes -- and the spread in sigma{k} would be on a different scale than the
# prediction it surrounds. Read from Module 6 if it ran the fit; 1.0 otherwise, which is
# the uncalibrated predictor and is recorded as such in the payload.
_cal_on = bool(globals().get('CALIBRATE', False) and globals().get('APPLY_CALIBRATION', False))
DEMO_ALPHA = {k: float(globals().get('alpha', {}).get(k, 1.0)) if _cal_on else 1.0
              for k in DEMO_HORIZONS}

def needed_frame_indices(origins, horizons):
    """Absolute trajectory-frame indices the export actually references.

    Module 9's own convention, read off the loop below rather than re-derived: ctx is
    tn[t-W:t], so the last OBSERVED frame of origin t is t-1; gt{k} is tn[t+k-1]."""
    idx = set()
    for t in origins:
        idx.add(int(t) - 1)
        for k in horizons:
            idx.add(int(t) + int(k) - 1)
    return sorted(idx)

def frame_file_map(indices):
    """Absolute frame index -> the path the page loads it from, relative to the page."""
    return {int(i): f'frames/{int(i):05d}.jpg' for i in indices}

def motion_by_horizon(frames, horizons):
    """Mean distance the tissue actually travelled, per horizon, in normalised units.

    This is the denominator of the app's skill score: error as a fraction of the motion
    there was to predict. A window in which nothing moved returns 0.0 -- the truth, and
    the value the app is specified to turn into "unknown" rather than dividing by."""
    out = {}
    for k in horizons:
        key = f'gt{k}'
        rows = [f for f in frames if key in f]
        if not rows:
            continue
        out[str(k)] = float(np.mean([
            np.linalg.norm(np.asarray(f[key]) - np.asarray(f['ctx']), axis=-1).mean()
            for f in rows]))
    return out

def camera_from(meta, frame_hw):
    """Pinhole intrinsics for drawing predictions onto the endoscopic frame.

    Module 6 projects with exactly these and states the trajectories are already in
    camera space -- "World (= camera) points" -- so no extrinsics are involved. When
    camera_meta.json is absent Module 1's documented fallback is a 75-degree horizontal
    FOV with a centred principal point; the overlay is then plausible but NOT measured,
    and 'calibrated' carries that distinction into the payload so the app can say so
    rather than presenting a guess with the confidence of a measurement."""
    if meta:
        return {'fx': float(meta['fx']), 'fy': float(meta['fy']),
                'cx': float(meta['cx']), 'cy': float(meta['cy']),
                'W': int(meta['W']), 'H': int(meta['H']), 'calibrated': True}
    H_, W_ = int(frame_hw[0]), int(frame_hw[1])
    fx = (W_ / 2) / math.tan(math.radians(75.0) / 2)
    return {'fx': fx, 'fy': fx, 'cx': W_ / 2.0, 'cy': H_ / 2.0,
            'W': W_, 'H': H_, 'calibrated': False}

def project_norm(world, cam):
    """Camera-space points -> image coordinates in [0, 1].

    Normalised rather than in pixels so the payload does not depend on the size the
    frames were exported at, and the browser can draw at whatever size the image
    happens to be displayed. z is clipped away from zero: a point at or behind the
    camera would otherwise produce a non-finite coordinate, and one NaN reaching a
    canvas path discards the whole path silently."""
    world = np.asarray(world, dtype=np.float64)
    z = np.clip(world[..., 2], 1e-6, None)
    u = (world[..., 0] / z * cam['fx'] + cam['cx']) / cam['W']
    v = (world[..., 1] / z * cam['fy'] + cam['cy']) / cam['H']
    return np.stack([u, v], -1)

def js_payload(obj):
    """The identical payload as a classic script, because file:// forbids fetch() and
    ES modules but permits <script src>. This is how the demonstrator loads its data
    off a USB stick on a laptop with no web server."""
    return 'window.DEMO_DATA = ' + json.dumps(obj, separators=(',', ':')) + ';\n'

def _subsample(n_total, n_want):
    return np.linspace(0, n_total - 1, min(n_want, n_total)).astype(int)

def _timing_for(ws, indices):
    """timing_ok / fps / gap_min / gap_max for this workspace, and nothing derived.

    Kept trajectory frames are not evenly spaced in time: Module 1 takes every second
    raw frame and then drops blurred ones, so consecutive trajectory frames can be 2 or
    10 raw frames apart. The app states the horizon in FRAMES, which is exact, and shows
    elapsed time only as a range and only when timing_ok. A single averaged millisecond
    figure over origins whose real gaps differ fivefold would be a fabrication.

    Module 6 recovers this at the timeline section, which is well past the point where a
    partial Module 6 run would have stopped -- so absent timing is normal, not an error,
    and the app falls back to frames-only."""
    tim = (globals().get('timing_by_ws') or {}).get(ws)
    if not tim or not tim.get('ok') or not tim.get('raw_index'):
        return {'timing_ok': False}
    raw = np.asarray(tim['raw_index'])
    lo, hi = int(min(indices)), int(max(indices))
    if hi >= len(raw) or hi <= lo:
        return {'timing_ok': False}
    gaps = np.diff(raw[lo:hi + 1])
    if gaps.size == 0:
        return {'timing_ok': False}
    return {'timing_ok': True, 'fps': float(tim.get('fps') or 0.0),
            'gap_min': int(gaps.min()), 'gap_max': int(gaps.max())}

def _frame_hw(ws):
    """(H, W) of this workspace's frames, for the uncalibrated-intrinsics fallback."""
    src_dir = f'{DRIVE_BASE}/{ws}/frames'
    try:
        import cv2
        import glob as _glob
        hit = sorted(_glob.glob(f'{src_dir}/*.png') + _glob.glob(f'{src_dir}/*.jpg'))
        if hit:
            im = cv2.imread(hit[0])
            if im is not None:
                return int(im.shape[0]), int(im.shape[1])
    except Exception:
        pass
    return 360, 640

def _export_frames(ws, indices):
    """Write frames/<index>.jpg for every index the payload references.

    Real endoscopic imagery, downscaled, referenced by <img src> -- which file:// allows.
    A missing frames/ directory on Drive is not fatal: the app is specified to show its
    "imagery comes from the export" panel rather than a broken-image icon, so this
    returns an empty map and says so."""
    src_dir = f'{DRIVE_BASE}/{ws}/frames'
    if not os.path.isdir(src_dir):
        print(f'    frames: {src_dir} absent — payload exports geometry only')
        return {}
    import cv2
    out_dir = f'{DEMO_DIR}/frames'
    os.makedirs(out_dir, exist_ok=True)
    written, missing = {}, 0
    for i in indices:
        src = next((p for p in (f'{src_dir}/{i:05d}.png', f'{src_dir}/{i:05d}.jpg')
                    if os.path.exists(p)), None)
        if src is None:
            missing += 1
            continue
        im = cv2.imread(src)
        if im is None:
            missing += 1
            continue
        h, w = im.shape[:2]
        if w > DEMO_FRAME_W:
            im = cv2.resize(im, (DEMO_FRAME_W, max(1, round(h * DEMO_FRAME_W / w))),
                            interpolation=cv2.INTER_AREA)
        cv2.imwrite(f'{out_dir}/{i:05d}.jpg', im,
                    [int(cv2.IMWRITE_JPEG_QUALITY), DEMO_FRAME_Q])
        written[int(i)] = f'frames/{int(i):05d}.jpg'
    mb = sum(os.path.getsize(f'{out_dir}/{os.path.basename(v)}')
             for v in written.values()) / 1e6
    print(f'    frames: {len(written)}/{len(indices)} written to {out_dir}  ({mb:.1f} MB)'
          + (f'  — {missing} not on Drive' if missing else ''))
    return written

@torch.no_grad()
def export_demo(vid, name):
    tn, dn = vid['traj_n'], vid['disp_norm']
    T_all = tn.shape[0]
    kmax = max(DEMO_HORIZONS)
    # HELD-OUT ONLY. Module 4 split each video temporally 80/20 and 'split_frame' is where
    # the validation tail begins, so an origin at or past it contributed no gradient step.
    # A demo that replays training frames is not a demo of prediction, it is a demo of
    # memorisation, and it is the first thing an examiner should be able to rule out.
    lo = max(WINDOW_SIZE, int(vid['split_frame']))
    hi = T_all - kmax
    assert hi > lo, (f'{name}: held-out tail runs {lo}..{T_all}, too short for '
                     f'W={WINDOW_SIZE} + k={kmax}')

    origins = (_subsample(hi - lo, DEMO_ORIGINS) + lo).tolist()
    pts = _subsample(tn.shape[1], DEMO_POINTS)

    # ── Overlay: the same points, in image coordinates ───────────────────────────
    # The trajectories here are normalised; Module 6's denorm (x * scl + ctr) returns
    # them to camera space, which is the space its own project() expects -- it calls
    # them "World (= camera) points", so there are no extrinsics in this at all. The
    # projection is done HERE rather than in the browser so the page carries no camera
    # maths and no assumption of its own; it just draws where it is told.
    _cam_path = f'{DRIVE_BASE}/{vid["workspace"]}/camera_meta.json'
    _meta = None
    if os.path.exists(_cam_path):
        with open(_cam_path) as fh:
            _meta = json.load(fh)
    cam = camera_from(_meta, _frame_hw(vid['workspace']))
    if not cam['calibrated']:
        print(f'    camera_meta.json absent — overlay uses Module 1\'s 75° fallback '
              f'and is flagged uncalibrated')

    def to_world(a):
        return np.asarray(a) * vid['scl'] + vid['ctr']

    frames = []

    for t in origins:
        ctx = tn[t - WINDOW_SIZE:t].unsqueeze(0)              # (1, W, N_A, 3)
        row = {'t': int(t), 'ctx': tn[t - 1][pts].cpu().numpy(), 'img': int(t) - 1}
        row['px_ctx'] = project_norm(to_world(row['ctx']), cam)
        for k in DEMO_HORIZONS:
            a = DEMO_ALPHA[k]
            hs = torch.full((1,), k, device=DEVICE, dtype=torch.long)
            row[f'gt{k}']   = tn[t + k - 1][pts].cpu().numpy()
            row[f'pred{k}'] = predict_direct(ctx, k, disp_norm=dn, alpha=a)[0][pts].cpu().numpy()
            row[f'cv{k}']   = base_pred(ctx, hs)[0][pts].cpu().numpy()
            row[f'img{k}']  = int(t) + int(k) - 1

            # Per-point uncertainty AT THIS HORIZON: independent futures from the
            # identical context, so the disagreement between them is the model's own and
            # not a difference in what it saw. One batched call of DEMO_SAMPLES rows
            # rather than DEMO_SAMPLES calls -- ddim_sample's own n_samples argument
            # averages its trajectories and would return the mean, which has no spread.
            hyps = predict_direct(ctx.repeat(DEMO_SAMPLES, 1, 1, 1), k, disp_norm=dn,
                                  eta=1.0, n_samples=1, alpha=a)       # (S, N_A, 3)
            row[f'sigma{k}'] = hyps.std(dim=0).norm(dim=-1)[pts].cpu().numpy()
            for tag in ('gt', 'pred', 'cv'):
                row[f'px_{tag}{k}'] = project_norm(to_world(row[f'{tag}{k}']), cam)
        frames.append(row)

    # One shared normalisation so the viewer never has to rescale mid-playback, which
    # would make motion look like camera movement.
    allpts = np.concatenate([f['ctx'] for f in frames] +
                            [f[f'gt{k}'] for f in frames for k in DEMO_HORIZONS])
    lo3, hi3 = allpts.min(axis=0), allpts.max(axis=0)
    centre = ((lo3 + hi3) / 2).tolist()
    radius = float(np.abs(allpts - np.array(centre)).max()) or 1.0

    def enc(a):
        return [round(float(v), DEMO_ROUND) for v in np.asarray(a).ravel()]

    err = {}
    for k in DEMO_HORIZONS:
        e_ours = float(np.mean([np.linalg.norm(f[f'pred{k}'] - f[f'gt{k}'], axis=-1).mean()
                                for f in frames]))
        e_cv   = float(np.mean([np.linalg.norm(f[f'cv{k}'] - f[f'gt{k}'], axis=-1).mean()
                                for f in frames]))
        e_still = float(np.mean([np.linalg.norm(f['ctx'] - f[f'gt{k}'], axis=-1).mean()
                                 for f in frames]))
        err[str(k)] = {'ours': e_ours, 'cv': e_cv, 'still': e_still}

    idx = needed_frame_indices(origins, DEMO_HORIZONS)
    frame_files = _export_frames(vid['workspace'], idx)

    out = {
        'video': name,
        'workspace': vid['workspace'],
        'n_points': int(len(pts)),
        'n_frames': int(len(frames)),
        'horizons': DEMO_HORIZONS,
        'window': int(WINDOW_SIZE),
        'centre': centre,
        'radius': radius,
        # The reconstructed tissue's extent in the same normalised units as every point
        # array, stated here rather than left for the app to assume. See the design note
        # on expressing error: span is a denominator the app has, not the one it leads
        # with, because every method in this project looks excellent against it.
        'span': 2.0 * radius,
        'target_mode': BUNDLE.get('target_mode', BUNDLE.get('mode', 'delta')),
        'checkpoint_epoch': BUNDLE.get('epoch', '?'),
        # Mean distance actually travelled per horizon: the denominator the app's skill
        # score uses instead of span.
        'motion': motion_by_horizon(frames, DEMO_HORIZONS),
        'calibrated': _cal_on,
        'alpha': {str(k): DEMO_ALPHA[k] for k in DEMO_HORIZONS},
        # Horizons where the fitted alpha is 0: the fit found the model's contribution
        # worth nothing there, so pred{k} is EXACTLY cv{k} and sigma{k} is identically
        # zero. Stated rather than left to be inferred, because the alternative is a
        # viewer drawing a zero-radius uncertainty cloud around a prediction it presents
        # as the model's, which is the most misleading thing this payload could carry.
        'degenerate_horizons': [str(k) for k in DEMO_HORIZONS if DEMO_ALPHA[k] == 0.0],
        # Which intrinsics the overlay used, and whether they were measured. An
        # uncalibrated overlay is plausible, not correct, and the app says so.
        'camera': cam,
        'timing': _timing_for(vid['workspace'], idx),
        'frame_files': {str(i): p for i, p in frame_files.items()},
        'mean_error': err,
        'frames': [
            dict({'t': f['t'], 'ctx': enc(f['ctx']), 'img': f['img'],
                  'px_ctx': enc(f['px_ctx'])},
                 **{f'{tag}{k}': (enc(f[f'{tag}{k}']) if tag != 'img' else f[f'img{k}'])
                    for k in DEMO_HORIZONS
                    for tag in ('gt', 'pred', 'cv', 'sigma', 'img')},
                 **{f'px_{tag}{k}': enc(f[f'px_{tag}{k}'])
                    for k in DEMO_HORIZONS
                    for tag in ('gt', 'pred', 'cv')})
            for f in frames
        ],
    }
    p_json = f'{DEMO_DIR}/demo_data.json'
    p_js   = f'{DEMO_DIR}/demo_data.js'
    with open(p_json, 'w') as fh:
        json.dump(out, fh, separators=(',', ':'))
    with open(p_js, 'w') as fh:
        fh.write(js_payload(out))
    print(f'  {name}: {len(frames)} origins x {len(pts)} points')
    print(f'    {p_json}  ({os.path.getsize(p_json) / 1e6:.1f} MB)')
    print(f'    {p_js}  ({os.path.getsize(p_js) / 1e6:.1f} MB)   <- the one file:// can load')
    for k in DEMO_HORIZONS:
        e, mo = err[str(k)], out['motion'].get(str(k), 0.0)
        best = min(e, key=e.get)
        skill = f'{100.0 * e["ours"] / mo:.1f}% of motion' if mo > 0 else 'motion ~ 0'
        print(f'    k={k}  ours {e["ours"]:.5f}   const-vel {e["cv"]:.5f}   '
              f'no-motion {e["still"]:.5f}   -> lowest: {best}   ({skill})')
    if out['degenerate_horizons']:
        ks = ', '.join(f'k={k}' for k in out['degenerate_horizons'])
        print(f'    WARNING  alpha = 0 at {ks}. The calibration fit found the model\'s'
              f' contribution\n             worthless there, so the prediction at that'
              f' horizon IS constant velocity\n             and its spread is identically'
              f' zero. Recorded in degenerate_horizons.')
    return out

# ════════════════════════════════════════════════════════════════════
# RUN
# ════════════════════════════════════════════════════════════════════
_vid = max(videos, key=lambda v: v['n_frames'] - int(v['split_frame']))
_name = str(_vid.get('video_id') or _vid.get('workspace') or 'video')
print(f'  Chosen video  : {_name}  ({_vid["n_frames"]} frames, held-out tail starts at '
      f'{_vid["split_frame"]})')
print(f'  Horizons      : {DEMO_HORIZONS} frames   |   calibration applied: {_cal_on}'
      + ('   alpha ' + ', '.join(f'k{k}={DEMO_ALPHA[k]:.3f}' for k in DEMO_HORIZONS)
         if _cal_on else ''))
export_demo(_vid, _name)
print(f'\n  Copy demo_data.js and frames/ out of {DEMO_DIR} to sit beside')
print('  demo/clinical/index.html, then open index.html in a browser.')
