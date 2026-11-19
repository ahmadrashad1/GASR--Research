# ════════════════════════════════════════════════════════════════════
# MODULE 5 — QUALITY TESTS ONLY  (no training, read-only)
#
# GENERATED FILE -- do not edit. Produced by tools/make_qt_cell.py, which lifts the
# model, the sampler, the metrics and the six tests verbatim out of
# module5_diffusion.py. Edit Module 5 and re-run that script.
#
# Runs QT1-QT6 against the checkpoint already saved on Drive, on the FULL validation
# set. It does not train, and it never writes a checkpoint: there is no torch.save and
# no os.rename anywhere in this file, so it cannot cost you the run that produced the
# weights.
#
# Every architectural and scaling constant is read FROM THE CHECKPOINT, not from
# constants here. That matters: if this file declared WINDOW_SIZE itself and disagreed
# with the checkpoint, the weights would refuse to load -- or worse, load into a model
# whose normalisation differs and report confident nonsense. The checkpoint is the
# single source of truth about the model it contains.
#
# Runtime: roughly 20-25 minutes. QT2 alone samples FINAL_SAMPLES trajectories x
# FINAL_STEPS DDIM steps for every validation window.
# ════════════════════════════════════════════════════════════════════
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, os, math, json, time, hashlib
from tqdm import tqdm

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

# Which checkpoint to score. 'best' is the one the paper should quote; 'latest' is
# wherever training stopped. They differ whenever the best validation score was not on
# the final epoch.
WHICH        = 'best'              # 'best' | 'latest'
# The checkpoint fingerprints the Module 4 arrays it was trained against. Set this False
# only if you deliberately want to score the model on a DIFFERENT dataset, and then do
# not quote the result as this model's accuracy.
REQUIRE_MATCHING_DATA = True

FINAL_STEPS   = 20                 # DDIM steps for QT2 -- matches Module 5's final QT2
FINAL_SAMPLES = 4                  # trajectories averaged per window
EVAL_ROWS     = 64                 # windows * n_samples resident at once
SEED          = 2026

torch.manual_seed(SEED)
np.random.seed(SEED)

print('=' * 70)
print('  MODULE 5 — QUALITY TESTS (read-only, no training)')
print('=' * 70)

# ── Checkpoint ───────────────────────────────────────────────────────
_path = CKPT_BEST if WHICH == 'best' else CKPT_LATEST
if not os.path.exists(_path):
    _other = CKPT_LATEST if WHICH == 'best' else CKPT_BEST
    assert os.path.exists(_other), (
        f'No checkpoint in {MODULE5_DIR}. Looked for {os.path.basename(CKPT_BEST)} and '
        f'{os.path.basename(CKPT_LATEST)}. There is nothing to evaluate.')
    print(f'  ⚠️  {os.path.basename(_path)} is absent; using '
          f'{os.path.basename(_other)} instead.')
    _path = _other

# Drive buffers writes, and this repository has already lost one large file that way, so
# say how big it is rather than letting a short read surface as a cryptic unpickling error.
print(f'  Checkpoint    : {os.path.basename(_path)} '
      f'({os.path.getsize(_path) / 1e6:.1f} MB)')
try:
    ck = torch.load(_path, map_location=DEVICE, weights_only=False)
except Exception as _e:
    raise SystemExit(
        f'\n  ✗ {_path} failed to load: {type(_e).__name__}: {_e}\n'
        f'    This cell does NOT rename or replace it -- it is read-only. If Drive\n'
        f'    truncated the write, the size above will be short of the model size.')

arch = ck.get('arch_cfg') or {}
cfg  = ck.get('config') or {}
assert arch, ('This checkpoint carries no arch_cfg, so the architecture it was trained '
              'with cannot be recovered. It predates horizon conditioning and cannot be '
              'evaluated by this cell.')

# Every constant below comes from the checkpoint. Nothing here second-guesses it.
WINDOW_SIZE   = int(arch['WINDOW_SIZE'])
LATENT_DIM    = int(arch['LATENT_DIM'])
HIDDEN_DIM    = int(arch['HIDDEN_DIM'])
N_HEADS       = int(arch['N_HEADS'])
N_TF_LAYERS   = int(arch['N_TF_LAYERS'])
N_ANCHOR      = int(arch['N_ANCHOR'])
T_DIFF        = int(arch['T_DIFF'])
MAX_PRED_STEP = int(arch['MAX_PRED_STEP'])
TARGET_MODE   = arch.get('TARGET_MODE', 'delta')
CTX_MODE      = arch.get('CTX_MODE', 'abs3')
DATA_SCALE    = float(cfg['DATA_SCALE'])
CTX_SCALE     = float(cfg.get('CTX_SCALE', 1.0))
DROPOUT       = 0.1                # eval-mode only; affects no parameter shape
# Nothing is trained here. BATCH_SIZE is read only because QT3, QT4 and QT6 use it to
# size their shared diagnostic subset, and taking Module 5's own value keeps those tests
# scoring the same number of windows they did during the run.
BATCH_SIZE    = int(cfg.get('BATCH_SIZE', 32))

assert CTX_MODE == 'rel6', (
    f"This checkpoint was trained with CTX_MODE={CTX_MODE!r}, but the context encoder in "
    f"this file takes 6 relative channels ('rel6'). Its weights have a 3-channel first "
    f"layer and different semantics throughout, so loading them here would report "
    f"confident nonsense rather than fail. Evaluate it with the Module 5 revision that "
    f"trained it.")

print(f'  Trained to    : epoch {ck.get("epoch", "?")}  |  best subset CD '
      f'{ck.get("best_cd", float("nan")):.6f}')
print(f'  Architecture  : W={WINDOW_SIZE}  latent={LATENT_DIM}  hidden={HIDDEN_DIM}  '
      f'heads={N_HEADS}  layers={N_TF_LAYERS}  anchors={N_ANCHOR}')
print(f'  Target mode   : {TARGET_MODE}' + ('   (network output is the correction to '
      'constant velocity)' if TARGET_MODE == 'cv_residual' else ''))
print(f'  DATA_SCALE    : {DATA_SCALE:.3f}   CTX_SCALE: {CTX_SCALE:.3f}'
      '   (both inverted before every reported metric)')

# ── Validation data, full set ────────────────────────────────────────
val_inp   = np.load(f'{COMBINED_DIR}/val_inputs.npy').astype(np.float32)
val_tgt   = np.load(f'{COMBINED_DIR}/val_targets.npy').astype(np.float32)
val_dnorm = np.load(f'{COMBINED_DIR}/val_disp_norm.npy').astype(np.float32)
val_ps    = np.load(f'{COMBINED_DIR}/val_pred_step.npy').astype(np.int64)

# Provenance, the same check Modules 5-8 make. *_inputs/_targets are written by the
# notebook's original Module 4 cell too, at a different window, while *_disp_norm and
# *_pred_step come only from module4_improved -- so the two can describe different
# datasets, with different window counts.
assert len(val_inp) == len(val_ps), (
    f'val_inputs.npy holds {len(val_inp):,} windows but val_pred_step.npy holds '
    f'{len(val_ps):,}. The arrays in {COMBINED_DIR} come from two different Module 4 '
    f'runs -- re-run Module 4 (module4_improved), then re-run this cell.')

_W_DATA = val_inp.shape[1]
assert _W_DATA >= WINDOW_SIZE, (
    f'The checkpoint needs {WINDOW_SIZE} context frames but Module 4 emitted '
    f'{_W_DATA}. Re-run Module 4 with a WINDOW_SIZE of at least {WINDOW_SIZE}.')
if _W_DATA > WINDOW_SIZE:
    val_inp = val_inp[:, -WINDOW_SIZE:]
    print(f'  Context       : last {WINDOW_SIZE} of the {_W_DATA} frames Module 4 built')

assert val_inp.shape[2] == N_ANCHOR, (
    f'Module 4 emitted {val_inp.shape[2]} anchors but the checkpoint expects {N_ANCHOR}.')

# The deltas, for the dataset fingerprint below. Shape is (M, N_ANCHOR, 3) -- not
# windowed -- so this is tens of megabytes, not the gigabyte the context arrays are.
train_delta = np.load(f'{COMBINED_DIR}/train_deltas.npy').astype(np.float32)
val_delta   = np.load(f'{COMBINED_DIR}/val_deltas.npy').astype(np.float32)

M_va = len(val_inp)
val_inp_t   = torch.from_numpy(val_inp).to(DEVICE)
val_tgt_t   = torch.from_numpy(val_tgt).to(DEVICE)
val_dnorm_t = torch.from_numpy(val_dnorm).to(DEVICE)
val_ps_t    = torch.from_numpy(val_ps).to(DEVICE)
print(f'  Val windows   : {M_va:,}  (full set, no subsampling)')
print(f'  Sampling      : {FINAL_STEPS} DDIM steps x {FINAL_SAMPLES} averaged samples')

# ── Diffusion schedule ──────────────────────────────────────────────
def representative_subset_indices(m, target_size):
    """Evenly-strided indices across the full array, not the first `target_size` rows.
    Module 4 concatenates val windows as contiguous per-video blocks, so val_inp[:N] for
    small N is dominated by whichever 1-2 videos happen to come first -- exactly the bias
    that made QT3/QT4's old 32-window diagnostic (and the periodic training-time check,
    before this fix) an unreliable stand-in for full-dataset quality. A stride smaller
    than the smallest per-video block guarantees every video contributes some windows."""
    stride = max(1, m // target_size)
    return np.arange(0, m, stride)

# ════════════════════════════════════════════════════════════════════
# MODEL
# ════════════════════════════════════════════════════════════════════
class GeomMLP(nn.Module):
    """Pointwise 4-layer encoder shared across all N_ANCHOR points."""
    def __init__(self, in_dim=3, hidden=HIDDEN_DIM, out_dim=LATENT_DIM, dropout=DROPOUT):
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
    """
    Context: W past absolute-position frames -> pointwise GeomMLP -> temporal PE ->
    per-point temporal self-attention transformer (attention over the W axis only,
    points folded into the batch dim -- cheap in FLOPs since attention is over a
    short W=4 sequence, but this is what makes activation memory and per-step op
    count scale with BATCH_SIZE*N_ANCHOR; see MICRO_BATCH/N_TF_LAYERS comments in
    CONFIG for the OOM and throughput consequences of that) -> last-token-pooled
    per-point context latent (B, N_A, D) (most recent frame, not mean-pooled --
    see encode_context for why).
    Noisy target: pointwise GeomMLP + sinusoidal timestep embedding -> (B, N_A, D).
    Two bidirectional cross-attention layers over the point axis (N_A tokens) let
    context and target exchange geometric information, then a 3-layer output MLP
    decodes the fused representation to predicted velocity (B, N_A, 3).
    """
    def __init__(self):
        super().__init__()
        D = LATENT_DIM
        self.context_geo = GeomMLP(in_dim=6)   # motion (3) + absolute anchor (3)
        self.target_geo  = GeomMLP()
        # A buffer, not a constructor argument: it travels inside the state_dict, so
        # Modules 6-8 rebuild the identical input transform from the checkpoint alone
        # and there is no second place for the number to drift out of sync.
        self.register_buffer('ctx_scale', torch.tensor(float(CTX_SCALE)))
        self.temporal_pe = nn.Parameter(torch.randn(WINDOW_SIZE, D) * 0.02)

        tf_layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=N_HEADS, dim_feedforward=HIDDEN_DIM,
            dropout=DROPOUT, activation='gelu', batch_first=True, norm_first=True)
        self.temporal_transformer = nn.TransformerEncoder(
            tf_layer, num_layers=N_TF_LAYERS, enable_nested_tensor=False)

        self.time_mlp = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))
        # Horizon conditioning. Injected the same way as the diffusion timestep -- a
        # per-window vector broadcast over all N_ANCHOR points -- because "how many frames
        # ahead am I predicting" is a property of the window, not of any individual point.
        # It is added to BOTH branches: to the target branch so the denoiser can scale its
        # velocity prediction with the horizon, and to the context branch so cross-attention
        # can read the window's motion at the right stride (a constant-velocity predictor,
        # which Module 4's QT7 shows already beats naive by 30-57%, is `last + v * k` and is
        # unrepresentable without k reaching the context features).
        self.horizon_emb = nn.Embedding(MAX_PRED_STEP + 1, D)
        self.horizon_mlp = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))
        nn.init.normal_(self.horizon_emb.weight, std=0.02)

        self.cross_t2c = nn.MultiheadAttention(D, N_HEADS, dropout=DROPOUT, batch_first=True)
        self.cross_c2t = nn.MultiheadAttention(D, N_HEADS, dropout=DROPOUT, batch_first=True)
        self.norm_t = nn.LayerNorm(D)
        self.norm_c = nn.LayerNorm(D)

        self.output_mlp = nn.Sequential(
            nn.Linear(2 * D, HIDDEN_DIM), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2), nn.GELU(),
            nn.Linear(HIDDEN_DIM // 2, 3),
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
        # Feed MOTION, not position. Module 4 normalises coordinates to [-1, 1], while a
        # single frame of tissue motion is ~0.004 of that range -- the signal is well
        # under 1% of the magnitude of the input carrying it. A pointwise encoder on raw
        # coordinates therefore spends its capacity describing where a point IS, and the
        # temporal transformer is left to recover movement by differencing tokens that
        # agree to three significant figures. That is the most likely origin of the
        # magnitude collapse Module 7 measured: the model reproduced the zero-motion
        # baseline to within 2% at every horizon, which is what a network outputs when it
        # cannot resolve the signal and falls back on the conditional mean.
        #
        # Every learned baseline in Module 7 already receives exactly this relative form
        # (see its prep_features: `rel = ctx - ctx[:, -1:]`). The diffusion model was the
        # only method in the comparison NOT given it, which is a fair part of why a
        # 136k-parameter MLP beat 16.3M parameters of conditional diffusion.
        #
        # The absolute last frame is concatenated back in so that spatially varying
        # deformation -- tissue near the instrument behaving differently from tissue at
        # the periphery -- stays representable. Relative-only would discard that.
        last = ctx[:, -1:]                                     # (B, 1, N_A, 3)
        feat = torch.cat([(ctx - last) * self.ctx_scale,       # motion, ~unit variance
                          last.expand(-1, Wn, -1, -1)],        # where that motion happens
                         dim=-1)                               # (B, W, N_A, 6)
        h = self.context_geo(feat)                             # (B, W, N_A, D)
        h = h + self.temporal_pe[None, :, None, :]
        h = h.permute(0, 2, 1, 3).reshape(B * N_A, Wn, -1)      # (B*N_A, W, D)
        h = self.temporal_transformer(h)                        # (B*N_A, W, D)
        # Last-token pooling (most recent context frame, index -1) rather than mean-pool:
        # for near-future prediction the most recent frame is the most predictive one, and
        # self-attention has already let it incorporate information from the older frames --
        # averaging back in the older frames' own tokens only dilutes that signal.
        h = h[:, -1, :]                                         # (B*N_A, D)
        return h.reshape(B, N_A, -1)                             # (B, N_A, D)

    def forward(self, ctx, x_noisy, t, hstep):
        h_emb = self.horizon_mlp(self.horizon_emb(hstep))[:, None, :]   # (B, 1, D)
        ctx_latent = self.encode_context(ctx) + h_emb            # (B, N_A, D)
        tgt_latent = self.target_geo(x_noisy)                    # (B, N_A, D)
        tgt_latent = tgt_latent + self.time_mlp(sinusoidal_embedding(t, tgt_latent.shape[-1]))[:, None, :]
        tgt_latent = tgt_latent + h_emb

        attn_t, _ = self.cross_t2c(tgt_latent, ctx_latent, ctx_latent)
        tgt_upd = self.norm_t(tgt_latent + attn_t)
        attn_c, _ = self.cross_c2t(ctx_latent, tgt_latent, tgt_latent)
        ctx_upd = self.norm_c(ctx_latent + attn_c)

        fused = torch.cat([tgt_upd, ctx_upd], dim=-1)             # (B, N_A, 2D)
        return self.output_mlp(fused)                              # (B, N_A, 3) predicted velocity

# ════════════════════════════════════════════════════════════════════
# DIFFUSION SCHEDULE (cosine, v-prediction)
# ════════════════════════════════════════════════════════════════════
def cosine_beta_schedule(T, s=0.008):
    steps = T + 1
    x = torch.linspace(0, T, steps)
    ac = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    ac = ac / ac[0]
    betas = 1 - (ac[1:] / ac[:-1])
    return torch.clip(betas, 1e-4, 0.999)

def pred_x0_from_v(x_t, v_pred, t):
    a = sqrt_ac[t][:, None, None]
    b = sqrt_1m_ac[t][:, None, None]
    return a * x_t - b * v_pred

def pred_eps_from_v(x_t, v_pred, t):
    a = sqrt_ac[t][:, None, None]
    b = sqrt_1m_ac[t][:, None, None]
    return b * x_t + a * v_pred

@torch.no_grad()
def ddim_sample(model, ctx, hstep, steps=50, eta=0.0, generator=None, n_samples=1):
    """Generalised DDIM (eta=0 deterministic, eta=1 ~ ancestral DDPM). Returns x0 in
    the model's *scaled* space (still needs /DATA_SCALE, /disp_norm to become physical).

    n_samples > 1 runs that many independent trajectories per window (different initial
    noise, batched together along the batch axis) and returns their MEAN. That mean is a
    Monte-Carlo estimate of E[x0 | context, horizon], which is the point prediction that
    minimises the squared error Chamfer Distance is scoring -- see the VAL_SAMPLES comment
    in CONFIG for why a single trajectory is ~sqrt(2)x worse than the model's own best
    achievable prediction. n_samples=1 gives the plain single-sample behaviour, which QT2
    still reports and QT3 needs (averaging would destroy the diversity it measures).

    Restores the model's previous train/eval mode rather than unconditionally calling
    .train() -- the old version left an EMA model in train mode after every evaluation.
    """
    was_training = model.training
    model.eval()
    B, N_A = ctx.shape[0], ctx.shape[2]
    if n_samples > 1:
        ctx = ctx.repeat_interleave(n_samples, dim=0)
        hstep = hstep.repeat_interleave(n_samples, dim=0)
    BK = ctx.shape[0]
    ts = torch.linspace(T_DIFF - 1, 0, steps, device=DEVICE).long()
    x = torch.randn(BK, N_A, 3, device=DEVICE, generator=generator)
    for i, t in enumerate(ts):
        t_batch = t.expand(BK)
        v_pred = model(ctx, x, t_batch, hstep)
        x0_pred = pred_x0_from_v(x, v_pred, t_batch)
        eps_pred = pred_eps_from_v(x, v_pred, t_batch)
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
    if was_training:
        model.train()
    if n_samples > 1:
        x = x.view(B, n_samples, N_A, 3).mean(dim=1)
    return x

@torch.no_grad()
def predict_mean_oneshot(model, ctx, hstep, generator=None):
    """One forward pass approximation of E[x0 | context, horizon], in scaled space.

    At t = T_DIFF-1 the cosine schedule has alpha_bar ~ 0, so x_t is essentially pure noise
    and carries no information about x0. The network's x0-estimate there is therefore its
    estimate of the conditional mean, obtained at 1/(n_samples*steps) of the cost of the
    ensemble average above (1 forward pass instead of 320+). Reported alongside the
    ensemble in QT2 as a cheap-inference option; the ensemble remains the headline number
    because this shortcut inherits whatever bias the network has at the single largest
    timestep, whereas the ensemble integrates over the whole trajectory."""
    was_training = model.training
    model.eval()
    B, N_A = ctx.shape[0], ctx.shape[2]
    t = torch.full((B,), T_DIFF - 1, device=DEVICE, dtype=torch.long)
    x_T = torch.randn(B, N_A, 3, device=DEVICE, generator=generator)
    x0 = pred_x0_from_v(x_T, model(ctx, x_T, t, hstep), t)
    if was_training:
        model.train()
    return x0

def chamfer_distance(pred, gt):
    """Symmetric Chamfer Distance. pred, gt: (B, N, 3) -> scalar (mean over batch)."""
    d = torch.cdist(pred, gt)
    d1 = d.min(dim=2)[0].mean(dim=1)
    d2 = d.min(dim=1)[0].mean(dim=1)
    return (d1 + d2).mean()

def to_physical(delta_scaled, dnorm):
    """Invert both scale transforms: model-space delta -> physical normalised-position delta."""
    return (delta_scaled / DATA_SCALE) / dnorm[:, None, None]

def const_velocity_pred(ctx, hstep):
    """Constant-velocity extrapolation: last + (last - previous) * horizon.

    Not used for training -- this is the honest reference bar for QT2. "Beats the naive
    zero-motion baseline" is a weak claim when Module 4's QT7 already shows that after
    adaptive temporal smoothing this two-line estimator beats naive by 30-57%. Reporting it
    makes it visible whether the diffusion model is actually adding anything over trivial
    linear extrapolation, which is the comparison a reader of the report will ask for."""
    last, prev = ctx[:, -1], ctx[:, -2]
    return last + (last - prev) * hstep.to(last.dtype)[:, None, None]

def base_pred(ctx, hstep):
    """The prediction the model's output is a correction TO -- i.e. what the scene looks
    like when the network emits exactly zero.

    Under TARGET_MODE='delta' that is the last observed frame (the naive baseline); under
    'cv_residual' it is constant-velocity extrapolation. Every place that turns a model
    output into a predicted geometry goes through here, so the two modes cannot drift
    apart and no evaluation can accidentally reconstruct a prediction the wrong way."""
    if TARGET_MODE == 'cv_residual':
        return const_velocity_pred(ctx, hstep)
    return ctx[:, -1]

# ════════════════════════════════════════════════════════════════════
# TRAINING LOOP — resumable, time-budgeted
# ════════════════════════════════════════════════════════════════════
def evaluate_cd(m, indices=None, steps=FINAL_STEPS, n_samples=FINAL_SAMPLES,
                 per_horizon=False, extra_baselines=False):
    """Window-count-weighted mean Chamfer Distance over the validation set, in physical
    (normalised-position) space.

    indices=None means the FULL validation set (used for the final QT2 number). Periodic
    in-training checks pass val_subset_idx instead -- a representative but much cheaper
    stand-in, which is what makes validating every VAL_EVERY epochs affordable.

    n_samples > 1 averages that many DDIM trajectories per window into a single prediction
    (see the VAL_SAMPLES comment in CONFIG): CD scores one point prediction, and the mean
    of the conditional is what minimises it, so scoring a lone random draw charges the
    model a ~sqrt(2)x error penalty it does not have to pay.

    Returns a dict so callers can pick the number they need without the signature growing
    a return-tuple per baseline. 'cd' is always the headline model number.
    """
    idx = np.arange(M_va) if indices is None else indices
    rows = max(1, EVAL_ROWS // max(1, n_samples))     # windows per batch, so that
                                                       # windows*n_samples stays ~EVAL_ROWS
    acc = {k: 0.0 for k in ('cd', 'cd_naive', 'cd_single', 'cd_cv', 'cd_mean1')}
    per_h = {}
    n = 0
    with torch.no_grad():
        for bi in range(0, len(idx), rows):
            sl = idx[bi:bi + rows]
            if len(sl) == 0:
                continue
            ctx   = val_inp_t[sl]
            tgt   = val_tgt_t[sl]
            dnorm = val_dnorm_t[sl]
            hstep = val_ps_t[sl]

            x0 = ddim_sample(m, ctx, hstep, steps=steps, eta=0.0, n_samples=n_samples)
            base = base_pred(ctx, hstep)
            pred = base + to_physical(x0, dnorm)
            cd_b = chamfer_distance(pred, tgt)
            naive_b = chamfer_distance(ctx[:, -1], tgt)
            acc['cd'] += cd_b.item() * len(sl)
            acc['cd_naive'] += naive_b.item() * len(sl)

            if extra_baselines:
                # Single-trajectory number: keeps the cost of sampling (rather than taking
                # the conditional mean) visible instead of quietly optimised away.
                x0_1 = ddim_sample(m, ctx, hstep, steps=steps, eta=0.0, n_samples=1)
                acc['cd_single'] += chamfer_distance(
                    base + to_physical(x0_1, dnorm), tgt).item() * len(sl)
                # One-forward-pass conditional-mean shortcut.
                x0_m = predict_mean_oneshot(m, ctx, hstep)
                acc['cd_mean1'] += chamfer_distance(
                    base + to_physical(x0_m, dnorm), tgt).item() * len(sl)
                # Constant velocity -- the real bar (Module 4 QT7).
                acc['cd_cv'] += chamfer_distance(
                    const_velocity_pred(ctx, hstep), tgt).item() * len(sl)

            if per_horizon:
                for k in torch.unique(hstep).tolist():
                    m_k = (hstep == k)
                    nk = int(m_k.sum())
                    d = per_h.setdefault(k, {'cd': 0.0, 'naive': 0.0, 'cv': 0.0, 'n': 0})
                    d['cd'] += chamfer_distance(pred[m_k], tgt[m_k]).item() * nk
                    d['naive'] += chamfer_distance(ctx[m_k, -1], tgt[m_k]).item() * nk
                    d['cv'] += chamfer_distance(
                        const_velocity_pred(ctx[m_k], hstep[m_k]), tgt[m_k]).item() * nk
                    d['n'] += nk
            n += len(sl)

    out = {k: v / n for k, v in acc.items()}
    out['n'] = n
    if per_horizon:
        out['per_horizon'] = {k: {kk: (vv / d['n'] if kk != 'n' else vv)
                                   for kk, vv in d.items()}
                               for k, d in sorted(per_h.items())}
    return out

def _compute_data_fingerprint():
    parts = [
        str(train_delta.shape), str(val_delta.shape),
        f'{float(train_delta.mean()):.8f}', f'{float(train_delta.std()):.8f}',
        f'{float(val_delta.mean()):.8f}', f'{float(val_delta.std()):.8f}',
    ]
    sample = train_delta.reshape(-1)[::997][:2000]
    parts.append(hashlib.sha256(sample.tobytes()).hexdigest())
    return hashlib.sha256('|'.join(parts).encode()).hexdigest()


# ── Is this the dataset the checkpoint was trained on? ──────────────
# The window-count check above catches two Module 4 runs disagreeing with each other.
# This catches something subtler and worse: arrays that are internally consistent but
# are not the ones these weights ever saw. Scoring a checkpoint against a different
# dataset produces a number for a pairing that never existed, and nothing about it
# looks wrong.
_fp_now  = _compute_data_fingerprint()
_fp_ckpt = ck.get('data_fingerprint')
if _fp_ckpt is None:
    print('  ⚠️  This checkpoint carries no data fingerprint, so whether it was trained '
          'on\n      the arrays now in module4_combined cannot be verified.')
elif _fp_now != _fp_ckpt:
    _msg = (f'\n  ✗ DATASET MISMATCH\n'
            f'      checkpoint was trained against : {_fp_ckpt[:16]}...\n'
            f'      module4_combined now holds     : {_fp_now[:16]}...\n'
            f'    Module 4 has been re-run since this checkpoint was saved, so these are\n'
            f'    not the arrays these weights were trained on. Any number produced here\n'
            f'    would describe a model/dataset pairing that never existed.\n'
            f'    Nothing has been modified -- this cell is read-only.')
    if REQUIRE_MATCHING_DATA:
        raise SystemExit(_msg)
    print(_msg + '\n    REQUIRE_MATCHING_DATA=False, continuing anyway.')
else:
    print(f'  Data check    : fingerprint matches the checkpoint ({_fp_now[:16]}...)')

betas = cosine_beta_schedule(T_DIFF).to(DEVICE)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_ac = torch.sqrt(alphas_cumprod)
sqrt_1m_ac = torch.sqrt(1.0 - alphas_cumprod)

# ── Model, from the checkpoint's EMA weights ─────────────────────────
# The EMA copy is what every reported Module 5 number was measured with; the raw weights
# are the noisier last-step snapshot.
# Module 5's ConditionalDenoiser takes no arguments: it reads WINDOW_SIZE, LATENT_DIM,
# HIDDEN_DIM, N_HEADS, N_TF_LAYERS, DROPOUT, MAX_PRED_STEP and CTX_SCALE from module
# scope. The preamble above set every one of them from the checkpoint, which is why the
# model it builds is the model the weights came from.
ema_model = ConditionalDenoiser().to(DEVICE)
_sd = ck.get('ema') or ck.get('model')
assert _sd is not None, "Checkpoint has neither an 'ema' nor a 'model' state_dict."
if ck.get('ema') is None:
    print('  ⚠️  No EMA weights in this checkpoint; scoring the raw weights instead.')
ema_model.load_state_dict(_sd)       # strict: a silent partial load would be worse
ema_model.eval()
_n_params = sum(p.numel() for p in ema_model.parameters())
print(f'  Model         : {_n_params:,} parameters loaded, eval mode')

# History, for QT1. Stored in the checkpoint, so it survives any disconnect.
train_losses = ck.get('train_losses') or []
val_cds      = ck.get('val_cds') or []
val_epochs   = ck.get('val_epochs') or []
best_cd      = float(ck.get('best_cd', float('inf')))

# Full set always: this cell exists to produce the reportable number.
_qt_idx = None
val_subset_idx = representative_subset_indices(M_va, min(M_va, 512))


print('\n' + '=' * 70)
print('  QUALITY TESTS')
print('=' * 70)
print(f'  Evaluation scope: FULL val set ({M_va:,} windows)')

if len(train_losses) >= 10:
    init_loss = np.mean(train_losses[:5])
    final_loss = np.mean(train_losses[-5:])
    reduction = (init_loss - final_loss) / max(init_loss, 1e-9) * 100
    print(f'\nQT1 — Loss convergence:')
    print(f'  Initial loss (first 5): {init_loss:.5f}')
    print(f'  Final loss  (last  5) : {final_loss:.5f}')
    print(f'  Reduction             : {reduction:.1f}%')
    qt1_pass = reduction > 30
    print(f'  {"✅ PASS" if qt1_pass else "⚠️  <30% — check lr schedule / train longer"}')
else:
    qt1_pass = False
    print('\nQT1 — Skipped (fewer than 10 epochs trained so far)')

if val_cds:
    # Full validation set, ensemble-mean prediction, plus every reference baseline in one
    # pass so the numbers are all measured on identical windows.
    _q = evaluate_cd(ema_model, indices=_qt_idx, steps=FINAL_STEPS, n_samples=FINAL_SAMPLES,
                      per_horizon=True, extra_baselines=True)
    final_cd       = _q['cd']
    final_cd_naive = _q['cd_naive']
    cd_single      = _q['cd_single']
    cd_cv          = _q['cd_cv']
    cd_mean1       = _q['cd_mean1']
    improvement    = (final_cd_naive - final_cd) / max(final_cd_naive, 1e-9) * 100
    print(f'\nQT2 — Chamfer vs baselines ({_q["n"]:,} windows, physical normalised-position space):')
    print(f'  Model CD, {FINAL_SAMPLES}-sample mean (EMA) : {final_cd:.6f}   ← headline')
    print(f'  Model CD, single trajectory      : {cd_single:.6f}   '
          f'({(cd_single - final_cd) / max(final_cd, 1e-9) * 100:+.1f}% vs the mean — the '
          f'price of scoring a random draw instead of the conditional mean)')
    print(f'  Model CD, 1-forward-pass mean    : {cd_mean1:.6f}   '
          f'(cheap-inference shortcut, {FINAL_STEPS * FINAL_SAMPLES}x fewer forward passes)')
    print(f'  Naive  (zero motion)             : {final_cd_naive:.6f}')
    print(f'  Constant velocity (last + v*k)   : {cd_cv:.6f}   ← the real bar (Module 4 QT7)')
    print(f'  Best subset CD during training   : {best_cd:.6f}')
    print(f'  Improvement vs naive             : {improvement:+.1f}%')
    print(f'  Improvement vs constant velocity : '
          f'{(cd_cv - final_cd) / max(cd_cv, 1e-9) * 100:+.1f}%')
    qt2_pass = final_cd < final_cd_naive
    print(f'  {"✅ PASS — beats naive baseline" if qt2_pass else "⚠️  Not beating naive — train more / check config"}')
    if qt2_pass and final_cd >= cd_cv:
        print(f'  ⚠️  Beats naive but not constant-velocity extrapolation — the diffusion '
              f'model is not yet earning its 15.8M parameters over a two-line estimator.')

    # Diagnostic subset for QT3/QT4. Strided across the WHOLE representative subset, not
    # its first BATCH_SIZE entries: val_subset_idx has stride 3, so its first 32 entries are
    # val windows 0..93 -- still entirely inside whichever video comes first in Module 4's
    # per-video concatenation. That is the same first-N bias the subset was introduced to
    # remove, and it is why the previous run's QT3/QT4 read ~0.0045-0.0053 against a real
    # full-set CD of ~0.019: those tests were quietly scoring one easy video.
    _dstride = max(1, len(val_subset_idx) // BATCH_SIZE)
    diag_idx = val_subset_idx[::_dstride][:BATCH_SIZE]
    diag_ctx = val_inp_t[diag_idx]
    diag_tgt = val_tgt_t[diag_idx]
    diag_dnorm = val_dnorm_t[diag_idx]
    diag_h = val_ps_t[diag_idx]
    diag_naive = chamfer_distance(diag_ctx[:, -1], diag_tgt).item()

    samples = []
    for s in range(5):
        gen = torch.Generator(device=DEVICE).manual_seed(1000 + s)
        x0_scaled = ddim_sample(ema_model, diag_ctx, diag_h, steps=FINAL_STEPS, eta=1.0,
                                 generator=gen, n_samples=1)
        delta_phys = to_physical(x0_scaled, diag_dnorm)
        samples.append(base_pred(diag_ctx, diag_h) + delta_phys)
    samples = torch.stack(samples, 0)                      # (5, n_diag, N_A, 3)
    sample_std = samples.std(dim=0).mean().item()
    cds_of_samples = [chamfer_distance(samples[s], diag_tgt).item() for s in range(5)]
    best_of_5 = min(cds_of_samples)
    mean_of_5 = float(np.mean(cds_of_samples))
    cd_of_mean = chamfer_distance(samples.mean(dim=0), diag_tgt).item()
    print(f'\nQT3 — Sample diversity (5 stochastic DDIM samples, eta=1.0, {len(diag_idx)} '
          f'windows spanning all videos):')
    print(f'  Mean std across samples: {sample_std:.6f}')
    print(f'  Per-sample CD (mean)   : {mean_of_5:.6f}')
    print(f'  Best-of-5 CD           : {best_of_5:.6f}')
    print(f'  CD of the 5-sample MEAN: {cd_of_mean:.6f}   '
          f'({(mean_of_5 - cd_of_mean) / max(mean_of_5, 1e-9) * 100:+.1f}% better than an '
          f'average single sample — this is the averaging effect QT2 exploits, measured '
          f'directly rather than assumed)')
    print(f'  Naive on same windows  : {diag_naive:.6f}')
    qt3_pass = sample_std > 1e-6
    print(f'  {"✅ PASS" if qt3_pass else "⚠️  Samples are not diverse — check stochastic sampling"}')

    # Both step counts on the SAME windows with the SAME sampling settings. The previous
    # version compared a 10-step number computed on the biased 32-window diag slice against
    # the 50-step number computed on the full val set, so its "10-step is 3.6x better"
    # reading was a dataset difference, not a step-count effect.
    cd_by_steps = {}
    # Batched, like evaluate_cd and QT6. This was the one sampling call in the module that
    # ran every diagnostic window at once: BATCH_SIZE windows x FINAL_SAMPLES trajectories
    # is 128 rows, and at W=32 that is 128 x N_ANCHOR x 32 token positions through the
    # context encoder -- a 12 GiB allocation that OOMed after QT1-QT3 had already passed,
    # losing QT4 to QT6 at the very end of an 11-hour run. EVAL_ROWS // FINAL_SAMPLES is
    # the same 64-row working set QT2 completes with.
    _rows4 = max(1, EVAL_ROWS // max(1, FINAL_SAMPLES))
    for _st in (5, 10, 20, 50):
        _parts = []
        for _b in range(0, len(diag_idx), _rows4):
            _sl = slice(_b, _b + _rows4)
            _parts.append(ddim_sample(ema_model, diag_ctx[_sl], diag_h[_sl], steps=_st,
                                      eta=0.0, n_samples=FINAL_SAMPLES))
        x0_st = torch.cat(_parts, dim=0)
        cd_by_steps[_st] = chamfer_distance(
            base_pred(diag_ctx, diag_h) + to_physical(x0_st, diag_dnorm), diag_tgt).item()
    print(f'\nQT4 — DDIM step count (EMA model, same {len(diag_idx)} windows, '
          f'{FINAL_SAMPLES}-sample mean, eta=0). Cheap here because the diagnostic set is '
          f'small; this is what justifies FINAL_STEPS={FINAL_STEPS} for the full-set QT2:')
    for _st, _v in cd_by_steps.items():
        print(f'  {_st:>2}-step CD : {_v:.6f}')
    print(f'  naive      : {diag_naive:.6f}')
    qt4_pass = min(cd_by_steps.values()) < diag_naive * 1.5
    print(f'  {"✅ PASS" if qt4_pass else "⚠️  High CD — check denoising"}')

    print(f'\nQT5 — Per-horizon breakdown ({_q["n"]:,} windows, {FINAL_SAMPLES}-sample mean):')
    print(f'  {"k":>3}  {"n":>6}  {"model":>9}  {"naive":>9}  {"const-vel":>9}  {"vs naive":>9}')
    qt5_pass = True
    for _k, _d in _q['per_horizon'].items():
        _imp = (_d['naive'] - _d['cd']) / max(_d['naive'], 1e-9) * 100
        qt5_pass = qt5_pass and _d['cd'] < _d['naive']
        print(f'  {_k:>3}  {_d["n"]:>6}  {_d["cd"]:>9.6f}  {_d["naive"]:>9.6f}  '
              f'{_d["cv"]:>9.6f}  {_imp:>8.1f}%')
    # Built outside the f-string deliberately: an apostrophe inside a single-quoted
    # f-string's REPLACEMENT FIELD only tokenizes on Python >= 3.12 (PEP 701). On an
    # older kernel it terminates the f-string early and the tokenizer then fails far
    # downstream with a misleading IndentationError.
    print('  ' + ('✅ PASS — beats naive at every horizon' if qt5_pass else
                  "⚠️  Loses to naive at some horizon — inspect that horizon's conditioning"))

    # QT6 — the diagnostic that says WHY QT2 reads the way it does.
    # Chamfer Distance alone cannot distinguish "the model learned nothing and emits noise"
    # from "the model learned the direction but overshoots the magnitude" -- both just look
    # like a number above the naive baseline. Comparing the predicted delta against the true
    # delta directly separates them, and comparing both against constant velocity shows how
    # much of the available signal is being captured:
    #   cos ~ 0                      -> no directional signal learned; more epochs of the
    #                                    same recipe will not help, the problem is upstream
    #                                    (conditioning, data signal-to-noise, or capacity).
    #   cos > 0 but ratio >> 1       -> right direction, overshooting -- a calibration issue,
    #                                    fixable at inference (fewer steps, more averaging).
    #   cos approaching const-vel's  -> the model is extracting the signal Module 4 showed
    #                                    exists, and CD should follow.
    _q6_idx = representative_subset_indices(M_va, 128)
    _cos_m, _cos_cv, _rat_m, _rat_cv, _n6 = 0.0, 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        _rows6 = max(1, EVAL_ROWS // max(1, FINAL_SAMPLES))
        for _bi in range(0, len(_q6_idx), _rows6):
            _sl = _q6_idx[_bi:_bi + _rows6]
            _ctx, _tgt = val_inp_t[_sl], val_tgt_t[_sl]
            _dn, _h = val_dnorm_t[_sl], val_ps_t[_sl]
            _true = _tgt - _ctx[:, -1]
            # Predicted DISPLACEMENT from the last observed frame, whatever the target mode:
            # under 'cv_residual' the network's output is only the correction, so the
            # constant-velocity part has to be added back before comparing with _true.
            _pred = (base_pred(_ctx, _h) - _ctx[:, -1]
                     + to_physical(ddim_sample(ema_model, _ctx, _h, steps=FINAL_STEPS,
                                               eta=0.0, n_samples=FINAL_SAMPLES), _dn))
            _cv = const_velocity_pred(_ctx, _h) - _ctx[:, -1]
            _tn = _true.norm(dim=-1).clamp_min(1e-12)
            for _v, _acc in ((_pred, 'm'), (_cv, 'cv')):
                _c = (F.cosine_similarity(_v, _true, dim=-1)).mean().item()
                _r = (_v.norm(dim=-1) / _tn).mean().item()
                if _acc == 'm':
                    _cos_m += _c * len(_sl); _rat_m += _r * len(_sl)
                else:
                    _cos_cv += _c * len(_sl); _rat_cv += _r * len(_sl)
            _n6 += len(_sl)
    _cos_m /= _n6; _cos_cv /= _n6; _rat_m /= _n6; _rat_cv /= _n6
    print(f'\nQT6 — Delta-space signal ({_n6} windows, predicted vs true displacement):')
    print(f'  {"":<18}{"cos sim":>10}{"|pred|/|true|":>16}')
    print(f'  {"model":<18}{_cos_m:>10.4f}{_rat_m:>16.3f}')
    print(f'  {"constant velocity":<18}{_cos_cv:>10.4f}{_rat_cv:>16.3f}')
    qt6_pass = _cos_m > 0.05
    if qt6_pass:
        print(f'  ✅ PASS — the predicted displacement is directionally correlated with the '
              f'true one ({_cos_m / max(_cos_cv, 1e-9) * 100:.0f}% of constant velocity\'s '
              f'correlation).')
    else:
        print(f'  ⚠️  Near-zero correlation — the model is emitting displacement that is '
              f'essentially unrelated to the true motion, which is why CD sits above naive. '
              f'More epochs of the same recipe will not fix this; look at conditioning, '
              f'data signal-to-noise, or capacity first.')

    all_pass = qt1_pass and qt2_pass and qt3_pass and qt4_pass and qt5_pass and qt6_pass
else:
    qt2_pass = qt3_pass = qt4_pass = qt5_pass = qt6_pass = False
    all_pass = False
    print('\nQT2-6 — Skipped (no validation pass has run yet, VAL_EVERY not reached)')

print(f'\n{"✅ ALL QUALITY TESTS PASSED" if all_pass else "⚠️  Review flagged tests above"}')

# ── Record, so the paper can read it without a rerun ────────────────
_out = {
    'checkpoint': os.path.basename(_path),
    'epoch': ck.get('epoch'),
    'which': WHICH,
    'n_val_windows': int(M_va),
    'scope': 'full validation set',
    'final_steps': FINAL_STEPS,
    'final_samples': FINAL_SAMPLES,
    'target_mode': TARGET_MODE,
    'window_size': WINDOW_SIZE,
    'data_scale': DATA_SCALE,
    'passes': {'qt1': bool(qt1_pass), 'qt2': bool(qt2_pass), 'qt3': bool(qt3_pass),
               'qt4': bool(qt4_pass), 'qt5': bool(qt5_pass), 'qt6': bool(qt6_pass)},
    'all_pass': bool(all_pass),
}
if val_cds:
    _out['cd'] = {'ours': float(final_cd), 'naive': float(final_cd_naive),
                  'const_velocity': float(cd_cv), 'single_sample': float(cd_single),
                  'improvement_vs_naive_pct': float(improvement),
                  'improvement_vs_cv_pct': float(
                      (cd_cv - final_cd) / max(cd_cv, 1e-9) * 100)}

_dest = f'{MODULE5_DIR}/module5_quality_tests.json'
with open(_dest, 'w') as _fh:
    json.dump(_out, _fh, indent=2)
print(f'\n  Written: {_dest}')
print('  (no checkpoint was modified: this cell contains no torch.save and no os.rename)')
