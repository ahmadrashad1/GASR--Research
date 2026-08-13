import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, os, math, time, copy
import matplotlib.pyplot as plt
from tqdm import tqdm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════
N_ANCHOR        = 512
WINDOW_SIZE     = 4
LATENT_DIM      = 512
HIDDEN_DIM      = 1024
N_HEADS         = 8
N_TF_LAYERS     = 3               # was 6 -- measured on real T4 hardware (185.36s/epoch at
                                   # N_TF_LAYERS=6, MICRO_BATCH=8) that the 6-layer temporal
                                   # transformer accounts for ~70% of per-step op count (~96 of
                                   # ~136 forward ops), processing a 4-TOKEN sequence -- deep
                                   # enough for long NLP sequences, overkill for W=4 and the
                                   # dominant reason a single epoch was kernel-launch-overhead-
                                   # bound rather than FLOPs-bound. Halved to fit a full 3000-
                                   # epoch target style run inside one 12h Colab Pro session;
                                   # see TOTAL_EPOCHS below for the resulting realistic budget.
DROPOUT         = 0.1
T_DIFF          = 500
MIN_SNR_GAMMA   = 5.0
AUX_X0_WEIGHT   = 0.1
NOISE_AUG       = 0.0003

BATCH_SIZE      = 32             # effective batch size (training dynamics) -- unchanged
MICRO_BATCH     = 16              # actual per-forward-pass batch -- fits a T4's ~14.5GB
                                   # (encode_context folds N_ANCHOR into the batch dim for its
                                   # temporal transformer, so activation memory scales with
                                   # BATCH_SIZE * N_ANCHOR = 32*512 = 16,384 rows through a
                                   # 6-layer/1024-wide transformer -- BATCH_SIZE=32 alone OOMs
                                   # a T4 on the very first forward pass. Gradient accumulation
                                   # over ACCUM_STEPS micro-batches reproduces the exact same
                                   # effective-batch=32 gradient without ever materialising more
                                   # than MICRO_BATCH*N_ANCHOR rows of activations at once.
                                   # 16 rather than 8: the batch=32 OOM only exceeded the T4's
                                   # budget by ~128MiB out of 14.56GiB, so batch=16 needs roughly
                                   # half that activation memory (~7.2GB) -- a large safety
                                   # margin -- while halving the per-epoch iteration count versus
                                   # 8, which is the main lever on wall-clock time: this
                                   # architecture is kernel-launch-overhead-bound at small batch
                                   # sizes, so epoch time tracks iteration count more than FLOPs.
                                   # Drop to 8 (or lower) only if this still OOMs in practice.
ACCUM_STEPS     = max(1, BATCH_SIZE // MICRO_BATCH)
LR              = 5e-4
WARMUP_EPOCHS   = 35
# TOTAL_EPOCHS: a generous CEILING, not a target -- raised from 350 now that a real run
# on A100 finished 350 epochs in 3h13m (33.11s/epoch) out of an 11h budget, leaving huge
# headroom. But that same run revealed the real constraint isn't epoch budget, it's
# overfitting: the epoch-100 checkpoint beat the epoch-350 checkpoint on the FULL
# validation set at every DDIM step count (~0.0173 flat vs ~0.018-0.020 and degrading
# with more steps -- see PATIENCE below). So the ceiling is raised to give early stopping
# room to find a genuine optimum if one exists beyond epoch 100, while PATIENCE is what
# actually decides when training stops -- not this number.
TOTAL_EPOCHS    = 1000
EMA_DECAY       = 0.9999
GRAD_CLIP       = 1.0
# VAL_EVERY / VAL_SUBSET_SIZE / PATIENCE: the previous run only validated every 50 epochs
# on the full 1,461-window set (~4min/round at 50 DDIM steps) -- too coarse to see *where*
# generalisation peaked, and too expensive to run more often. Validating on a smaller but
# still representative subset (see representative_subset_indices below) makes frequent
# checks cheap, which is what patience-based early stopping needs to actually catch the
# epoch-100-ish peak instead of training 250 epochs past it like last time.
VAL_EVERY        = 10
VAL_SUBSET_SIZE  = 480             # ~1/3 of the full 1,461-window val set, evenly sampled
                                    # across every video's block -- not the biased first-N
PATIENCE         = 8               # stop after 8 validation rounds (80 epochs) with no
                                    # improvement in subset CD; last run went ~250 epochs
                                    # past its real peak with no such check in place
# LR_PATIENCE / LR_FACTOR: a fixed-horizon cosine schedule (the old lr_lambda) is
# fundamentally incompatible with early-stopping-driven training, where the true training
# length isn't known in advance -- that's the whole point of early stopping. A real run
# proved this concretely: resuming a converged checkpoint (LR fully annealed to 0 under
# TOTAL_EPOCHS=350) with a larger TOTAL_EPOCHS=1000 recomputed progress against the new
# horizon and jumped LR back up to ~3.2e-4 (64% of peak) -- perturbing an already-good
# model for 80 epochs rather than fine-tuning it, and confounding whether the following
# early-stop was measuring a genuine ceiling or just this discontinuity's damage. Fixed by
# replacing the horizon-dependent cosine decay with ReduceLROnPlateau, driven by the same
# subset-CD signal as early stopping: cut LR when stuck, only give up (early-stop) if
# still stuck after that. LR_PATIENCE is intentionally shorter than PATIENCE so a plateau
# tries "reduce LR" before "give up".
LR_PATIENCE      = max(1, PATIENCE // 2)
LR_FACTOR        = 0.5
CHECKPOINT_EVERY = 20              # clean multiple of VAL_EVERY
SESSION_BUDGET_HOURS = 11.0        # Colab Pro 12h session -- 1h reserved as buffer
SEED            = 42

COMBINED_DIR = f'{DRIVE_BASE}/module4_combined'
MODULE5_DIR  = f'{DRIVE_BASE}/module5'
os.makedirs(MODULE5_DIR, exist_ok=True)
CKPT_LATEST  = f'{MODULE5_DIR}/diffusion_ckpt.pth'
CKPT_BEST    = f'{MODULE5_DIR}/diffusion_ckpt_best.pth'

torch.manual_seed(SEED)
np.random.seed(SEED)

assert os.path.exists(COMBINED_DIR), f'Combined dataset not found at {COMBINED_DIR}'
assert os.path.exists(f'{COMBINED_DIR}/train_deltas.npy'), (
    'train_deltas.npy not found -- run the updated Module 4 (displacement-equalised) first.')

# ════════════════════════════════════════════════════════════════════
# DATA
# Context = absolute normalised positions (train_inputs.npy), unchanged.
# Target  = Module 4's cross-video-equalised delta (train_deltas.npy),
# further rescaled by DATA_SCALE = 1/std(train_deltas) so the diffusion
# process sees a near-unit-variance x0 -- this is the fix for the
# "signal drowned in diffusion noise" failure mode: Module 4 only fixed
# *cross-video* scale consistency, not the *overall* scale (~0.003-0.007),
# which is still far too small for v-prediction to condition on well.
# Both transforms (this one, and Module 4's per-window disp_norm) are
# inverted before every Chamfer Distance computation so evaluation always
# happens in true physical (normalised-position) space.
# ════════════════════════════════════════════════════════════════════
train_inp    = np.load(f'{COMBINED_DIR}/train_inputs.npy').astype(np.float32)
train_tgt    = np.load(f'{COMBINED_DIR}/train_targets.npy').astype(np.float32)
train_delta  = np.load(f'{COMBINED_DIR}/train_deltas.npy').astype(np.float32)
train_dnorm  = np.load(f'{COMBINED_DIR}/train_disp_norm.npy').astype(np.float32)

val_inp      = np.load(f'{COMBINED_DIR}/val_inputs.npy').astype(np.float32)
val_tgt      = np.load(f'{COMBINED_DIR}/val_targets.npy').astype(np.float32)
val_delta    = np.load(f'{COMBINED_DIR}/val_deltas.npy').astype(np.float32)
val_dnorm    = np.load(f'{COMBINED_DIR}/val_disp_norm.npy').astype(np.float32)

DATA_SCALE = float(1.0 / (train_delta.std() + 1e-8))

train_x0 = train_delta * DATA_SCALE       # near-unit-variance training target
val_x0   = val_delta   * DATA_SCALE

M_tr, W, N_A, _ = train_inp.shape
M_va = len(val_inp)
assert W == WINDOW_SIZE and N_A == N_ANCHOR, 'Module 4 dataset shape does not match config'

print('='*60)
print('  MODULE 5 — Conditional Diffusion Model')
print('='*60)
print(f'  Device        : {DEVICE}')
print(f'  Train windows : {M_tr:,}  |  Val windows: {M_va:,}')
print(f'  DATA_SCALE    : {DATA_SCALE:.3f}  (train_delta std {train_delta.std():.6f} -> ~1.0)')

train_inp_t   = torch.from_numpy(train_inp).to(DEVICE)   # whole dataset is ~180MB -- keep it
train_x0_t    = torch.from_numpy(train_x0).to(DEVICE)    # resident on GPU, avoid a transfer+sync
val_inp_t     = torch.from_numpy(val_inp).to(DEVICE)      # every micro-batch (was CPU-resident)
val_tgt_t     = torch.from_numpy(val_tgt).to(DEVICE)
val_dnorm_t   = torch.from_numpy(val_dnorm).to(DEVICE)


def representative_subset_indices(m, target_size):
    """Evenly-strided indices across the full array, not the first `target_size` rows.
    Module 4 concatenates val windows as contiguous per-video blocks, so val_inp[:N] for
    small N is dominated by whichever 1-2 videos happen to come first -- exactly the bias
    that made QT3/QT4's old 32-window diagnostic (and the periodic training-time check,
    before this fix) an unreliable stand-in for full-dataset quality. A stride smaller
    than the smallest per-video block guarantees every video contributes some windows."""
    stride = max(1, m // target_size)
    return np.arange(0, m, stride)


val_subset_idx = representative_subset_indices(M_va, VAL_SUBSET_SIZE)
print(f'  Val subset    : {len(val_subset_idx):,} windows (stride {max(1, M_va // VAL_SUBSET_SIZE)}, '
      f'representative across all videos) — used for periodic checks; full set used for final QT2.')


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
        self.context_geo = GeomMLP()
        self.target_geo  = GeomMLP()
        self.temporal_pe = nn.Parameter(torch.randn(WINDOW_SIZE, D) * 0.02)

        tf_layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=N_HEADS, dim_feedforward=HIDDEN_DIM,
            dropout=DROPOUT, activation='gelu', batch_first=True, norm_first=True)
        self.temporal_transformer = nn.TransformerEncoder(
            tf_layer, num_layers=N_TF_LAYERS, enable_nested_tensor=False)

        self.time_mlp = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))

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
        B, Wn, N_A, _ = ctx.shape
        h = self.context_geo(ctx)                              # (B, W, N_A, D)
        h = h + self.temporal_pe[None, :, None, :]
        h = h.permute(0, 2, 1, 3).reshape(B * N_A, Wn, -1)      # (B*N_A, W, D)
        h = self.temporal_transformer(h)                        # (B*N_A, W, D)
        # Last-token pooling (most recent context frame, index -1) rather than mean-pool:
        # for near-future prediction the most recent frame is the most predictive one, and
        # self-attention has already let it incorporate information from the older frames --
        # averaging back in the older frames' own tokens only dilutes that signal.
        h = h[:, -1, :]                                         # (B*N_A, D)
        return h.reshape(B, N_A, -1)                             # (B, N_A, D)

    def forward(self, ctx, x_noisy, t):
        ctx_latent = self.encode_context(ctx)                    # (B, N_A, D)
        tgt_latent = self.target_geo(x_noisy)                    # (B, N_A, D)
        tgt_latent = tgt_latent + self.time_mlp(sinusoidal_embedding(t, tgt_latent.shape[-1]))[:, None, :]

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

betas = cosine_beta_schedule(T_DIFF).to(DEVICE)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)                    # (T,)
sqrt_ac = torch.sqrt(alphas_cumprod)
sqrt_1m_ac = torch.sqrt(1.0 - alphas_cumprod)
snr = alphas_cumprod / (1.0 - alphas_cumprod)
min_snr_weight = torch.minimum(snr, torch.full_like(snr, MIN_SNR_GAMMA)) / (snr + 1.0)  # v-pred weighting


def q_sample(x0, t, noise):
    a = sqrt_ac[t][:, None, None]
    b = sqrt_1m_ac[t][:, None, None]
    return a * x0 + b * noise


def v_target_fn(x0, noise, t):
    a = sqrt_ac[t][:, None, None]
    b = sqrt_1m_ac[t][:, None, None]
    return a * noise - b * x0


def pred_x0_from_v(x_t, v_pred, t):
    a = sqrt_ac[t][:, None, None]
    b = sqrt_1m_ac[t][:, None, None]
    return a * x_t - b * v_pred


def pred_eps_from_v(x_t, v_pred, t):
    a = sqrt_ac[t][:, None, None]
    b = sqrt_1m_ac[t][:, None, None]
    return b * x_t + a * v_pred


@torch.no_grad()
def ddim_sample(model, ctx, steps=50, eta=0.0, generator=None):
    """Generalised DDIM (eta=0 deterministic, eta=1 ~ ancestral DDPM). Returns x0 in
    the model's *scaled* space (still needs /DATA_SCALE, /disp_norm to become physical)."""
    model.eval()
    B, N_A = ctx.shape[0], ctx.shape[2]
    ts = torch.linspace(T_DIFF - 1, 0, steps, device=DEVICE).long()
    x = torch.randn(B, N_A, 3, device=DEVICE, generator=generator)
    for i, t in enumerate(ts):
        t_batch = t.expand(B)
        v_pred = model(ctx, x, t_batch)
        x0_pred = pred_x0_from_v(x, v_pred, t_batch)
        eps_pred = pred_eps_from_v(x, v_pred, t_batch)
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
    model.train()
    return x


def chamfer_distance(pred, gt):
    """Symmetric Chamfer Distance. pred, gt: (B, N, 3) -> scalar (mean over batch)."""
    d = torch.cdist(pred, gt)
    d1 = d.min(dim=2)[0].mean(dim=1)
    d2 = d.min(dim=1)[0].mean(dim=1)
    return (d1 + d2).mean()


def to_physical(delta_scaled, dnorm):
    """Invert both scale transforms: model-space delta -> physical normalised-position delta."""
    return (delta_scaled / DATA_SCALE) / dnorm[:, None, None]


# ════════════════════════════════════════════════════════════════════
# MODEL / OPTIMIZER / EMA SETUP + RESUME
# ════════════════════════════════════════════════════════════════════
model = ConditionalDenoiser().to(DEVICE)
ema_model = copy.deepcopy(model).to(DEVICE)
for p in ema_model.parameters():
    p.requires_grad_(False)

n_params = sum(p.numel() for p in model.parameters())
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# Mixed precision: T4-class GPUs get most of their throughput from FP16 tensor
# cores, which plain FP32 eager training leaves on the table -- this is the
# main lever for fitting more epochs into each session's time budget. A no-op
# on CPU (autocast(enabled=False) short-circuits regardless of device_type).
USE_AMP = (DEVICE == 'cuda')
scaler = torch.amp.GradScaler(DEVICE, enabled=USE_AMP)


def set_warmup_lr(epoch):
    """Linear warmup, epoch-indexed only -- unlike the old cosine schedule this has no
    dependency on TOTAL_EPOCHS, so it can never jump discontinuously if the ceiling changes
    between sessions."""
    lr = LR * (epoch + 1) / WARMUP_EPOCHS
    for pg in optimizer.param_groups:
        pg['lr'] = lr


# Plateau-based, not horizon-based: steps only on validation rounds (via scheduler.step(cd)
# in the training loop), using the same subset-CD signal that drives early stopping.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=1e-6)

# Only architecture-relevant keys gate checkpoint compatibility -- a change in LR,
# epochs, or other pure-training hyperparams must NOT invalidate a valid checkpoint
# (the original notebook compared the whole config dict, which over-triggered resets).
_arch_cfg = {
    'LATENT_DIM': LATENT_DIM, 'HIDDEN_DIM': HIDDEN_DIM, 'N_HEADS': N_HEADS,
    'N_TF_LAYERS': N_TF_LAYERS, 'N_ANCHOR': N_ANCHOR, 'WINDOW_SIZE': WINDOW_SIZE,
    'T_DIFF': T_DIFF,
}
_cfg = dict(_arch_cfg, DATA_SCALE=DATA_SCALE, LR=LR, TOTAL_EPOCHS=TOTAL_EPOCHS,
            BATCH_SIZE=BATCH_SIZE, MICRO_BATCH=MICRO_BATCH, MIN_SNR_GAMMA=MIN_SNR_GAMMA)

# Separate from _arch_cfg: fingerprints Module 4's actual OUTPUT DATA (not just its config),
# so a re-run of Module 4 (e.g. the adaptive-smoothing fix, or adding/removing a video) is
# detected even though nothing about the model architecture changed. Without this, a resumed
# checkpoint silently keeps training against a NEW dataset while still carrying over
# best_cd / epochs_without_improvement measured on the OLD one -- concretely observed: a
# checkpoint resumed after Module 4 was regenerated needed only ~10 epochs to exhaust an
# early-stopping counter that was already nearly-tripped from the *previous* (pre-fix)
# dataset's plateau, stopping at 47% of TOTAL_EPOCHS with the model barely exposed to the
# corrected data at all. Cheap deterministic subsample + summary stats, not cryptographic --
# only needs to change when the underlying arrays actually change.
import hashlib
def _compute_data_fingerprint():
    parts = [
        str(train_delta.shape), str(val_delta.shape),
        f'{float(train_delta.mean()):.8f}', f'{float(train_delta.std()):.8f}',
        f'{float(val_delta.mean()):.8f}', f'{float(val_delta.std()):.8f}',
    ]
    sample = train_delta.reshape(-1)[::997][:2000]
    parts.append(hashlib.sha256(sample.tobytes()).hexdigest())
    return hashlib.sha256('|'.join(parts).encode()).hexdigest()

_data_fingerprint = _compute_data_fingerprint()


@torch.no_grad()
def update_ema(decay):
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.mul_(decay).add_(p.detach(), alpha=1 - decay)
    for ema_b, b in zip(ema_model.buffers(), model.buffers()):
        ema_b.copy_(b)


def make_ckpt(epoch, train_losses, val_cds, val_epochs, best_cd, epochs_without_improvement):
    return {
        'epoch': epoch, 'model': model.state_dict(), 'ema': ema_model.state_dict(),
        'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'train_losses': train_losses, 'val_cds': val_cds, 'val_epochs': val_epochs,
        'best_cd': best_cd, 'epochs_without_improvement': epochs_without_improvement,
        'config': _cfg, 'arch_cfg': _arch_cfg, 'data_fingerprint': _data_fingerprint,
        'torch_rng_state': torch.get_rng_state(), 'numpy_rng_state': np.random.get_state(),
    }


start_epoch, best_cd = 0, float('inf')
train_losses, val_cds, val_epochs = [], [], []
epochs_without_improvement = 0   # early-stopping counter, persisted across resumes

if os.path.exists(CKPT_LATEST):
    try:
        ck = torch.load(CKPT_LATEST, map_location=DEVICE, weights_only=False)
        arch_ok = ck.get('arch_cfg') == _arch_cfg
        data_ok = ck.get('data_fingerprint') == _data_fingerprint
        if arch_ok and not data_ok:
            backup = CKPT_LATEST.replace('.pth', f'_predates_data_change_{int(time.time())}.pth')
            os.rename(CKPT_LATEST, backup)
            print(f'⚠️  Module 4\'s dataset has changed since this checkpoint was saved (different '
                  f'data fingerprint -- e.g. Module 4 was re-run with a fix, or a video was '
                  f'added/removed) — backed up to {backup}, starting fresh. Resuming old weights '
                  f'against changed data would also resume best_cd/epochs_without_improvement '
                  f'measured on the OLD data, letting early stopping trigger almost immediately '
                  f'without the model actually learning the new data.')
        elif arch_ok and data_ok:
            model.load_state_dict(ck['model'])
            ema_model.load_state_dict(ck['ema'])
            optimizer.load_state_dict(ck['optimizer'])
            try:
                scheduler.load_state_dict(ck['scheduler'])
            except (KeyError, RuntimeError, ValueError):
                # A checkpoint saved under the old fixed-horizon cosine LambdaLR has a
                # state_dict shape ReduceLROnPlateau can't consume -- that's expected for
                # any checkpoint from before this fix, not a corruption. Losing the LR
                # scheduler's own internal best/patience bookkeeping is low-consequence (it
                # just restarts LR_PATIENCE fresh from here); model/ema/optimizer weights
                # and the early-stopping state below are unaffected, so keep going rather
                # than discarding real training progress over a scheduler-format mismatch.
                print('⚠️  Saved scheduler state is from a different LR schedule (expected '
                      'when resuming a checkpoint from before the plateau-LR fix) -- LR '
                      'scheduler restarting fresh; model weights and training progress '
                      'are unaffected.')
            if 'scaler' in ck:
                scaler.load_state_dict(ck['scaler'])
            start_epoch = ck['epoch']
            best_cd = ck['best_cd']
            train_losses = ck['train_losses']
            val_cds = ck['val_cds']
            val_epochs = ck['val_epochs']
            epochs_without_improvement = ck.get('epochs_without_improvement', 0)
            if 'torch_rng_state' in ck:
                torch.set_rng_state(ck['torch_rng_state'].cpu().to(torch.uint8))
            print(f'✅ Resumed from epoch {start_epoch}/{TOTAL_EPOCHS}  |  best CD: {best_cd:.6f}')
        else:
            backup = CKPT_LATEST.replace('.pth', f'_incompatible_{int(time.time())}.pth')
            os.rename(CKPT_LATEST, backup)
            print(f'⚠️  Checkpoint architecture mismatch — backed up to {backup}, starting fresh')
    except Exception as e:
        backup = CKPT_LATEST.replace('.pth', f'_corrupt_{int(time.time())}.pth')
        os.rename(CKPT_LATEST, backup)
        print(f'⚠️  Checkpoint failed to load ({e}) — backed up to {backup}, starting fresh')

print(f'Model: {n_params:,} parameters  |  EMA: enabled  |  V-prediction: enabled  |  '
      f'AMP: {"enabled (fp16)" if USE_AMP else "disabled (cpu)"}')
print(f'Training — target epoch {TOTAL_EPOCHS}  |  warmup={WARMUP_EPOCHS}  |  '
      f'batch={BATCH_SIZE} (micro={MICRO_BATCH} x {ACCUM_STEPS} accum)  |  '
      f'lr={LR}  |  session budget={SESSION_BUDGET_HOURS}h')


# ════════════════════════════════════════════════════════════════════
# TRAINING LOOP — resumable, time-budgeted
# ════════════════════════════════════════════════════════════════════
def evaluate_cd(m, indices=None, steps=50):
    """indices=None means the FULL validation set (used for the final QT2 number and the
    eval-sweep script). Periodic in-training checks pass val_subset_idx instead -- a
    representative but much cheaper stand-in, which is what makes frequent early-stopping
    checks affordable (see VAL_EVERY/PATIENCE in CONFIG)."""
    m.eval()
    idx = np.arange(M_va) if indices is None else indices
    total_cd, total_naive, n = 0.0, 0.0, 0
    with torch.no_grad():
        for bi in range(0, len(idx), BATCH_SIZE):
            sl = idx[bi:bi + BATCH_SIZE]
            if len(sl) == 0:
                continue
            ctx = val_inp_t[sl]
            tgt = val_tgt_t[sl]
            dnorm = val_dnorm_t[sl]
            x0_scaled = ddim_sample(m, ctx, steps=steps, eta=0.0)
            delta_phys = to_physical(x0_scaled, dnorm)
            pred_pos = ctx[:, -1] + delta_phys
            cd = chamfer_distance(pred_pos, tgt)
            cd_naive = chamfer_distance(ctx[:, -1], tgt)
            total_cd += cd.item() * len(sl)
            total_naive += cd_naive.item() * len(sl)
            n += len(sl)
    m.train()
    return total_cd / n, total_naive / n


if start_epoch >= TOTAL_EPOCHS:
    print(f'Training already complete ({start_epoch} epochs). Skipping to evaluation.')
else:
    session_start = time.time()
    rng = np.random.default_rng(SEED + start_epoch)
    pbar = tqdm(range(start_epoch, TOTAL_EPOCHS), initial=start_epoch, total=TOTAL_EPOCHS)
    stop_reason = None   # 'epochs' | 'time_budget' | 'early_stop', set on whichever exit fires

    for epoch in pbar:
        if epoch < WARMUP_EPOCHS:
            set_warmup_lr(epoch)

        perm = rng.permutation(M_tr)
        epoch_loss, n_batches = 0.0, 0

        for bi in range(0, M_tr - BATCH_SIZE + 1, BATCH_SIZE):
            batch_idx = perm[bi:bi + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            step_loss_sum = 0.0

            # Gradient accumulation: MICRO_BATCH rows per forward/backward instead of the
            # full BATCH_SIZE, so peak activation memory never exceeds what one micro-batch
            # needs. Summing ACCUM_STEPS backward() calls, each on a loss pre-scaled by
            # 1/ACCUM_STEPS, produces exactly the same gradient as one backward() on the
            # full batch=32 mean loss (mean-of-equal-size-group-means == overall mean).
            for mb in range(ACCUM_STEPS):
                sl = batch_idx[mb * MICRO_BATCH:(mb + 1) * MICRO_BATCH]
                ctx = train_inp_t[sl]              # already GPU-resident -- no per-step transfer
                x0 = train_x0_t[sl]

                ctx = ctx + torch.randn_like(ctx) * NOISE_AUG

                t = torch.randint(0, T_DIFF, (ctx.shape[0],), device=DEVICE)
                noise = torch.randn_like(x0)
                x_t = q_sample(x0, t, noise)
                v_target = v_target_fn(x0, noise, t)

                with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=USE_AMP):
                    v_pred = model(ctx, x_t, t)
                    w = min_snr_weight[t][:, None, None]
                    v_loss = (w * (v_pred - v_target) ** 2).mean()

                    x0_pred = pred_x0_from_v(x_t, v_pred, t)
                    aux_loss = F.mse_loss(x0_pred, x0)

                    micro_loss = v_loss + AUX_X0_WEIGHT * aux_loss

                scaler.scale(micro_loss / ACCUM_STEPS).backward()
                step_loss_sum += micro_loss.item()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            update_ema(EMA_DECAY)

            epoch_loss += step_loss_sum / ACCUM_STEPS
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_loss)

        # Periodic validation uses the representative SUBSET (cheap -- see VAL_SUBSET_SIZE),
        # not the full 1,461-window set, so checking every VAL_EVERY=10 epochs is affordable.
        # This is what makes early stopping actually able to catch the point where the model
        # starts overfitting (measured directly: a prior run's epoch-100 checkpoint beat its
        # epoch-350 checkpoint on the full val set at every DDIM step count -- training had
        # continued 250 epochs past the real peak with no mechanism to notice or stop).
        did_validate = False
        if (epoch + 1) % VAL_EVERY == 0 or epoch == TOTAL_EPOCHS - 1:
            cd, cd_naive = evaluate_cd(ema_model, indices=val_subset_idx)
            val_cds.append(cd); val_epochs.append(epoch + 1)
            did_validate = True
            better = cd < best_cd
            if better:
                best_cd = cd
                epochs_without_improvement = 0
                torch.save(make_ckpt(epoch + 1, train_losses, val_cds, val_epochs,
                                      best_cd, epochs_without_improvement), CKPT_BEST)
            else:
                epochs_without_improvement += 1
            # Only let the plateau scheduler act post-warmup -- during warmup, LR is driven
            # manually by set_warmup_lr, and stepping the plateau scheduler concurrently would
            # let it start accumulating "no improvement" state (and potentially cut LR) while
            # warmup is still deliberately ramping it up.
            if epoch >= WARMUP_EPOCHS:
                scheduler.step(cd)

        cur_lr = optimizer.param_groups[0]['lr']
        elapsed_hr = (time.time() - session_start) / 3600
        will_stop_early = did_validate and epochs_without_improvement >= PATIENCE
        should_checkpoint = ((epoch + 1) % CHECKPOINT_EVERY == 0
                              or elapsed_hr >= SESSION_BUDGET_HOURS
                              or will_stop_early
                              or epoch == TOTAL_EPOCHS - 1)
        if should_checkpoint:
            # CKPT_LATEST must reflect wherever training actually stops -- on ANY break path,
            # not just CHECKPOINT_EVERY boundaries -- or a resume would restart from a stale
            # epoch and re-discover the same early-stop/time-budget condition redundantly.
            torch.save(make_ckpt(epoch + 1, train_losses, val_cds, val_epochs,
                                  best_cd, epochs_without_improvement), CKPT_LATEST)

        postfix = {'loss': f'{avg_loss:.5f}', 'lr': f'{cur_lr:.2e}'}
        if did_validate:
            postfix['cd'] = f'{cd:.5f}'
            postfix['cd_naive'] = f'{cd_naive:.5f}'
            postfix['no_improve'] = f'{epochs_without_improvement}/{PATIENCE}'
        pbar.set_postfix(**postfix)

        if elapsed_hr >= SESSION_BUDGET_HOURS:
            print(f'\n⏱  Session time budget ({SESSION_BUDGET_HOURS}h) reached at epoch '
                  f'{epoch + 1}/{TOTAL_EPOCHS} — checkpoint saved to {CKPT_LATEST}.')
            print(f'   Re-run this cell to continue training from epoch {epoch + 1}.')
            stop_reason = 'time_budget'
            break

        if will_stop_early:
            print(f'\n🛑 Early stopping — no subset-CD improvement for {PATIENCE} validation '
                  f'rounds ({PATIENCE * VAL_EVERY} epochs) as of epoch {epoch + 1}. '
                  f'Best CD {best_cd:.6f} is in {CKPT_BEST}.')
            stop_reason = 'early_stop'
            break

        if epoch == TOTAL_EPOCHS - 1:
            stop_reason = 'epochs'

    if stop_reason == 'epochs':
        print(f'\n✅ Training complete — {TOTAL_EPOCHS} epochs reached.')


# ════════════════════════════════════════════════════════════════════
# QUALITY TESTS (run whenever training has produced at least one validation point)
# ════════════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('  MODULE 5 — Quality Tests')
print('='*60)

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
    final_cd, final_cd_naive = evaluate_cd(ema_model)
    improvement = (final_cd_naive - final_cd) / max(final_cd_naive, 1e-9) * 100
    print(f'\nQT2 — Chamfer vs naive baseline (physical normalised-position space):')
    print(f'  Model CD (EMA) : {final_cd:.6f}')
    print(f'  Naive CD       : {final_cd_naive:.6f}')
    print(f'  Best CD so far : {best_cd:.6f}')
    print(f'  Improvement    : {improvement:+.1f}%')
    qt2_pass = final_cd < final_cd_naive
    print(f'  {"✅ PASS — beats naive baseline" if qt2_pass else "⚠️  Not beating naive — train more / check config"}')

    # Representative subset, not val_inp_t[:n_diag] -- the first N windows are all from
    # whichever 1-2 videos happen to come first in Module 4's per-video concatenation, which
    # is exactly what made an earlier version of this diagnostic misleadingly optimistic
    # (CD=0.0055 on a 32-window slice vs the real full-set CD of ~0.017-0.020).
    diag_idx = val_subset_idx[:BATCH_SIZE] if len(val_subset_idx) >= BATCH_SIZE else val_subset_idx
    diag_ctx = val_inp_t[diag_idx]
    diag_tgt = val_tgt_t[diag_idx]
    diag_dnorm = val_dnorm_t[diag_idx]
    samples = []
    for s in range(5):
        gen = torch.Generator(device=DEVICE).manual_seed(1000 + s)
        x0_scaled = ddim_sample(ema_model, diag_ctx, steps=50, eta=1.0, generator=gen)
        delta_phys = to_physical(x0_scaled, diag_dnorm)
        samples.append(diag_ctx[:, -1] + delta_phys)
    samples = torch.stack(samples, 0)                      # (5, n_diag, N_A, 3)
    sample_std = samples.std(dim=0).mean().item()
    cds_of_samples = [chamfer_distance(samples[s], diag_tgt).item() for s in range(5)]
    best_of_5 = min(cds_of_samples)
    print(f'\nQT3 — Sample diversity (5 stochastic DDIM samples, eta=1.0):')
    print(f'  Mean std across samples: {sample_std:.6f}')
    print(f'  Best-of-5 Chamfer      : {best_of_5:.6f}')
    qt3_pass = sample_std > 1e-6
    print(f'  {"✅ PASS" if qt3_pass else "⚠️  Samples are not diverse — check stochastic sampling"}')

    gen = torch.Generator(device=DEVICE).manual_seed(2026)
    x0_scaled_10 = ddim_sample(ema_model, diag_ctx, steps=10, eta=0.0, generator=gen)
    delta_10 = to_physical(x0_scaled_10, diag_dnorm)
    cd_10 = chamfer_distance(diag_ctx[:, -1] + delta_10, diag_tgt).item()
    print(f'\nQT4 — Denoising trajectory (10-step vs 50-step DDIM, EMA model):')
    print(f'  10-step CD : {cd_10:.6f}')
    print(f'  50-step CD : {final_cd:.6f}')
    qt4_pass = final_cd < final_cd_naive * 1.5
    print(f'  {"✅ PASS" if qt4_pass else "⚠️  High CD — check denoising"}')

    all_pass = qt1_pass and qt2_pass and qt3_pass and qt4_pass
else:
    qt2_pass = qt3_pass = qt4_pass = False
    all_pass = False
    print('\nQT2-4 — Skipped (no validation pass has run yet, VAL_EVERY not reached)')

print(f'\n{"✅ ALL QUALITY TESTS PASSED" if all_pass else "⚠️  Review flagged tests above"}')

# ════════════════════════════════════════════════════════════════════
# VISUALISATION
# ════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))

axes[0].plot(train_losses, linewidth=1, color='steelblue')
axes[0].set_title('Training Loss', fontweight='bold')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].grid(alpha=0.3)

if val_cds:
    axes[1].plot(val_epochs, val_cds, marker='o', markersize=3, color='crimson', label='Model CD')
    axes[1].axhline(final_cd_naive, color='gray', linestyle='--', linewidth=1, label='Naive CD')
    axes[1].set_title('Validation Chamfer Distance', fontweight='bold')
    axes[1].set_xlabel('Epoch'); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    n_show = min(4, N_ANCHOR)
    ax = axes[2]
    ctx_last = diag_ctx[0, -1].detach().cpu().numpy()
    tgt_show = diag_tgt[0].detach().cpu().numpy()
    pred_show = samples[0, 0].detach().cpu().numpy()
    ax.scatter(ctx_last[:, 0], ctx_last[:, 1], s=4, c='lightgray', alpha=0.6, label='last context (naive)')
    ax.scatter(tgt_show[:, 0], tgt_show[:, 1], s=6, c='gold', marker='*', label='ground truth target')
    ax.scatter(pred_show[:, 0], pred_show[:, 1], s=4, c='crimson', alpha=0.6, label='model prediction')
    ax.set_title('Sample Prediction (val window 0, XY)', fontweight='bold')
    ax.legend(fontsize=7); ax.set_aspect('equal'); ax.grid(alpha=0.3)

plt.suptitle(f'Module 5: Diffusion Training ({n_params:,} params, epoch {len(train_losses)}/{TOTAL_EPOCHS})',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{MODULE5_DIR}/module5_qa.png', dpi=120, bbox_inches='tight')
plt.show()

print(f'\n{"="*60}')
print(f'  MODULE 5 STATUS')
print(f'{"="*60}')
print(f'  Parameters      : {n_params:,}')
print(f'  Epochs trained  : {len(train_losses)}/{TOTAL_EPOCHS}')
if val_cds:
    print(f'  Best val CD     : {best_cd:.6f}   (subset, checkpoint: {CKPT_BEST})')
    print(f'  Naive baseline  : {final_cd_naive:.6f}   (full val set)')
print(f'  Latest checkpoint: {CKPT_LATEST}')

_stop_reason = globals().get('stop_reason')   # unset only if this run found start_epoch >= TOTAL_EPOCHS
if _stop_reason == 'early_stop':
    print(f'  → Stopped early: no improvement for {PATIENCE} validation rounds. '
          f'Use {CKPT_BEST} (best_cd={best_cd:.6f}) -- further training from here is unlikely '
          f'to help without changing PATIENCE or addressing overfitting directly (e.g. more '
          f'dropout/weight decay). See module5_eval_sweep.py for a full-dataset check.')
elif _stop_reason == 'time_budget':
    print(f'  → Session time budget reached. Re-run this cell to continue '
          f'({TOTAL_EPOCHS - len(train_losses)} epochs remaining toward the ceiling).')
elif _stop_reason == 'epochs':
    print(f'  → Training complete ({TOTAL_EPOCHS} epoch ceiling reached without early stopping -- '
          f'consider whether PATIENCE is too high, or the model was still improving). '
          f'Use {CKPT_BEST} for the best-generalising checkpoint.')
else:
    print(f'  → Training was already complete from a previous run. Use {CKPT_BEST} '
          f'(best_cd={best_cd:.6f}) or proceed to Module 6.')
