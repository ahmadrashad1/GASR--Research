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
# MAX_PRED_STEP: size of the horizon-conditioning embedding table (valid indices
# 1..MAX_PRED_STEP; row 0 is unused).
#
# Module 4 builds windows for PRED_STEPS=[1,2,3]: the SAME W=4 context window appears
# three times in the dataset with three DIFFERENT targets -- 1, 2 and 3 frames ahead --
# whose delta magnitudes scale with the horizon (Module 4's own QT2 measures
# |G_t+n - G_t| / mean_disp ~= the pred_step). Module 5 was never told which horizon it
# was being asked for, so p(delta | context) was a 3-component mixture over an
# UNOBSERVABLE variable and no amount of training could resolve it.
#
# The concrete consequence, and the reason this is worth fixing before another long run:
# Module 4's QT7 measures that after adaptive smoothing, plain constant-velocity
# extrapolation beats naive by 30-57% on this exact data. Constant velocity is
# `last + velocity * k`. Without k reaching the network, that estimator -- the one
# already demonstrated to work here -- is not representable by the model at all, whatever
# its capacity. Conditioning on the horizon is what makes the objective well-posed; it is
# a necessary condition for beating naive, not a guarantee of it. QT5 (per-horizon
# breakdown) and QT6 (delta-space correlation) are what actually confirm whether the model
# now uses the signal, and they are the first things to read after this run.
#
# Deliberately larger than max(PRED_STEPS) so widening Module 4's PRED_STEPS later does
# not change the architecture or invalidate checkpoints.
MAX_PRED_STEP   = 8

# TARGET_MODE — what the diffusion model is actually asked to produce.
#
#   'delta'       : the displacement itself,  G_{t+k} - G_t         (the previous behaviour)
#   'cv_residual' : the CORRECTION to constant-velocity extrapolation,
#                   G_{t+k} - (G_t + (G_t - G_{t-1}) * k)
#
# Why this changed. A full 500-epoch 'delta' run never beat the naive zero-motion baseline:
# best validation CD 0.01568 against naive 0.01557, reached at epoch ~40-60 and drifting
# upward after ~260 as the model overfitted. The diagnosis is not "no signal learned" --
# Module 6 measures cosine +0.40 between predicted and true displacement at two frames
# ahead -- it is that the direction is roughly right while the magnitude is not, and an
# error in magnitude is enough to lose to predicting nothing at all.
#
# 'cv_residual' fixes the floor rather than the ceiling. Under 'delta', a model that gives
# up and emits zero reproduces the NAIVE baseline. Under 'cv_residual', the same zero
# output reproduces CONSTANT VELOCITY, which Module 4's QT7 measures as 30-57% better than
# naive on this exact data after adaptive smoothing. So the worst case stops being "ties
# with naive" and becomes "ties with a baseline that already beats naive", and everything
# the model does learn is spent on the part linear extrapolation cannot express.
#
# This also makes the objective easier in the way that matters: the residual is smaller
# and much less correlated with the context than the raw displacement, so DATA_SCALE
# (recomputed below from whichever target is selected) no longer has to stretch a signal
# that is mostly predictable by a two-line formula.
#
# Set back to 'delta' to reproduce the earlier run. Listed in _arch_cfg, so switching
# modes retires the old checkpoint automatically instead of silently resuming a model
# trained against a different objective.
TARGET_MODE     = 'cv_residual'
assert TARGET_MODE in ('delta', 'cv_residual')

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
# TOTAL_EPOCHS: with EARLY_STOP=False (see below) this is a real TARGET that training runs
# all the way to, not a ceiling that early stopping is expected to cut short. That also
# makes it the fixed horizon the cosine LR schedule anneals over.
#
# 500, cut from 1000 because of Colab compute-unit limits. Changing this mid-run is NOT
# free -- it re-derives the cosine's progress against a new denominator -- but the two
# directions are not equally dangerous, and the resume path now distinguishes them:
#   SHORTENING (1000 -> 500), what happened here: progress (epoch-warmup)/(H-warmup) goes
#     UP, so the cosine steps LR DOWN at the resume epoch. That is an accelerated anneal.
#     It is a discontinuity, but a benign one -- it never re-heats converged weights -- so
#     the resume path allows it and prints the exact before/after LR.
#   LENGTHENING (350 -> 1000), the failure documented under LR_SCHEDULE: progress goes
#     DOWN, LR steps back UP mid-run, and an already-good model gets perturbed. Still
#     refused outright.
# Either way the checkpoint records the horizon it was actually annealed against, so the
# run stays self-describing after the change.
# 200, cut from 500. The 500-epoch run reached its best validation CD at epoch ~40-60 and
# then got steadily WORSE -- training loss still falling while validation error rose, i.e.
# 400 of those epochs were spent overfitting. 200 leaves generous margin past where the
# previous run peaked while letting the cosine schedule anneal fully, and CKPT_BEST still
# keeps whichever epoch actually validates best.
TOTAL_EPOCHS    = 200
EMA_DECAY       = 0.9999
GRAD_CLIP       = 1.0
# VAL_EVERY / VAL_SUBSET_SIZE: an older run validated only every 50 epochs on the full
# 1,461-window set (~4min/round at 50 DDIM steps) -- too coarse to see *where*
# generalisation peaked, and too expensive to run more often. Validating on a smaller but
# still representative subset (see representative_subset_indices below) is what makes a
# regular cadence affordable. With early stopping off these rounds no longer decide when
# training stops; they decide which checkpoint CKPT_BEST keeps, and they draw the
# validation curve that shows whether the run was still improving at TOTAL_EPOCHS.
VAL_EVERY        = 20              # was 10. Over the 500-epoch horizon that is 25
                                    # validation points -- still enough to resolve the shape
                                    # of the curve -- while halving the ~10% wall-clock
                                    # overhead that validation adds across a run this long.
VAL_SUBSET_SIZE  = 480             # ~1/3 of the full 1,461-window val set, evenly sampled
                                    # across every video's block -- not the biased first-N

# EARLY_STOP: disabled. The previous run stopped at epoch 460/1000 on a plateau that we
# now know was an artefact of un-conditioned horizons (see MAX_PRED_STEP): the model had
# converged to the best fit available for an ambiguous objective, so "no improvement for
# 80 epochs" was measuring the ceiling of a broken problem statement, not of training.
# With the horizon exposed to the model the objective is well-posed again, so this run is
# taken to the full TOTAL_EPOCHS. CKPT_BEST still tracks the best validation checkpoint
# throughout, so running the full horizon costs nothing in final quality even if the model
# does start to overfit part-way -- it only costs wall-clock.
EARLY_STOP       = False
PATIENCE         = 8               # only consulted when EARLY_STOP is True
# LR_SCHEDULE — 'cosine' or 'plateau'.
# A fixed-horizon cosine schedule is incompatible with early-stopping-driven training,
# where the true training length isn't known in advance. A real run proved this: resuming
# a converged checkpoint (LR fully annealed under TOTAL_EPOCHS=350) with a larger
# TOTAL_EPOCHS=1000 recomputed progress against the new horizon and jumped LR back up to
# ~3.2e-4 (64% of peak), perturbing an already-good model. That is why the previous
# revision switched to ReduceLROnPlateau.
# Now that EARLY_STOP is False the horizon IS known and fixed at TOTAL_EPOCHS, so cosine
# is the right schedule again -- and it is strictly better here than plateau, because
# plateau with LR_PATIENCE=4 rounds can collapse LR to min_lr within ~40 epochs of any
# flat stretch and then spend the remaining ~400 epochs learning nothing. The original
# discontinuity failure is guarded against directly: the checkpoint stores the horizon the
# schedule was computed against ('lr_horizon') and the resume path refuses to silently
# continue a cosine run under a LONGER TOTAL_EPOCHS (see TOTAL_EPOCHS for why a shorter
# one is allowed instead).
LR_SCHEDULE      = 'cosine'
MIN_LR           = 1e-6
LR_PATIENCE      = max(1, PATIENCE // 2)   # 'plateau' schedule only
LR_FACTOR        = 0.5                     # 'plateau' schedule only
WEIGHT_DECAY     = 0.05            # Raised from AdamW's 0.01 default: the 500-epoch run
                                    # overfitted measurably (validation CD rising from
                                    # ~epoch 260 while training loss kept falling), which is
                                    # exactly the condition the old comment here said to
                                    # raise this for. With early stopping off, regularisation
                                    # strength is the main defence against that, so it is a
                                    # visible, deliberate knob rather than a library default.
EMA_WARMUP_STEPS = 2000            # EMA_DECAY=0.9999 has a ~10k-step memory, so from a
                                    # random init the EMA weights stay dominated by noise
                                    # for the first ~50 epochs and every validation round
                                    # in that stretch scores garbage. Ramping the decay in
                                    # (see ema_decay_at) makes early rounds meaningful,
                                    # which matters more now that this run starts from
                                    # epoch 0 rather than resuming a warm checkpoint.

# ── Evaluation sampling ──────────────────────────────────────────────
# Chamfer Distance scores a SINGLE point prediction, and the point prediction that
# minimises it is the conditional MEAN E[delta | context, horizon] -- not a random draw
# from the conditional distribution, which is what one DDIM trajectory gives you. If the
# conditional has standard deviation s, a single sample carries error variance ~2s^2
# against the ground truth while the mean carries ~s^2: sampling is a factor of ~sqrt(2)
# worse than the model's own best achievable prediction, for free, on every metric.
# That penalty was being charged to the model in every QT2 so far. Averaging N_SAMPLES
# independent eta=0 trajectories is a Monte-Carlo estimate of that mean and removes it
# (error variance ~s^2(1 + 1/N)). QT2 reports the single-sample number too, so the cost
# of sampling stays visible rather than being quietly optimised away.
# Budget these against measured throughput rather than by taste. Sampling cost is
# windows x n_samples x steps forward passes, and on the observed ~100s/epoch hardware one
# training epoch is ~5,900 window-forward-backwards -- so ~18,000 window-evaluations costs
# roughly one epoch of wall-clock. That makes the old "full val set at 50 steps" evaluation
# (1,461 x 50 = 73k) about 4 epochs, and a naive 8-sample version of it (584k) about half
# an hour of pure evaluation -- per session, since a time-budgeted session runs the quality
# tests on every stop. 4 samples x 20 steps keeps the ensemble benefit (most of the
# variance reduction is in the first few samples: 1 - 1/N is 0.75 at N=4 and only 0.875 at
# N=8) while keeping the final evaluation near ~10 minutes. QT4 still sweeps step counts on
# a small diagnostic set, so the choice of 20 stays checkable rather than assumed.
VAL_STEPS        = 10              # DDIM steps for periodic in-training validation
VAL_SAMPLES      = 4               # trajectories averaged per window during training
FINAL_STEPS      = 20              # DDIM steps for the final QT2
FINAL_SAMPLES    = 4               # trajectories averaged per window for the final QT2
EVAL_ROWS        = 64              # eval rows resident at once = windows * n_samples.
                                    # Double the old eval batch of 32: with no backward
                                    # pass there are no stored activations, and this
                                    # architecture is kernel-launch-overhead-bound at small
                                    # batch (see MICRO_BATCH), so wider eval batches are
                                    # close to free in time. Drop back to 32 on an OOM.

CHECKPOINT_EVERY = 20              # clean multiple of VAL_EVERY
SESSION_BUDGET_HOURS = 11.0        # Colab Pro 12h session -- 1h reserved as buffer
FRESH_START      = False           # True forces epoch 0 even from a compatible checkpoint.
                                    # Not needed to start this run clean: adding horizon
                                    # conditioning changes _arch_cfg, so any pre-horizon
                                    # checkpoint is detected as incompatible, backed up and
                                    # restarted automatically.
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
assert os.path.exists(f'{COMBINED_DIR}/train_pred_step.npy'), (
    'train_pred_step.npy not found -- Module 5 now conditions on the prediction horizon '
    '(see MAX_PRED_STEP); re-run Module 4 to emit train/val_pred_step.npy.')

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
train_ps     = np.load(f'{COMBINED_DIR}/train_pred_step.npy').astype(np.int64)

val_inp      = np.load(f'{COMBINED_DIR}/val_inputs.npy').astype(np.float32)
val_tgt      = np.load(f'{COMBINED_DIR}/val_targets.npy').astype(np.float32)
val_delta    = np.load(f'{COMBINED_DIR}/val_deltas.npy').astype(np.float32)
val_dnorm    = np.load(f'{COMBINED_DIR}/val_disp_norm.npy').astype(np.float32)
val_ps       = np.load(f'{COMBINED_DIR}/val_pred_step.npy').astype(np.int64)
assert train_ps.max() <= MAX_PRED_STEP and val_ps.max() <= MAX_PRED_STEP, (
    f'Module 4 emitted pred_steps up to {max(train_ps.max(), val_ps.max())} but the horizon '
    f'embedding only has room for {MAX_PRED_STEP} -- raise MAX_PRED_STEP (this changes the '
    f'architecture and so restarts training).')

# Constant-velocity displacement for every window, in the SAME displacement-equalised space
# as Module 4's deltas: (last - previous) * k, scaled by that window's disp_norm factor.
# Module 4 already applied disp_norm to its deltas, so the two are directly comparable and
# no Module 4 re-run is needed to switch target modes.
def _cv_delta(inp, ps, dnorm):
    vel = inp[:, -1] - inp[:, -2]                      # per-window velocity, normalised space
    return vel * ps[:, None, None].astype(np.float32) * dnorm[:, None, None]

train_cv = _cv_delta(train_inp, train_ps, train_dnorm)
val_cv   = _cv_delta(val_inp,   val_ps,   val_dnorm)

if TARGET_MODE == 'cv_residual':
    train_target_x0 = train_delta - train_cv          # what constant velocity gets wrong
    val_target_x0   = val_delta   - val_cv
else:
    train_target_x0 = train_delta
    val_target_x0   = val_delta

# DATA_SCALE normalises whichever target was selected to ~unit variance. Recomputed rather
# than reused: the residual is a different (smaller) quantity than the raw delta, and
# feeding a differently-scaled x0 into the same schedule is exactly the "signal drowned in
# diffusion noise" failure this scaling exists to prevent.
DATA_SCALE = float(1.0 / (train_target_x0.std() + 1e-8))

train_x0 = train_target_x0 * DATA_SCALE   # near-unit-variance training target
val_x0   = val_target_x0   * DATA_SCALE

M_tr, W, N_A, _ = train_inp.shape
M_va = len(val_inp)
assert W == WINDOW_SIZE and N_A == N_ANCHOR, 'Module 4 dataset shape does not match config'

print('='*60)
print('  MODULE 5 — Conditional Diffusion Model')
print('='*60)
print(f'  Device        : {DEVICE}')
print(f'  Train windows : {M_tr:,}  |  Val windows: {M_va:,}')
print(f'  Target        : {TARGET_MODE}' + ('   (model predicts the CORRECTION to '
      'constant velocity; zero output = constant-velocity extrapolation)'
      if TARGET_MODE == 'cv_residual' else
      '   (model predicts the raw displacement; zero output = naive no-motion)'))
print(f'  DATA_SCALE    : {DATA_SCALE:.3f}  (target std {train_target_x0.std():.6f} -> ~1.0; '
      f'raw delta std {train_delta.std():.6f})')
PRED_STEPS_PRESENT = sorted(set(np.unique(train_ps).tolist()) | set(np.unique(val_ps).tolist()))
print(f'  Horizons      : {PRED_STEPS_PRESENT}  (train '
      + ', '.join(f'k={k}:{int((train_ps == k).sum()):,}' for k in PRED_STEPS_PRESENT)
      + ')  — conditioned on, not marginalised over')

train_inp_t   = torch.from_numpy(train_inp).to(DEVICE)   # whole dataset is ~180MB -- keep it
train_x0_t    = torch.from_numpy(train_x0).to(DEVICE)    # resident on GPU, avoid a transfer+sync
train_ps_t    = torch.from_numpy(train_ps).to(DEVICE)     # every micro-batch (was CPU-resident)
val_inp_t     = torch.from_numpy(val_inp).to(DEVICE)
val_tgt_t     = torch.from_numpy(val_tgt).to(DEVICE)
val_dnorm_t   = torch.from_numpy(val_dnorm).to(DEVICE)
val_ps_t      = torch.from_numpy(val_ps).to(DEVICE)


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
# MODEL / OPTIMIZER / EMA SETUP + RESUME
# ════════════════════════════════════════════════════════════════════
model = ConditionalDenoiser().to(DEVICE)
ema_model = copy.deepcopy(model).to(DEVICE)
for p in ema_model.parameters():
    p.requires_grad_(False)

n_params = sum(p.numel() for p in model.parameters())
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# Mixed precision: T4-class GPUs get most of their throughput from FP16 tensor
# cores, which plain FP32 eager training leaves on the table -- this is the
# main lever for fitting more epochs into each session's time budget. A no-op
# on CPU (autocast(enabled=False) short-circuits regardless of device_type).
USE_AMP = (DEVICE == 'cuda')
scaler = torch.amp.GradScaler(DEVICE, enabled=USE_AMP)


def lr_at(epoch, horizon=None):
    """The warmup+cosine LR at `epoch` for a given anneal `horizon`, as a pure function of
    its arguments -- no optimizer state read or written.

    Parameterised on the horizon (rather than always reading TOTAL_EPOCHS) so the resume
    path can price what changing TOTAL_EPOCHS actually does to the LR before deciding
    whether to allow it, instead of asserting on the horizon and leaving the magnitude of
    the change unstated."""
    H = TOTAL_EPOCHS if horizon is None else horizon
    if epoch < WARMUP_EPOCHS:
        return LR * (epoch + 1) / WARMUP_EPOCHS
    prog = (epoch - WARMUP_EPOCHS) / max(1, H - WARMUP_EPOCHS)
    prog = min(max(prog, 0.0), 1.0)
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * prog))


def set_lr_for_epoch(epoch):
    """Linear warmup for WARMUP_EPOCHS, then (LR_SCHEDULE == 'cosine') a cosine anneal from
    LR down to MIN_LR across the remaining epochs of the FIXED TOTAL_EPOCHS horizon. Under
    LR_SCHEDULE == 'plateau' this only drives warmup and the ReduceLROnPlateau scheduler
    takes over afterwards, exactly as in the previous revision."""
    if epoch < WARMUP_EPOCHS or LR_SCHEDULE == 'cosine':
        lr = lr_at(epoch)
    else:
        return optimizer.param_groups[0]['lr']   # plateau scheduler owns LR post-warmup
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    return lr


def ema_decay_at(step):
    """Ramped EMA decay. A flat 0.9999 averages over ~10k steps (~54 epochs at 184
    steps/epoch), so starting from a random init the EMA weights stay dominated by the
    initialisation for the whole early-training stretch -- every validation round in that
    window scores near-noise, and CKPT_BEST can latch onto a meaningless early minimum.
    Ramping in (equivalent to a running mean over all steps so far, capped at EMA_DECAY)
    makes early validation rounds reflect the actual model. Matters more now that this run
    starts from epoch 0 instead of resuming already-converged weights."""
    return min(EMA_DECAY, (1.0 + step) / (EMA_WARMUP_STEPS / 10.0 + step + 1e-8))


# Only constructed for LR_SCHEDULE == 'plateau'; under 'cosine' the horizon is fixed and
# known (EARLY_STOP is False), so set_lr_for_epoch owns the schedule outright.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=MIN_LR)

# Only architecture-relevant keys gate checkpoint compatibility -- a change in LR,
# epochs, or other pure-training hyperparams must NOT invalidate a valid checkpoint
# (the original notebook compared the whole config dict, which over-triggered resets).
_arch_cfg = {
    'LATENT_DIM': LATENT_DIM, 'HIDDEN_DIM': HIDDEN_DIM, 'N_HEADS': N_HEADS,
    'N_TF_LAYERS': N_TF_LAYERS, 'N_ANCHOR': N_ANCHOR, 'WINDOW_SIZE': WINDOW_SIZE,
    'T_DIFF': T_DIFF,
    # Horizon conditioning adds parameters and changes forward()'s signature, so any
    # checkpoint from before it is genuinely incompatible -- listing it here is what makes
    # the resume path back such a checkpoint up and start from epoch 0 automatically,
    # rather than raising a state_dict load error.
    'MAX_PRED_STEP': MAX_PRED_STEP,
    # Not an architecture key, but a checkpoint trained to predict the constant-velocity
    # RESIDUAL means something entirely different from one trained to predict the raw
    # displacement: loading one as the other silently reconstructs every prediction wrong.
    # Listing it here makes switching modes retire the old checkpoint automatically.
    'TARGET_MODE': TARGET_MODE,
}
_cfg = dict(_arch_cfg, DATA_SCALE=DATA_SCALE, LR=LR, TOTAL_EPOCHS=TOTAL_EPOCHS,
            BATCH_SIZE=BATCH_SIZE, MICRO_BATCH=MICRO_BATCH, MIN_SNR_GAMMA=MIN_SNR_GAMMA,
            LR_SCHEDULE=LR_SCHEDULE, WEIGHT_DECAY=WEIGHT_DECAY, EARLY_STOP=EARLY_STOP)

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
    """decay is now supplied per-step by ema_decay_at(global_step), not a constant."""
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.mul_(decay).add_(p.detach(), alpha=1 - decay)
    for ema_b, b in zip(ema_model.buffers(), model.buffers()):
        ema_b.copy_(b)


def make_ckpt(epoch, train_losses, val_cds, val_epochs, best_cd, epochs_without_improvement):
    return {
        'epoch': epoch, 'model': model.state_dict(), 'ema': ema_model.state_dict(),
        'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(), 'global_step': global_step,
        'train_losses': train_losses, 'val_cds': val_cds, 'val_epochs': val_epochs,
        'val_cds_naive': val_cds_naive,
        'best_cd': best_cd, 'epochs_without_improvement': epochs_without_improvement,
        'config': _cfg, 'arch_cfg': _arch_cfg, 'data_fingerprint': _data_fingerprint,
        # The horizon the LR schedule was computed against. Resuming a cosine run under a
        # different TOTAL_EPOCHS silently re-derives progress against the new horizon and
        # steps LR back UP mid-run -- the exact failure documented under LR_SCHEDULE. Stored
        # so the resume path can refuse instead of repeating it.
        'lr_horizon': TOTAL_EPOCHS if LR_SCHEDULE == 'cosine' else None,
        'lr_schedule': LR_SCHEDULE,
        'torch_rng_state': torch.get_rng_state(), 'numpy_rng_state': np.random.get_state(),
    }


start_epoch, best_cd = 0, float('inf')
train_losses, val_cds, val_epochs, val_cds_naive = [], [], [], []
global_step = 0                  # optimizer steps taken, drives the EMA decay ramp
epochs_without_improvement = 0   # early-stopping counter, persisted across resumes

if FRESH_START and os.path.exists(CKPT_LATEST):
    _backup = CKPT_LATEST.replace('.pth', f'_pre_fresh_start_{int(time.time())}.pth')
    os.rename(CKPT_LATEST, _backup)
    print(f'♻️  FRESH_START=True — existing checkpoint moved to {_backup}; training from epoch 0.')

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
            val_cds_naive = ck.get('val_cds_naive', [])
            global_step = ck.get('global_step', 0)
            epochs_without_improvement = ck.get('epochs_without_improvement', 0)
            # Guard the documented LR-discontinuity failure: a cosine schedule is only
            # continuous across a resume if it is annealing over the SAME horizon. But the
            # two ways of changing that horizon are not equally harmful, and treating them
            # alike costs a whole run's training for no reason, so the check is on the
            # DIRECTION of the change (and its measured size), not on mere inequality:
            #
            #   LENGTHENING (the documented 350 -> 1000 failure): progress
            #     (epoch-warmup)/(H-warmup) shrinks, so the cosine hands back a HIGHER LR
            #     than the weights were last trained at. That re-heats an already-annealed
            #     model. Still refused outright.
            #   SHORTENING (e.g. 1000 -> 500 on a tighter compute budget): progress grows,
            #     so LR steps DOWN. The schedule just finishes annealing sooner; converged
            #     weights are never perturbed upward. Allowed, but never silently -- the
            #     actual before/after LR is printed, because a large step down can still
            #     stall learning and that is a judgement the reader should get to make.
            #
            # Either way the next make_ckpt() records the horizon actually in force, so the
            # checkpoint stays an accurate description of how it was trained.
            _saved_horizon = ck.get('lr_horizon')
            if LR_SCHEDULE == 'cosine' and _saved_horizon not in (None, TOTAL_EPOCHS):
                _lr_was = lr_at(start_epoch, _saved_horizon)
                _lr_now = lr_at(start_epoch, TOTAL_EPOCHS)
                if TOTAL_EPOCHS > _saved_horizon:
                    _msg = (f'❌ This checkpoint was trained with a cosine LR schedule annealing '
                            f'over {_saved_horizon} epochs, but TOTAL_EPOCHS is now the LONGER '
                            f'{TOTAL_EPOCHS}. Resuming would recompute LR against the new horizon '
                            f'and step it back UP mid-run, from {_lr_was:.2e} to {_lr_now:.2e} at '
                            f'epoch {start_epoch} -- re-heating already-annealed weights. Either '
                            f'set TOTAL_EPOCHS={_saved_horizon} to continue this run, or set '
                            f'FRESH_START=True to start a new {TOTAL_EPOCHS}-epoch run from '
                            f'scratch.')
                    # Printed as well as raised: IPython swallows SystemExit's message, so a
                    # bare raise would stop the cell with no visible explanation.
                    print(_msg)
                    raise SystemExit(_msg)
                print(f'⚠️  LR horizon SHORTENED from {_saved_horizon} to {TOTAL_EPOCHS} epochs. '
                      f'Allowed: shortening moves the cosine further along, so LR steps DOWN '
                      f'(at epoch {start_epoch}: {_lr_was:.2e} -> {_lr_now:.2e}), which anneals '
                      f'faster rather than re-heating the weights. Model, EMA, optimizer and '
                      f'step count all resume normally; the remaining {max(0, TOTAL_EPOCHS - start_epoch)} '
                      f'epochs anneal to MIN_LR={MIN_LR:.0e} by epoch {TOTAL_EPOCHS}.')
            if 'torch_rng_state' in ck:
                torch.set_rng_state(ck['torch_rng_state'].cpu().to(torch.uint8))
            print(f'✅ Resumed from epoch {start_epoch}/{TOTAL_EPOCHS}  |  best CD: {best_cd:.6f}  '
                  f'|  global_step: {global_step:,}')
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
      f'lr={LR} ({LR_SCHEDULE})  |  wd={WEIGHT_DECAY}  |  session budget={SESSION_BUDGET_HOURS}h')
print(f'Early stopping : {"ENABLED (patience %d rounds)" % PATIENCE if EARLY_STOP else "DISABLED — running the full horizon"}')
print(f'Validation     : every {VAL_EVERY} epochs, {VAL_STEPS}-step DDIM x {VAL_SAMPLES} '
      f'averaged samples  |  final QT2: {FINAL_STEPS}-step x {FINAL_SAMPLES}')
if start_epoch < TOTAL_EPOCHS:
    print(f'Remaining      : {TOTAL_EPOCHS - start_epoch} epochs — if that exceeds the '
          f'{SESSION_BUDGET_HOURS}h budget, the cell checkpoints and exits; just re-run it '
          f'in a new session to continue (LR/EMA/step count all resume exactly).')


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


if start_epoch >= TOTAL_EPOCHS:
    print(f'Training already complete ({start_epoch} epochs). Skipping to evaluation.')
else:
    session_start = time.time()
    rng = np.random.default_rng(SEED + start_epoch)
    pbar = tqdm(range(start_epoch, TOTAL_EPOCHS), initial=start_epoch, total=TOTAL_EPOCHS)
    stop_reason = None   # 'epochs' | 'time_budget' | 'early_stop', set on whichever exit fires

    for epoch in pbar:
        # Called every epoch, not only during warmup: under LR_SCHEDULE='cosine' this owns
        # the whole schedule, and because it is a pure function of (epoch, TOTAL_EPOCHS) the
        # LR after a mid-run session resume is identical to what it would have been in one
        # continuous run.
        set_lr_for_epoch(epoch)

        perm = rng.permutation(M_tr)
        # Accumulated on-GPU: the previous version called .item() on every micro-batch loss,
        # forcing a host-device sync 368 times per epoch in a loop the CONFIG comments
        # already identify as kernel-launch-overhead-bound. One sync per epoch instead.
        epoch_loss_t = torch.zeros((), device=DEVICE)
        n_batches = 0

        for bi in range(0, M_tr - BATCH_SIZE + 1, BATCH_SIZE):
            batch_idx = perm[bi:bi + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            step_loss_sum = torch.zeros((), device=DEVICE)

            # Gradient accumulation: MICRO_BATCH rows per forward/backward instead of the
            # full BATCH_SIZE, so peak activation memory never exceeds what one micro-batch
            # needs. Summing ACCUM_STEPS backward() calls, each on a loss pre-scaled by
            # 1/ACCUM_STEPS, produces exactly the same gradient as one backward() on the
            # full batch=32 mean loss (mean-of-equal-size-group-means == overall mean).
            for mb in range(ACCUM_STEPS):
                sl = batch_idx[mb * MICRO_BATCH:(mb + 1) * MICRO_BATCH]
                ctx = train_inp_t[sl]              # already GPU-resident -- no per-step transfer
                x0 = train_x0_t[sl]
                hstep = train_ps_t[sl]             # which horizon this window's target is at

                ctx = ctx + torch.randn_like(ctx) * NOISE_AUG

                t = torch.randint(0, T_DIFF, (ctx.shape[0],), device=DEVICE)
                noise = torch.randn_like(x0)
                x_t = q_sample(x0, t, noise)
                v_target = v_target_fn(x0, noise, t)

                with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=USE_AMP):
                    v_pred = model(ctx, x_t, t, hstep)
                    w = min_snr_weight[t][:, None, None]
                    v_loss = (w * (v_pred - v_target) ** 2).mean()

                    x0_pred = pred_x0_from_v(x_t, v_pred, t)
                    aux_loss = F.mse_loss(x0_pred, x0)

                    micro_loss = v_loss + AUX_X0_WEIGHT * aux_loss

                scaler.scale(micro_loss / ACCUM_STEPS).backward()
                step_loss_sum += micro_loss.detach().float()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            update_ema(ema_decay_at(global_step))

            epoch_loss_t += step_loss_sum / ACCUM_STEPS
            n_batches += 1

        avg_loss = epoch_loss_t.item() / max(n_batches, 1)
        train_losses.append(avg_loss)

        # Periodic validation uses the representative SUBSET (cheap -- see VAL_SUBSET_SIZE),
        # not the full 1,461-window set, so a VAL_EVERY cadence is affordable. With
        # EARLY_STOP=False these rounds do not end training; they keep CKPT_BEST pointed at
        # the best-generalising weights, which is the safety net that makes running the full
        # horizon free -- a prior run's epoch-100 checkpoint beat its epoch-350 checkpoint on
        # the full val set at every DDIM step count, and this is what captures such a peak
        # even when training continues past it.
        did_validate = False
        if (epoch + 1) % VAL_EVERY == 0 or epoch == TOTAL_EPOCHS - 1:
            # VAL_SAMPLES averaged trajectories at VAL_STEPS DDIM steps. This tracks the same
            # quantity QT2 reports (an ensemble-mean prediction), so "best" checkpoints are
            # selected on the metric that is actually reported -- previously the tracked
            # signal was a single noisy draw, which made CKPT_BEST partly a lottery over
            # sampling noise. Fewer DDIM steps than the final evaluation keeps the round
            # affordable at VAL_SAMPLES>1: 10 steps x 4 samples is 40 window-evaluations
            # versus the old 50 x 1, i.e. cost is comparable while the metric is far less
            # noisy -- QT4 confirms low step counts lose little accuracy here.
            _v = evaluate_cd(ema_model, indices=val_subset_idx,
                              steps=VAL_STEPS, n_samples=VAL_SAMPLES)
            cd, cd_naive = _v['cd'], _v['cd_naive']
            val_cds.append(cd); val_cds_naive.append(cd_naive); val_epochs.append(epoch + 1)
            did_validate = True
            better = cd < best_cd
            if better:
                best_cd = cd
                epochs_without_improvement = 0
                torch.save(make_ckpt(epoch + 1, train_losses, val_cds, val_epochs,
                                      best_cd, epochs_without_improvement), CKPT_BEST)
            else:
                epochs_without_improvement += 1
            # Plateau scheduler only owns LR under LR_SCHEDULE='plateau', and only post-warmup
            # -- during warmup LR is deliberately ramping, and stepping the scheduler
            # concurrently would let it accumulate "no improvement" state (and potentially cut
            # LR) against that ramp.
            if LR_SCHEDULE == 'plateau' and epoch >= WARMUP_EPOCHS:
                scheduler.step(cd)

        cur_lr = optimizer.param_groups[0]['lr']
        elapsed_hr = (time.time() - session_start) / 3600
        will_stop_early = (EARLY_STOP and did_validate
                            and epochs_without_improvement >= PATIENCE)
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
            postfix['vs_naive'] = f'{(cd_naive - cd) / max(cd_naive, 1e-9) * 100:+.1f}%'
            if EARLY_STOP:
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

# The final QT2 evaluates FINAL_SAMPLES trajectories x FINAL_STEPS DDIM steps per window,
# which over all 1,461 val windows is ~15-20 minutes of GPU time. Worth it once, at the end
# of the run -- but this cell also reaches the quality tests every time a session stops on
# the time budget part-way through, and paying that on every intermediate session is pure
# waste. Intermediate sessions therefore score the representative subset instead (the same
# windows the training-time checks use) and say so.
_training_complete = (globals().get('stop_reason') in ('epochs', 'early_stop')
                       or start_epoch >= TOTAL_EPOCHS)
_qt_idx = None if _training_complete else val_subset_idx
_qt_scope = ('FULL val set' if _training_complete
              else f'{len(val_subset_idx)}-window representative subset — run is incomplete, '
                   f'the full-set number is computed once training reaches TOTAL_EPOCHS')
print(f'  Evaluation scope: {_qt_scope}')

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
    for _st in (5, 10, 20, 50):
        x0_st = ddim_sample(ema_model, diag_ctx, diag_h, steps=_st, eta=0.0,
                             n_samples=FINAL_SAMPLES)
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

# ════════════════════════════════════════════════════════════════════
# VISUALISATION
# ════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))

axes[0].plot(train_losses, linewidth=1, color='steelblue')
axes[0].set_title('Training Loss', fontweight='bold')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].grid(alpha=0.3)

if val_cds:
    axes[1].plot(val_epochs, val_cds, marker='o', markersize=3, color='crimson',
                 label=f'Model CD ({VAL_SAMPLES}-sample mean, subset)')
    # Per-round naive rather than one axhline from the final full-set number: the periodic
    # rounds score the subset, so a full-set naive line is not the baseline those points
    # were actually measured against.
    if val_cds_naive and len(val_cds_naive) == len(val_epochs):
        axes[1].plot(val_epochs, val_cds_naive, color='gray', linestyle='--', linewidth=1,
                     label='Naive CD (same windows)')
    else:
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

plt.suptitle(f'Module 5: Diffusion Training ({n_params:,} params, epoch {len(train_losses)}/{TOTAL_EPOCHS} '
             f'= {len(train_losses) / TOTAL_EPOCHS * 100:.0f}%)',
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
    print(f'  Final model CD  : {final_cd:.6f}   ({FINAL_SAMPLES}-sample mean, {_q["n"]:,} windows)')
    print(f'  Naive baseline  : {final_cd_naive:.6f}')
    print(f'  Const-vel bar   : {cd_cv:.6f}')
print(f'  Latest checkpoint: {CKPT_LATEST}')

_stop_reason = globals().get('stop_reason')   # unset only if this run found start_epoch >= TOTAL_EPOCHS
_done_epochs = globals().get('epoch', start_epoch - 1) + 1
if _stop_reason == 'early_stop':
    print(f'  → Stopped early: no improvement for {PATIENCE} validation rounds. '
          f'Use {CKPT_BEST} (best_cd={best_cd:.6f}). Set EARLY_STOP=False to run the full '
          f'{TOTAL_EPOCHS}-epoch horizon instead. See module5_eval_sweep.py for a step-count sweep.')
elif _stop_reason == 'time_budget':
    _left = TOTAL_EPOCHS - _done_epochs
    print(f'  → Session time budget reached at epoch {_done_epochs}/{TOTAL_EPOCHS} '
          f'({_done_epochs / TOTAL_EPOCHS * 100:.0f}% of the run). This is a PAUSE, not a stop: '
          f'{_left} epochs remain.')
    print(f'     Start a fresh Colab session and re-run this cell — LR (cosine over the fixed '
          f'{TOTAL_EPOCHS}-epoch horizon), EMA step count, optimizer and RNG state all resume '
          f'exactly where they left off, so the multi-session run is equivalent to one '
          f'continuous {TOTAL_EPOCHS}-epoch run.')
    print(f'     Do NOT change TOTAL_EPOCHS between sessions — the resume path will refuse, '
          f'because that would step the cosine LR back up mid-run.')
elif _stop_reason == 'epochs':
    print(f'  → Training complete — the full {TOTAL_EPOCHS}-epoch horizon was reached (100%). '
          f'Use {CKPT_BEST} for the best-generalising checkpoint, or {CKPT_LATEST} for the '
          f'final-epoch weights.')
else:
    print(f'  → Training was already complete from a previous run. Use {CKPT_BEST} '
          f'(best_cd={best_cd:.6f}) or proceed to Module 6.')
