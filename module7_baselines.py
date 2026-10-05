import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, os, math, json, time, copy
import matplotlib.pyplot as plt
from tqdm import tqdm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ════════════════════════════════════════════════════════════════════
# MODULE 7 — COMPARATIVE ANALYSIS AGAINST PUBLISHED BASELINES
#
# Every method below answers the identical question on the identical data:
#   given the positions of N_ANCHOR tracked tissue points over the last W frames,
#   where are those points k frames later?
#
# WHY THIS MODULE EXISTS
# Modules 5 and 6 compare the diffusion model against two trivial estimators
# (zero-motion and constant velocity). That is necessary but not sufficient for a
# conference submission: a reviewer will ask what published method the work improves
# on. This module implements a representative set of them under one protocol -- same
# windows, same temporal split, same metrics, same training budget, same optimiser --
# so that the resulting table is a like-for-like comparison rather than a collection
# of numbers copied from different papers' own datasets.
#
# THE BASE PAPER
# PointRNN (Fan & Yang, 2019, arXiv:1910.08287) is the closest published method: it
# predicts future point clouds from past ones and is evaluated with Chamfer Distance,
# exactly our task and metric. Its mechanism is a point-based spatiotemporally-local
# correlation,
#     S_t^i = pool_{j in kNN(P_t^i)} { W . [X_t^i, S_{t-1}^j, P_t^i - P_{t-1}^j] + b }
# which aggregates the previous step's hidden states from spatial neighbours together
# with the displacement between them. PointLSTM applies that operation in place of the
# matrix products of a standard LSTM. Both are implemented faithfully below.
#
# ONE DIFFERENCE FROM THE PUBLISHED SETTING, STATED UP FRONT
# PointRNN and MoNet operate on unordered LiDAR scans where point identity is NOT
# preserved between frames, which is why they must infer correspondence through kNN and
# why Chamfer Distance (a correspondence-free metric) is the natural score. Our Module 3
# tracks the SAME Gaussian through every frame, so correspondence is known. That makes
# per-point sequence models (GRU, LSTM, Transformer) directly applicable, and it makes a
# stricter metric available. We therefore report both:
#   CD  -- Chamfer Distance, correspondence-free, comparable to the published literature
#   MPE -- Mean Point Error, the mean L2 distance between a point and its OWN future
#          position. Strictly harder than CD, and the honest metric when identity is known
# A method can score well on CD by producing a cloud of roughly the right shape in roughly
# the right place while every individual point is wrong; MPE catches that, CD does not.
#
# FAIRNESS RULES APPLIED TO EVERY LEARNED METHOD
#   1. Identical data: Module 4's windows, Module 4's temporal 80/20 split per video.
#   2. Identical inputs: the W-frame context, expressed relative to the last observed
#      frame, plus the prediction horizon k.
#   3. Identical target: the displacement, in Module 4's displacement-equalised space.
#   4. Identical budget: BASELINE_EPOCHS, AdamW, cosine schedule, same batch size.
#   5. Best-validation checkpoint selected for each, so no method is reported at a
#      disadvantage because its loss happened to be rising at the final epoch.
# The diffusion model is NOT retrained here; it is loaded from its Module 5 checkpoint
# and evaluated on the same windows through the same metric code.
# ════════════════════════════════════════════════════════════════════

# ── Paths ────────────────────────────────────────────────────────────
COMBINED_DIR = f'{DRIVE_BASE}/module4_combined'
MODULE5_DIR  = f'{DRIVE_BASE}/module5'
MODULE7_DIR  = f'{DRIVE_BASE}/module7'
CKPT_BEST    = f'{MODULE5_DIR}/diffusion_ckpt_best.pth'
CKPT_LATEST  = f'{MODULE5_DIR}/diffusion_ckpt.pth'
os.makedirs(MODULE7_DIR, exist_ok=True)

# ── Training protocol, applied identically to every learned baseline ──
BASELINE_EPOCHS = 60     # these are 0.1-2M parameter models on 5,910 windows; they
                          # converge well inside this. The diffusion model needed
                          # hundreds of epochs, which is itself a reportable difference
                          # in training cost
BASELINE_LR     = 1e-3
BASELINE_WD     = 1e-2
BASELINE_BATCH  = 32     # windows per step (each carries N_ANCHOR point sequences)
VAL_EVERY       = 5
HIDDEN          = 256    # shared width for GRU / LSTM / MLP / PointLSTM, so that the
                          # comparison is between MECHANISMS and not between capacities
TF_LAYERS       = 3
TF_HEADS        = 8
POINTRNN_CHUNK  = 128    # points per chunk in the PointRNN correlation. The gate tensor
                          # is (batch, points, k, 4*HIDDEN); at batch 32 and all 512 points
                          # at once that is 537 MB in a single allocation, which exhausts a
                          # T4 once autograd keeps it. Chunking to 128 points costs nothing
                          # mathematically -- the correlation is independent per point.
POINTRNN_K      = 8      # neighbours in the PointRNN correlation. Fan & Yang use ball
                          # query or kNN; kNN is used here because our anchor density
                          # varies between videos and a fixed radius would return an
                          # inconsistent number of neighbours
SEED            = 7

# ── Diffusion-model evaluation settings (must match Module 6's QT1) ──
DDIM_STEPS   = 20
DDIM_SAMPLES = 4
EVAL_ROWS    = 64

torch.manual_seed(SEED)
np.random.seed(SEED)

assert os.path.exists(f'{COMBINED_DIR}/train_inputs.npy'), (
    f'Module 4 dataset not found in {COMBINED_DIR} -- run Module 4 first.')

print('=' * 70)
print('  MODULE 7 — Comparative Analysis Against Published Baselines')
print('=' * 70)

# ════════════════════════════════════════════════════════════════════
# DATA — Module 4's windows, untouched
# ════════════════════════════════════════════════════════════════════
def _load(split):
    return {
        'inp':   np.load(f'{COMBINED_DIR}/{split}_inputs.npy').astype(np.float32),
        'tgt':   np.load(f'{COMBINED_DIR}/{split}_targets.npy').astype(np.float32),
        'delta': np.load(f'{COMBINED_DIR}/{split}_deltas.npy').astype(np.float32),
        'dnorm': np.load(f'{COMBINED_DIR}/{split}_disp_norm.npy').astype(np.float32),
        'ps':    np.load(f'{COMBINED_DIR}/{split}_pred_step.npy').astype(np.int64),
        'vid':   np.load(f'{COMBINED_DIR}/{split}_video_id.npy').astype(np.int64),
    }

tr, va = _load('train'), _load('val')
M_tr, W, N_A, _ = tr['inp'].shape
M_va = len(va['inp'])
PRED_STEPS = sorted(set(np.unique(tr['ps']).tolist()))
meta4 = json.load(open(f'{COMBINED_DIR}/module4_metadata.json'))
video_order = meta4['video_order']

# Targets are scaled to ~unit variance for optimisation only; every reported metric is
# computed after inverting this, in physical normalised-position units.
BASE_SCALE = float(1.0 / (tr['delta'].std() + 1e-8))

T = {k: torch.from_numpy(v).to(DEVICE) for k, v in tr.items()}
V = {k: torch.from_numpy(v).to(DEVICE) for k, v in va.items()}

print(f'  Device        : {DEVICE}')
print(f'  Train / Val   : {M_tr:,} / {M_va:,} windows')
print(f'  Points        : {N_A}   context frames: {W}   horizons: {PRED_STEPS}')
print(f'  Videos        : {len(video_order)}')


# ════════════════════════════════════════════════════════════════════
# METRICS — identical code for every method
# ════════════════════════════════════════════════════════════════════
def chamfer_distance(pred, gt):
    """Symmetric Chamfer Distance, per window. Correspondence-free: this is the metric
    the point-cloud-prediction literature reports, and the one that lets our numbers sit
    beside PointRNN's and MoNet's without redefinition."""
    d = torch.cdist(pred, gt)
    return d.min(dim=2)[0].mean(dim=1) + d.min(dim=1)[0].mean(dim=1)


def mean_point_error(pred, gt):
    """Mean L2 distance between each point and its OWN future position, per window.
    Only computable because Module 3 preserves Gaussian identity across frames. Strictly
    harder than Chamfer Distance: a prediction that reproduces the right shape with every
    point in the wrong place scores well on CD and badly here."""
    return (pred - gt).norm(dim=-1).mean(dim=-1)


def to_physical(delta_equalised, dnorm):
    """Module 4 scaled each video's delta by disp_norm so no high-motion video dominated
    training. Undo it so all reported numbers are in one physical space."""
    return delta_equalised / dnorm[:, None, None]


# ════════════════════════════════════════════════════════════════════
# CLASSICAL BASELINES — closed form, no training
#
# These matter more than they look. Module 4's own measurements showed constant velocity
# beating the zero-motion baseline by 30-57% on this data, so any learned method that
# cannot clear them is not earning its parameters.
# ════════════════════════════════════════════════════════════════════
def b_zero_motion(ctx, k):
    """The tissue does not move. The weakest honest reference."""
    return torch.zeros_like(ctx[:, -1])


def b_constant_velocity(ctx, k):
    """last + (last - previous) * k. One finite difference."""
    return (ctx[:, -1] - ctx[:, -2]) * k[:, None, None].to(ctx.dtype)


def b_constant_acceleration(ctx, k):
    """Second-order extrapolation from the last three frames. Included because tissue
    under instrument contact accelerates, and because it tests whether the extra
    derivative helps or merely amplifies reconstruction noise."""
    # Second-order backward differences. A plain (x_t - x_{t-1}) is the velocity at the
    # MIDPOINT t-0.5; using it in a Taylor step leaves the prediction short by half an
    # acceleration. (3x_t - 4x_{t-1} + x_{t-2})/2 is the velocity AT t, which makes this
    # estimator exact on quadratic motion.
    v = (3 * ctx[:, -1] - 4 * ctx[:, -2] + ctx[:, -3]) * 0.5
    a = ctx[:, -1] - 2 * ctx[:, -2] + ctx[:, -3]
    kk = k[:, None, None].to(ctx.dtype)
    return v * kk + 0.5 * a * kk * kk


def b_linear_fit(ctx, k):
    """Least-squares straight line through all W frames, extrapolated k ahead. Unlike
    constant velocity this uses the whole window, so it is less sensitive to noise in any
    single frame -- the natural stronger version of the same idea."""
    Wn = ctx.shape[1]
    t = torch.arange(Wn, device=ctx.device, dtype=ctx.dtype)
    t_mean = t.mean()
    denom = ((t - t_mean) ** 2).sum()
    # slope of the per-point trajectory, fitted over the window
    slope = ((t - t_mean)[None, :, None, None] * (ctx - ctx.mean(dim=1, keepdim=True))
             ).sum(dim=1) / denom
    # value the fitted line takes at the last observed frame, so the prediction is a
    # displacement FROM that frame and is comparable with the other baselines
    fitted_last = ctx.mean(dim=1) + slope * (t[-1] - t_mean)
    kk = k[:, None, None].to(ctx.dtype)
    return (fitted_last + slope * kk) - ctx[:, -1]


@torch.no_grad()
def b_kalman(ctx, k):
    """Per-point two-state constant-velocity Kalman filter over the W observed frames,
    then propagated k steps ahead.

    The classical estimator for exactly this problem. Unlike the finite differences above
    it models measurement noise explicitly and weighs each observation accordingly, which
    matters here because Module 4 measured this trajectory as noise-dominated before
    smoothing. State is [position, velocity] per point per coordinate -- the coordinates
    are independent under a constant-velocity model, so they are filtered in parallel. The
    2x2 covariance is carried as four scalar fields rather than a matrix, which keeps the
    whole filter vectorised over (batch, points, 3)."""
    Wn = ctx.shape[1]
    q, r = 1e-5, 1e-3                       # process noise (on velocity), measurement noise
    x = ctx[:, 0]                           # filtered position
    v = torch.zeros_like(x)                 # filtered velocity
    p00 = torch.full_like(x, 1e-2)          # Cov(x,x)
    p01 = torch.zeros_like(x)               # Cov(x,v)
    p10 = torch.zeros_like(x)               # Cov(v,x)
    p11 = torch.full_like(x, 1e-2)          # Cov(v,v)
    for i in range(1, Wn):
        # predict: F = [[1,1],[0,1]], P <- F P F^T + Q, Q = diag(0, q)
        x = x + v
        p00 = p00 + p01 + p10 + p11
        p01 = p01 + p11
        p10 = p10 + p11
        p11 = p11 + q
        # update with the position measurement: H = [1, 0]
        S = p00 + r
        k0, k1 = p00 / S, p10 / S
        resid = ctx[:, i] - x
        x = x + k0 * resid
        v = v + k1 * resid
        # all four entries come from the PRE-update covariance, so take copies first
        n00, n01 = (1 - k0) * p00, (1 - k0) * p01
        n10, n11 = p10 - k1 * p00, p11 - k1 * p01
        p00, p01, p10, p11 = n00, n01, n10, n11
    # propagate the filtered state k steps, as a displacement from the last OBSERVED frame
    return (x + v * k[:, None, None].to(ctx.dtype)) - ctx[:, -1]


CLASSICAL = {
    'Zero motion (naive)':      b_zero_motion,
    'Constant velocity':        b_constant_velocity,
    'Constant acceleration':    b_constant_acceleration,
    'Linear fit over window':   b_linear_fit,
    'Kalman filter (CV model)': b_kalman,
}


# ════════════════════════════════════════════════════════════════════
# LEARNED BASELINES
#
# All take the context expressed RELATIVE to the last observed frame. That removes the
# absolute position of the tissue, which is irrelevant to how it is about to move, and it
# is what lets a model of this size learn anything from 5,910 windows. The horizon k is
# supplied to every method, because Modules 4 and 5 established that without it the task
# has three different correct answers for the same input.
# ════════════════════════════════════════════════════════════════════
def prep(ctx, k):
    """(B,W,N,3) absolute -> (B,N,W,3) relative to last frame, plus horizon (B,N,1)."""
    rel = (ctx - ctx[:, -1:]).permute(0, 2, 1, 3)              # (B, N, W, 3)
    kk = k.to(ctx.dtype)[:, None, None].expand(-1, ctx.shape[2], 1)
    return rel, kk


class MLPBaseline(nn.Module):
    """Flatten the window and regress the displacement. The simplest learned method that
    could work, and the control for whether any sequence modelling is needed at all."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(W * 3 + 1, HIDDEN), nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN), nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN), nn.GELU(),
            nn.Linear(HIDDEN, 3))

    def forward(self, ctx, k):
        rel, kk = prep(ctx, k)
        return self.net(torch.cat([rel.flatten(2), kk], dim=-1))


class RecurrentBaseline(nn.Module):
    """Per-point GRU or LSTM over the W-frame history. Applicable here only because point
    identity is preserved; on unordered LiDAR scans it would not be."""
    def __init__(self, cell='gru'):
        super().__init__()
        rnn = nn.GRU if cell == 'gru' else nn.LSTM
        self.rnn = rnn(3, HIDDEN, num_layers=2, batch_first=True)
        self.head = nn.Sequential(nn.Linear(HIDDEN + 1, HIDDEN), nn.GELU(),
                                  nn.Linear(HIDDEN, 3))

    def forward(self, ctx, k):
        rel, kk = prep(ctx, k)
        B, N, Wn, _ = rel.shape
        out, _ = self.rnn(rel.reshape(B * N, Wn, 3))
        h = out[:, -1].reshape(B, N, -1)
        return self.head(torch.cat([h, kk], dim=-1))


class TransformerBaseline(nn.Module):
    """A deterministic transformer over the same four-frame history, with the same depth
    and head count as the diffusion model's temporal encoder.

    This is the ablation that matters most for the paper: it isolates the generative
    formulation. If this matches or beats the diffusion model, then diffusion is not
    earning its cost here and the contribution is the representation, not the generator."""
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(3, HIDDEN)
        self.pe = nn.Parameter(torch.randn(W, HIDDEN) * 0.02)
        layer = nn.TransformerEncoderLayer(HIDDEN, TF_HEADS, HIDDEN * 2, dropout=0.1,
                                           activation='gelu', batch_first=True,
                                           norm_first=True)
        self.tf = nn.TransformerEncoder(layer, TF_LAYERS, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(HIDDEN + 1, HIDDEN), nn.GELU(),
                                  nn.Linear(HIDDEN, 3))

    def forward(self, ctx, k):
        rel, kk = prep(ctx, k)
        B, N, Wn, _ = rel.shape
        h = self.inp(rel.reshape(B * N, Wn, 3)) + self.pe[None]
        h = self.tf(h)[:, -1].reshape(B, N, -1)
        return self.head(torch.cat([h, kk], dim=-1))


class PointLSTM(nn.Module):
    """PointRNN / PointLSTM (Fan & Yang, 2019) — the base paper.

    The published mechanism replaces an LSTM's matrix products with a point-based
    spatiotemporally-local correlation:

        S_t^i = pool_{j in kNN(P_t^i)} { W . [X_t^i, S_{t-1}^j, P_t^i - P_{t-1}^j] + b }

    so a point's new state is pooled from the states of its spatial neighbours at the
    previous frame, together with the displacement to each of them. Unlike the recurrent
    baselines above, this does NOT assume point identity is preserved -- it re-derives the
    correspondence geometrically at every step, which is why it transfers to unordered
    point clouds and why it is the fair published comparison for this task.

    Implemented at a single resolution rather than the paper's three-level hierarchy: our
    clouds are 512 points, where the paper's downsampling levels (which target tens of
    thousands of LiDAR returns) would leave too few points per level to pool over. The
    correlation operator itself is unchanged."""
    def __init__(self, k_neigh=POINTRNN_K):
        super().__init__()
        self.k = k_neigh
        # one shared FC per gate, consuming [features, neighbour state, displacement]
        self.gates = nn.Linear(3 + HIDDEN + 3, 4 * HIDDEN)
        self.head = nn.Sequential(nn.Linear(HIDDEN + 1, HIDDEN), nn.GELU(),
                                  nn.Linear(HIDDEN, 3))

    def correlate(self, P_cur, P_prev, X_cur, S_prev):
        """The point-based spatiotemporally-local correlation, one frame.

        Chunked over the query points: each point's result depends only on its own
        neighbours, so this is exactly the unchunked computation with a bounded peak
        allocation."""
        kk = min(self.k, P_prev.shape[1])
        out = []
        for a in range(0, P_cur.shape[1], POINTRNN_CHUNK):
            q = P_cur[:, a:a + POINTRNN_CHUNK]                 # (B, n, 3)
            xq = X_cur[:, a:a + POINTRNN_CHUNK]
            n = q.shape[1]
            idx = torch.cdist(q, P_prev).topk(kk, dim=-1, largest=False).indices
            nb_state = torch.gather(S_prev[:, None].expand(-1, n, -1, -1), 2,
                                    idx[..., None].expand(-1, -1, -1, S_prev.shape[-1]))
            nb_pos = torch.gather(P_prev[:, None].expand(-1, n, -1, -1), 2,
                                  idx[..., None].expand(-1, -1, -1, 3))
            disp = q[:, :, None] - nb_pos                      # P_t^i - P_{t-1}^j
            feat = torch.cat([xq[:, :, None].expand(-1, -1, kk, -1), nb_state, disp], -1)
            out.append(self.gates(feat).max(dim=2).values)     # max-pool over neighbours
        return torch.cat(out, dim=1)

    def forward(self, ctx, k):
        B, Wn, N, _ = ctx.shape
        rel = ctx - ctx[:, -1:]                                # same relative frame as others
        S = torch.zeros(B, N, HIDDEN, device=ctx.device, dtype=ctx.dtype)
        C = torch.zeros_like(S)
        for t in range(Wn):
            P_prev = rel[:, max(t - 1, 0)]
            g = self.correlate(rel[:, t], P_prev, rel[:, t], S)
            i, f, o, u = g.chunk(4, dim=-1)
            C = torch.sigmoid(f) * C + torch.sigmoid(i) * torch.tanh(u)
            S = torch.sigmoid(o) * torch.tanh(C)
        kk = k.to(ctx.dtype)[:, None, None].expand(-1, N, 1)
        return self.head(torch.cat([S, kk], dim=-1))


LEARNED = {
    'MLP regressor':            lambda: MLPBaseline(),
    'GRU (per-point)':          lambda: RecurrentBaseline('gru'),
    'LSTM (per-point)':         lambda: RecurrentBaseline('lstm'),
    'Transformer (determ.)':    lambda: TransformerBaseline(),
    'PointLSTM [base paper]':   lambda: PointLSTM(),
}


# ════════════════════════════════════════════════════════════════════
# UNIFIED TRAINER — identical budget and schedule for every learned method
# ════════════════════════════════════════════════════════════════════
def train_baseline(build):
    model = build().to(DEVICE)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=BASELINE_LR, weight_decay=BASELINE_WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, BASELINE_EPOCHS, eta_min=1e-6)
    rng = np.random.default_rng(SEED)
    best_mpe, best_state, history = float('inf'), None, []
    t0 = time.time()

    for epoch in range(BASELINE_EPOCHS):
        model.train()
        perm = rng.permutation(M_tr)
        tot = 0.0
        for bi in range(0, M_tr - BASELINE_BATCH + 1, BASELINE_BATCH):
            sl = perm[bi:bi + BASELINE_BATCH]
            pred = model(T['inp'][sl], T['ps'][sl])
            loss = F.mse_loss(pred, T['delta'][sl] * BASE_SCALE)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        sched.step()
        if (epoch + 1) % VAL_EVERY == 0 or epoch == BASELINE_EPOCHS - 1:
            m = evaluate_learned(model)['overall']['mpe']
            history.append({'epoch': epoch + 1, 'val_mpe': m})
            if m < best_mpe:                     # best-validation selection, applied to
                best_mpe = m                     # every method so none is reported at a
                best_state = copy.deepcopy(model.state_dict())   # disadvantage
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {'parameters': n_par, 'train_seconds': time.time() - t0,
                   'best_val_mpe': best_mpe, 'history': history}


@torch.no_grad()
def evaluate_learned(model, batch=64):
    """Evaluate a learned baseline on the full validation set, per horizon and overall."""
    model.eval()
    return _accumulate(lambda ctx, k: model(ctx, k) / BASE_SCALE, batch)


@torch.no_grad()
def evaluate_closed_form(fn, batch=256):
    """Evaluate a classical baseline. These predict directly in physical
    normalised-position space, so they skip the equalised-space round trip the learned
    methods make. Both paths end in the same physical units, which is what the metrics
    consume, so no method gains or loses from the scaling."""
    return _accumulate(lambda ctx, k: fn(ctx, k), batch, closed_form=True)


@torch.no_grad()
def _accumulate(predict_delta, batch, closed_form=False):
    acc = {k: {'cd': 0.0, 'mpe': 0.0, 'n': 0} for k in PRED_STEPS}
    per_video = {}
    for bi in range(0, M_va, batch):
        sl = slice(bi, min(bi + batch, M_va))
        ctx, tgt = V['inp'][sl], V['tgt'][sl]
        k, dn, vid = V['ps'][sl], V['dnorm'][sl], V['vid'][sl]
        d_eq = predict_delta(ctx, k)
        pred = ctx[:, -1] + (d_eq if closed_form else to_physical(d_eq, dn))
        cd = chamfer_distance(pred, tgt)
        mpe = mean_point_error(pred, tgt)
        for kk in torch.unique(k).tolist():
            m = (k == kk)
            acc[kk]['cd'] += cd[m].sum().item()
            acc[kk]['mpe'] += mpe[m].sum().item()
            acc[kk]['n'] += int(m.sum())
        for vv in torch.unique(vid).tolist():
            m = (vid == vv)
            d = per_video.setdefault(int(vv), {'cd': 0.0, 'mpe': 0.0, 'n': 0})
            d['cd'] += cd[m].sum().item()
            d['mpe'] += mpe[m].sum().item()
            d['n'] += int(m.sum())
    out = {'per_horizon': {k: {'cd': v['cd'] / max(v['n'], 1),
                               'mpe': v['mpe'] / max(v['n'], 1), 'n': v['n']}
                           for k, v in acc.items()},
           'per_video': {video_order[v] if v < len(video_order) else str(v):
                         {'cd': d['cd'] / max(d['n'], 1), 'mpe': d['mpe'] / max(d['n'], 1),
                          'n': d['n']} for v, d in sorted(per_video.items())}}
    n_all = sum(v['n'] for v in acc.values())
    out['overall'] = {'cd': sum(v['cd'] for v in acc.values()) / max(n_all, 1),
                      'mpe': sum(v['mpe'] for v in acc.values()) / max(n_all, 1),
                      'n': n_all}
    return out


# ════════════════════════════════════════════════════════════════════
# OUR METHOD — loaded from the Module 5 checkpoint, never retrained here
# Architecture is re-declared because this module must run standalone; dimensions come
# from the checkpoint's own descriptor so it cannot disagree with what was trained.
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
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / half)
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return F.pad(emb, (0, 1)) if dim % 2 else emb


class ConditionalDenoiser(nn.Module):
    def __init__(self, window_size, latent_dim, hidden_dim, n_heads, n_tf_layers,
                 dropout, max_pred_step):
        super().__init__()
        D = latent_dim
        self.context_geo = GeomMLP(3, hidden_dim, D, dropout)
        self.target_geo = GeomMLP(3, hidden_dim, D, dropout)
        self.temporal_pe = nn.Parameter(torch.randn(window_size, D) * 0.02)
        layer = nn.TransformerEncoderLayer(D, n_heads, hidden_dim, dropout=dropout,
                                           activation='gelu', batch_first=True,
                                           norm_first=True)
        self.temporal_transformer = nn.TransformerEncoder(layer, n_tf_layers,
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

    def encode_context(self, ctx):
        B, Wn, N, _ = ctx.shape
        h = self.context_geo(ctx) + self.temporal_pe[None, :, None, :]
        h = h.permute(0, 2, 1, 3).reshape(B * N, Wn, -1)
        return self.temporal_transformer(h)[:, -1, :].reshape(B, N, -1)

    def forward(self, ctx, x_noisy, t, hstep):
        h_emb = self.horizon_mlp(self.horizon_emb(hstep))[:, None, :]
        c = self.encode_context(ctx) + h_emb
        g = self.target_geo(x_noisy)
        g = g + self.time_mlp(sinusoidal_embedding(t, g.shape[-1]))[:, None, :] + h_emb
        a, _ = self.cross_t2c(g, c, c)
        g2 = self.norm_t(g + a)
        b, _ = self.cross_c2t(c, g, g)
        c2 = self.norm_c(c + b)
        return self.output_mlp(torch.cat([g2, c2], dim=-1))


def cosine_beta_schedule(T_, s=0.008):
    x = torch.linspace(0, T_, T_ + 1)
    ac = torch.cos(((x / T_) + s) / (1 + s) * math.pi * 0.5) ** 2
    ac = ac / ac[0]
    return torch.clip(1 - (ac[1:] / ac[:-1]), 1e-4, 0.999)


def load_diffusion():
    path = CKPT_BEST if os.path.exists(CKPT_BEST) else CKPT_LATEST
    if not os.path.exists(path):
        return None
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    a = ck['arch_cfg']
    m = ConditionalDenoiser(a['WINDOW_SIZE'], a['LATENT_DIM'], a['HIDDEN_DIM'],
                            a['N_HEADS'], a['N_TF_LAYERS'], 0.1,
                            a.get('MAX_PRED_STEP', 8)).to(DEVICE)
    m.load_state_dict(ck['ema'])
    m.eval()
    acum = torch.cumprod(1.0 - cosine_beta_schedule(a['T_DIFF']).to(DEVICE), dim=0)
    return {'model': m, 'T': a['T_DIFF'], 'ac': acum,
            'sq': torch.sqrt(acum), 'sq1': torch.sqrt(1.0 - acum),
            'scale': ck['config']['DATA_SCALE'],
            'mode': a.get('TARGET_MODE', 'delta'),
            'epoch': ck.get('epoch', '?'),
            'params': sum(p.numel() for p in m.parameters()),
            'path': os.path.basename(path)}


@torch.no_grad()
def diffusion_delta(bundle, ctx, k, dnorm, steps=DDIM_STEPS, n_samples=DDIM_SAMPLES):
    """Predicted displacement in Module 4's equalised space, so it enters the same metric
    path as every other method. Mirrors Module 6's QT1 settings exactly."""
    B, N = ctx.shape[0], ctx.shape[2]
    c = ctx.repeat_interleave(n_samples, 0) if n_samples > 1 else ctx
    h = k.repeat_interleave(n_samples, 0) if n_samples > 1 else k
    rows = c.shape[0]
    ts = torch.linspace(bundle['T'] - 1, 0, steps, device=DEVICE).long()
    x = torch.randn(rows, N, 3, device=DEVICE)
    for i, t in enumerate(ts):
        tb = t.expand(rows)
        v = bundle['model'](c, x, tb, h)
        x0 = bundle['sq'][tb][:, None, None] * x - bundle['sq1'][tb][:, None, None] * v
        if i == len(ts) - 1:
            x = x0
            break
        eps = bundle['sq1'][tb][:, None, None] * x + bundle['sq'][tb][:, None, None] * v
        an = bundle['ac'][ts[i + 1]]
        x = torch.sqrt(an) * x0 + torch.sqrt(torch.clamp(1 - an, min=0.0)) * eps
    if n_samples > 1:
        x = x.view(B, n_samples, N, 3).mean(1)
    net = x / bundle['scale']
    if bundle['mode'] == 'cv_residual':
        # The network predicts only the correction to constant velocity. Add that part
        # back, converted into Module 4's equalised space so the result is the same
        # quantity every other method returns.
        net = net + b_constant_velocity(ctx, k) * dnorm[:, None, None]
    return net


# ════════════════════════════════════════════════════════════════════
# RUN EVERYTHING
# ════════════════════════════════════════════════════════════════════
results = {}

print('\n' + '-' * 70)
print('  Classical baselines (closed form, no training)')
print('-' * 70)
for name, fn in CLASSICAL.items():
    r = evaluate_closed_form(fn)
    results[name] = {'family': 'classical', 'parameters': 0, 'train_seconds': 0.0, **r}
    print(f'  {name:<26} CD {r["overall"]["cd"]:.6f}   MPE {r["overall"]["mpe"]:.6f}')

print('\n' + '-' * 70)
print(f'  Learned baselines ({BASELINE_EPOCHS} epochs each, identical protocol)')
print('-' * 70)
for name, build in LEARNED.items():
    model, info = train_baseline(build)
    r = evaluate_learned(model)
    results[name] = {'family': 'learned', **info, **r}
    print(f'  {name:<26} CD {r["overall"]["cd"]:.6f}   MPE {r["overall"]["mpe"]:.6f}   '
          f'({info["parameters"]:,} params, {info["train_seconds"]:.0f}s)')
    torch.save(model.state_dict(), f'{MODULE7_DIR}/baseline_{name.split()[0].lower()}.pth')

print('\n' + '-' * 70)
print('  Ours — conditional diffusion (loaded from Module 5, not retrained)')
print('-' * 70)
_bundle = load_diffusion()
if _bundle is None:
    print('  ⚠️  No Module 5 checkpoint found — our method is absent from the comparison.')
else:
    print(f'  checkpoint {_bundle["path"]} (epoch {_bundle["epoch"]}, '
          f'target mode "{_bundle["mode"]}")')
    # the diffusion sampler is far heavier than the baselines, so it gets the smaller
    # batch Module 6 uses rather than the baselines' evaluation batch
    rows = max(1, EVAL_ROWS // DDIM_SAMPLES)
    acc = {k: {'cd': 0.0, 'mpe': 0.0, 'n': 0} for k in PRED_STEPS}
    per_video = {}
    for bi in tqdm(range(0, M_va, rows), desc='  sampling'):
        sl = slice(bi, min(bi + rows, M_va))
        ctx, tgt = V['inp'][sl], V['tgt'][sl]
        k, dn, vid = V['ps'][sl], V['dnorm'][sl], V['vid'][sl]
        pred = ctx[:, -1] + to_physical(diffusion_delta(_bundle, ctx, k, dn), dn)
        cd, mpe = chamfer_distance(pred, tgt), mean_point_error(pred, tgt)
        for kk in torch.unique(k).tolist():
            m = (k == kk)
            acc[kk]['cd'] += cd[m].sum().item(); acc[kk]['mpe'] += mpe[m].sum().item()
            acc[kk]['n'] += int(m.sum())
        for vv in torch.unique(vid).tolist():
            m = (vid == vv)
            d = per_video.setdefault(int(vv), {'cd': 0.0, 'mpe': 0.0, 'n': 0})
            d['cd'] += cd[m].sum().item(); d['mpe'] += mpe[m].sum().item()
            d['n'] += int(m.sum())
    n_all = sum(v['n'] for v in acc.values())
    results['Ours — diffusion'] = {
        'family': 'ours', 'parameters': _bundle['params'], 'train_seconds': None,
        'checkpoint_epoch': _bundle['epoch'], 'target_mode': _bundle['mode'],
        'per_horizon': {k: {'cd': v['cd'] / max(v['n'], 1), 'mpe': v['mpe'] / max(v['n'], 1),
                            'n': v['n']} for k, v in acc.items()},
        'per_video': {video_order[v] if v < len(video_order) else str(v):
                      {'cd': d['cd'] / max(d['n'], 1), 'mpe': d['mpe'] / max(d['n'], 1),
                       'n': d['n']} for v, d in sorted(per_video.items())},
        'overall': {'cd': sum(v['cd'] for v in acc.values()) / max(n_all, 1),
                    'mpe': sum(v['mpe'] for v in acc.values()) / max(n_all, 1), 'n': n_all}}
    o = results['Ours — diffusion']['overall']
    print(f'  {"Ours — diffusion":<26} CD {o["cd"]:.6f}   MPE {o["mpe"]:.6f}   '
          f'({_bundle["params"]:,} params)')


# ════════════════════════════════════════════════════════════════════
# COMPARISON TABLE — the figure that goes in the paper
# ════════════════════════════════════════════════════════════════════
order = sorted(results, key=lambda n: results[n]['overall']['mpe'])
best = order[0]
naive_cd = results['Zero motion (naive)']['overall']['cd']
naive_mpe = results['Zero motion (naive)']['overall']['mpe']

print('\n' + '=' * 70)
print('  COMPARATIVE RESULTS — full validation set, identical protocol')
print('=' * 70)
hdr = f'  {"Method":<26} {"params":>9} {"CD":>10} {"MPE":>10} {"vs naive":>9}'
for k in PRED_STEPS:
    hdr += f' {"MPE k=" + str(k):>9}'
print(hdr)
print('  ' + '-' * (len(hdr) - 2))
for name in order:
    r = results[name]
    o = r['overall']
    par = f'{r["parameters"]:,}' if r['parameters'] else '—'
    row = (f'  {name:<26} {par:>9} {o["cd"]:>10.6f} {o["mpe"]:>10.6f} '
           f'{(naive_mpe - o["mpe"]) / naive_mpe * 100:>8.1f}%')
    for k in PRED_STEPS:
        row += f' {r["per_horizon"].get(k, {}).get("mpe", float("nan")):>9.6f}'
    print(row + ('   <-- best' if name == best else ''))
print('\n  CD  = Chamfer Distance (correspondence-free, comparable with the point-cloud'
      '\n        prediction literature). MPE = Mean Point Error (each point against its own'
      '\n        future position; stricter, available because Module 3 preserves identity).'
      '\n  "vs naive" is on MPE; positive means better than assuming no motion.')

if best != 'Ours — diffusion' and 'Ours — diffusion' in results:
    ours = results['Ours — diffusion']['overall']['mpe']
    gap = (ours - results[best]['overall']['mpe']) / results[best]['overall']['mpe'] * 100
    print(f'\n  ⚠️  The diffusion model is not the best method on this protocol: '
          f'"{best}" is {gap:.1f}% better on MPE.')
    print('      This is the comparison as it stands and is reported as such. The two '
          'corrective\n      measures implemented in Modules 5 and 6 (constant-velocity '
          'residual target and\n      magnitude calibration) are not reflected in this '
          'checkpoint; re-running after\n      retraining is what tests whether they close '
          'the gap.')

json.dump({'protocol': {'epochs': BASELINE_EPOCHS, 'lr': BASELINE_LR, 'batch': BASELINE_BATCH,
                        'hidden': HIDDEN, 'pointrnn_k': POINTRNN_K, 'seed': SEED,
                        'ddim_steps': DDIM_STEPS, 'ddim_samples': DDIM_SAMPLES,
                        'val_windows': M_va, 'train_windows': M_tr,
                        'horizons': PRED_STEPS},
           'results': results, 'ranking_by_mpe': order},
          open(f'{MODULE7_DIR}/module7_comparison.json', 'w'), indent=2)
print(f'\n  Results written : {MODULE7_DIR}/module7_comparison.json')


# ════════════════════════════════════════════════════════════════════
# FIGURES
# ════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

ax = axes[0]
names = [n for n in order]
vals = [results[n]['overall']['mpe'] for n in names]
cols = ['#1F6F6B' if results[n]['family'] == 'ours'
        else '#9A3B2E' if results[n]['family'] == 'learned' else '#777777' for n in names]
ax.barh(range(len(names)), vals, color=cols)
ax.axvline(naive_mpe, color='black', ls='--', lw=1, label='naive')
ax.set_yticks(range(len(names)))
ax.set_yticklabels([n.replace(' [base paper]', '*') for n in names], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel('Mean Point Error')
ax.set_title('Overall accuracy (lower is better)', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis='x')

ax = axes[1]
for n in names:
    ks = sorted(results[n]['per_horizon'])
    ax.plot(ks, [results[n]['per_horizon'][k]['mpe'] for k in ks], marker='o', lw=1.3,
            label=n.replace(' [base paper]', '*'))
ax.set_xlabel('horizon k (frames)')
ax.set_ylabel('Mean Point Error')
ax.set_xticks(PRED_STEPS)
ax.set_title('Accuracy vs prediction horizon', fontweight='bold')
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

ax = axes[2]
lrn = [n for n in names if results[n]['family'] in ('learned', 'ours')]
ax.scatter([results[n]['parameters'] for n in lrn], [results[n]['overall']['mpe'] for n in lrn],
           s=60, c=['#1F6F6B' if results[n]['family'] == 'ours' else '#9A3B2E' for n in lrn])
for n in lrn:
    ax.annotate(n.replace(' [base paper]', '*').split('(')[0].strip(),
                (results[n]['parameters'], results[n]['overall']['mpe']),
                fontsize=7, xytext=(4, 4), textcoords='offset points')
ax.axhline(naive_mpe, color='black', ls='--', lw=1)
ax.set_xscale('log')
ax.set_xlabel('parameters (log scale)')
ax.set_ylabel('Mean Point Error')
ax.set_title('Accuracy vs model size', fontweight='bold')
ax.grid(alpha=0.3)

plt.suptitle('Module 7: Comparative analysis against published baselines  '
             f'({M_va:,} held-out windows, * = base paper)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{MODULE7_DIR}/module7_comparison.png', dpi=120, bbox_inches='tight')
plt.show()
print(f'  Figure written  : {MODULE7_DIR}/module7_comparison.png')
print('=' * 70)
