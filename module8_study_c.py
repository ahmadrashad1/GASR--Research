# ════════════════════════════════════════════════════════════════════
# MODULE 8 — STUDY C ONLY: optimisation sensitivity
#
# GENERATED FILE -- do not edit. Produced by tools/make_study_c_cell.py from
# module8_hpo.py. Edit that, then re-run the generator.
#
# Studies A and B are finished (14.3 h between them) and their results live in
# module8/module8_hpo.json on Drive. This cell RELOADS them, adds Study C, and
# writes all three back -- so the output JSON and the figure still carry A and B
# without re-measuring either. Study D is not in this cell at all.
#
# NEEDS   module4_combined/ on Drive, and module8/module8_hpo.json if A and B are
#         to be preserved. It does NOT need the Module 5 checkpoint: only Study A
#         loads that.
# COSTS   16 backbones at W=4, 12 epochs each -- about 2.9 h on a T4. The cell
#         prints its own budget, from Study B's measured seconds, before training.
# ════════════════════════════════════════════════════════════════════
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, os, math, json, time, copy, itertools
import matplotlib.pyplot as plt
from tqdm import tqdm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# DRIVE_BASE and the Drive mount come from the setup cell at the top of the
# notebook. Recreate them here if this cell is the first thing run in a fresh
# runtime: a NameError sixty lines in costs a session for nothing.
if 'DRIVE_BASE' not in globals():
    DRIVE_BASE = '/content/drive/MyDrive/EndoGaussian'
    if not os.path.isdir(DRIVE_BASE):
        from google.colab import drive
        drive.mount('/content/drive')
    assert os.path.isdir(DRIVE_BASE), f'Drive mounted, but {DRIVE_BASE} is absent'
    print(f'DRIVE_BASE recovered locally -> {DRIVE_BASE}')

COMBINED_DIR = f'{DRIVE_BASE}/module4_combined'
MODULE5_DIR  = f'{DRIVE_BASE}/module5'
MODULE8_DIR  = f'{DRIVE_BASE}/module8'
CKPT_BEST    = f'{MODULE5_DIR}/diffusion_ckpt_best.pth'
CKPT_LATEST  = f'{MODULE5_DIR}/diffusion_ckpt.pth'
os.makedirs(MODULE8_DIR, exist_ok=True)

# Studies A and B completed on 19 November 2026 and their results are in
# module8_hpo.json on Drive, which this cell reloads at startup -- so turning them off
# here does NOT drop them from the output or the figure. Only re-enable one to re-measure
# it. Study A cost 8.3 h and Study B 6.0 h; neither needs repeating.
RUN_C = True              # this cell is Study C and nothing else


# ── Study A: inference cost/accuracy ─────────────────────────────────
A_STEPS   = [2, 5, 10, 20, 50]
A_SAMPLES = [1, 2, 4, 8]
A_SUBSET  = 480          # held-out windows per cell. The full set x 20 cells would be

# ── Study B/C: cheap backbone ────────────────────────────────────────
BACKBONE_EPOCHS = 40
BACKBONE_HIDDEN = 256
BACKBONE_BATCH  = 32     # reference batch AT W=4. Activation memory scales with
                          # batch x N_ANCHOR x window, so a fixed 32 would be 16x the
                          # W=4 footprint once the window is 32 and would OOM a T4
                          # before Study B produced a single number. batch_for_window()
                          # holds that product constant instead, which also keeps the
                          # narrow configs fast rather than crippling them to suit the
                          # widest one.
BACKBONE_LR     = 1e-3
# ── Study C: budget ──────────────────────────────────────────────────
# Study C trains 16 backbones. At the full 32-frame window that is 16 x the 10,242 s
# Study B measured there -- about 45 h, which is not a study, it is a weekend. It runs
# at a narrower window and a shorter schedule instead, because Study C measures the
# RELATIVE sensitivity to optimiser, learning rate, schedule, decay, batch, width and
# depth, and that ordering is what transfers -- not the absolute MPE of each cell.
C_WINDOW = 4             # Study B's measured cost per 40-epoch run: W=4 2156s, W=8
                          # 3389s, W=16 5780s, W=32 10242s. At W=8 the 16 runs come to
                          # 4.5 h; at W=4, 2.9 h. W=4 also sits closer to W=8 (the best
                          # window) in MPE than W=32 does, so nothing is gained by
                          # probing the widest.
C_EPOCHS = 12            # against BACKBONE_EPOCHS=40. Cutting epochs is the RISKIER of

B_WINDOWS = [4, 8, 16, 32]   # truncations of the 32-frame context. This is the cheap

# ── Study D: diffusion ablations ─────────────────────────────────────
D_EPOCHS        = 25
D_TRAIN_FRAC    = 0.40
D_BATCH         = 16     # effective batch, preserved by accumulation over D_MICRO
D_MICRO         = 2      # rows per forward/backward. The denoiser sees
                          # batch x N_ANCHOR x W token positions, the same quantity that
                          # forced Module 5's MICRO_BATCH down to 2 at W=32. Without the
                          # split, Study D OOMs on its first config; without preserving
                          # the effective batch, it would screen at batch 2 and rank
                          # configs on gradient noise rather than on the configs.
D_LR            = 5e-4
D_PROMOTE_TOP   = 2      # best configs are re-run at full budget, a two-rung successive
D_PROMOTE_EPOCH = 75     # halving: cheap screening, then a fair comparison of finalists

SEED = 11
torch.manual_seed(SEED); np.random.seed(SEED)
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

def to_physical(delta_eq, dnorm):
    return delta_eq / dnorm[:, None, None]

# ════════════════════════════════════════════════════════════════════
# DETERMINISTIC BACKBONE — the vehicle for studies B and C
# Same shape as the diffusion model's temporal encoder, minus the generative part, so
# conclusions about optimisation transfer while each run costs about a minute.
# ════════════════════════════════════════════════════════════════════
def batch_for_window(window):
    """Hold batch x window constant at the W=4 reference, so every Study B config runs
    at the same activation footprint instead of the widest one deciding for all."""
    return max(2, BACKBONE_BATCH * 4 // max(1, window))

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

# Studies are expensive and they crash: Study C hit an OOM on 19 November after Studies
# A and B had already cost 14.3 h, and because the only json.dump sat at the very end of
# the cell, all of it was lost. Results are now reloaded from the last run and written
# back after EVERY study, so a later failure costs one study instead of all of them.
RESULTS_PATH = f'{MODULE8_DIR}/module8_hpo.json'
results = {}
if os.path.exists(RESULTS_PATH):
    try:
        results = json.load(open(RESULTS_PATH)).get('results', {}) or {}
        # JSON stringifies B_context's integer window keys. Restore them, or sorted()
        # orders them lexically ('16','32','4','8') and the Study B panel plots nonsense.
        if isinstance(results.get('B_context'), dict):
            results['B_context'] = {int(k): v for k, v in results['B_context'].items()}
        print(f'  Resuming        : {len(results)} study block(s) loaded from '
              f'{RESULTS_PATH}')
        for _k in sorted(results):
            print(f'                    · {_k}')
    except Exception as _e:
        print(f'  ⚠️  {RESULTS_PATH} unreadable ({_e}); starting with none. It will be '
              f'overwritten.')

def save_results(stage):
    """Write everything gathered so far. Called after each study, so an OOM or a
    disconnect in a later study cannot discard an earlier one."""
    json.dump({'config': {'backbone_epochs': BACKBONE_EPOCHS, 'seed': SEED,
                          'a_subset': A_SUBSET, 'a_steps': A_STEPS,
                          'a_samples': A_SAMPLES, 'b_windows': B_WINDOWS,
                          'c_window': C_WINDOW, 'c_epochs': C_EPOCHS,
                          'd_epochs': D_EPOCHS, 'd_micro': D_MICRO,
                          'd_fraction': D_TRAIN_FRAC, 'val_windows': M_va},
               'results': results},
              open(RESULTS_PATH, 'w'), indent=2)
    print(f'    ✅ saved after {stage} -> {RESULTS_PATH}')

# ════════════════════════════════════════════════════════════════════
# STUDY C — optimisation sensitivity, one factor at a time
# ════════════════════════════════════════════════════════════════════
if RUN_C:
    print('\n  STUDY C — optimisation sensitivity (one factor at a time)')

    # The reference MUST size its batch for the window it actually runs at. The original
    # passed batch=BACKBONE_BATCH (32, the W=4 reference) while defaulting window=W_FULL
    # (32), i.e. 8x the activation footprint batch_for_window() was written to hold
    # constant -- and it OOMed on the very first training, before a single number.
    C_BATCH = batch_for_window(C_WINDOW)
    ref = dict(optim='adamw', lr=BACKBONE_LR, schedule='cosine', wd=1e-2,
               batch=C_BATCH, hidden=BACKBONE_HIDDEN, layers=3,
               window=C_WINDOW, epochs=C_EPOCHS)

    # A batch is only admissible if batch x points x window stays inside the footprint
    # Study B already proved safe, doubled -- the backbone is far lighter per token than
    # the diffusion model, but 'vary the batch' must not mean 'OOM at the top value'.
    C_TOKEN_CAP = 2 * BACKBONE_BATCH * 4 * N_A

    def c_batch_values(ref_batch):
        cand = [max(2, ref_batch // 2), ref_batch * 2]
        keep, drop = [], []
        for b in dict.fromkeys(cand):
            if b == ref_batch:
                continue
            (keep if b * N_A * C_WINDOW <= C_TOKEN_CAP else drop).append(b)
        for b in drop:
            print(f'    batch={b} skipped: {b * N_A * C_WINDOW:,} token positions exceeds '
                  f'the {C_TOKEN_CAP:,} cap')
        return keep

    factors = {
        'optimiser':     ('optim', ['adam', 'sgd']),
        'learning rate': ('lr', [1e-4, 3e-4, 3e-3]),
        'schedule':      ('schedule', ['constant', 'step']),
        'weight decay':  ('wd', [0.0, 5e-2]),
        'batch size':    ('batch', c_batch_values(C_BATCH)),
        'width':         ('hidden', [128, 512]),
        'depth':         ('layers', [1, 6]),
    }

    # Budget, from Study B's own measured seconds at this window rather than a guess.
    _n_runs = 1 + sum(len(v) for _, v in factors.values())
    _b = results.get('B_context') or {}
    _b_secs = next((_b[k].get('seconds') for k in (C_WINDOW, str(C_WINDOW)) if k in _b),
                   None)
    if _b_secs:
        _est = _n_runs * (_b_secs / BACKBONE_EPOCHS) * C_EPOCHS
        print(f'    budget: {_n_runs} backbones at W={C_WINDOW}, {C_EPOCHS} epochs each '
              f'≈ {_est / 3600:.1f} h\n            (Study B measured {_b_secs:.0f}s for '
              f'{BACKBONE_EPOCHS} epochs at W={C_WINDOW}; width and depth variants move '
              f'this either way)')
    else:
        print(f'    budget: {_n_runs} backbones at W={C_WINDOW}, {C_EPOCHS} epochs each. '
              f'No Study B timing for W={C_WINDOW} to estimate from.')

    ref_mpe, ref_s, _ = train_backbone(**ref)
    print(f'    reference {ref} -> MPE {ref_mpe:.6f} ({ref_s:.0f}s)')
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
    results['C_optimisation'] = {'reference': {**ref, 'mpe': ref_mpe},
                                 'window': C_WINDOW, 'epochs': C_EPOCHS,
                                 'note': (f'Run at W={C_WINDOW} over {C_EPOCHS} epochs, '
                                          f'not the deployed W={W_FULL}/'
                                          f'{BACKBONE_EPOCHS}: this study ranks factors '
                                          f'against each other, and the absolute MPE of '
                                          f'each cell is not comparable with Study A or '
                                          f'Module 7.'),
                                 'factors': sens}
    print('\n    sensitivity ranking (spread in MPE across the values tried):')
    for label, d in sorted(sens.items(), key=lambda kv: -kv[1]['spread_pct']):
        print(f'      {label:<15} {d["spread_pct"]:>6.1f}%')
    save_results('Study C')

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

# ════════════════════════════════════════════════════════════════════
# OUTPUT
# ════════════════════════════════════════════════════════════════════
save_results('all studies')
print(f'  Results written : {RESULTS_PATH}')

# Keyed on what results actually hold, not on the RUN_* flags: a resumed run has
# Study A and B loaded from disk with RUN_A/RUN_B False, and they still belong on the
# figure.
has_A = 'A_inference' in results
has_B = 'B_context' in results
has_C = 'C_optimisation' in results
n_panels = sum([has_A, has_B, has_C])
if n_panels:
    fig, axes = plt.subplots(1, max(n_panels, 2), figsize=(6.5 * max(n_panels, 2), 5))
    ax_i = 0
    if has_A:
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
    if has_B:
        ax = axes[ax_i]; ax_i += 1
        ws = sorted(results['B_context'], key=int)
        ax.plot(ws, [results['B_context'][w]['mpe'] for w in ws], marker='o', color='#1F6F6B')
        ax.set_xticks(ws); ax.set_xlabel('context frames W'); ax.set_ylabel('Mean Point Error')
        ax.set_title('B — does more past help?', fontweight='bold'); ax.grid(alpha=0.3)
    if has_C:
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
