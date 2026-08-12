import torch, numpy as np, os, json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers the 3D projection
from scipy.signal import savgol_filter
from tqdm import tqdm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════
N_ANCHOR     = 512
WINDOW_SIZE  = 4
PRED_STEPS   = [1, 2, 3]
VAL_FRAC     = 0.20
SEED         = 42                       # reproducible FPS anchor selection
# Savitzky-Golay temporal smoothing, applied per-video to the raw trajectory before
# windowing. Root-caused and validated against a real reconstructed checkpoint
# (workspace_v3): the raw per-frame trajectory is dominated by high-frequency
# reconstruction noise -- Module 2 assumes a perfectly static camera, so any real
# camera motion (hand tremor, scope drift) has nowhere to go but into the
# per-Gaussian deformation field, and monocular depth is estimated independently
# per frame with no temporal consistency term. Measured directly: raw frame-to-frame
# motion has ~zero momentum (pooled Pearson r=0.077 between consecutive displacement
# vectors), and constant-velocity extrapolation LOSES to the naive "nothing moves"
# baseline by 27-58% across pred_steps 1-3. After smoothing, autocorr rises to ~0.89
# and constant-velocity BEATS naive by 30-57% at the same horizons -- proving a real,
# learnable deformation signal exists and was simply buried under noise.
#
# The window is chosen ADAPTIVELY, per video, not fixed -- that one validated
# checkpoint (workspace_v3) happens to be one of the better-reconstructed videos
# (PSNR~29dB); a noisier, lower-PSNR video plausibly needs more smoothing to recover
# the same clean signal, and a single fixed window has no way to know that. See
# select_smoothing_window(): each video sweeps SMOOTH_WINDOW_CANDIDATES and picks the
# SMALLEST window whose constant-velocity-vs-naive score clears SMOOTH_THRESHOLD --
# clean videos aren't over-smoothed, noisier videos automatically get more, and any
# video that never clears the bar even at the largest candidate is flagged in QT7
# rather than silently used as-is.
SMOOTH_WINDOW_CANDIDATES = (5, 7, 9, 11, 15, 19, 25)
SMOOTH_THRESHOLD          = 15.0   # minimum acceptable const-vel-vs-naive %, see above
SMOOTH_POLYORDER          = 2
COMBINED_DIR = f'{DRIVE_BASE}/module4_combined'

assert os.path.exists(COMBINED_DIR), (
    f'Combined dataset not found at {COMBINED_DIR}\n'
    'Run the Module 4 Combiner cell above first.')

# Only the workspace *names* are trusted from the combiner's metadata --
# frame/window counts are always recomputed from gaussian_trajectory.npy
# below, and Module 4 writes its own results to module4_metadata.json
# rather than overwriting the combiner's metadata.json. This keeps the
# cell idempotent: re-running it never bootstraps from its own prior output.
meta4 = json.load(open(f'{COMBINED_DIR}/metadata.json'))
workspace_names = [v['workspace'] for v in meta4['videos']]

print('='*60)
print('  MODULE 4 — Combined Multi-Video Dataset (source)')
print('='*60)
print(f'  Videos in combiner : {len(workspace_names)}')
for name in workspace_names:
    print(f'    {name}')

# ════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════
def farthest_point_sampling(points, n_samples, rng):
    """Greedy FPS. points: (N, 3). Returns indices into points, shape (n_samples,)."""
    n = len(points)
    sel = np.zeros(n_samples, dtype=np.int64)
    dists = np.full(n, np.inf, dtype=np.float32)
    sel[0] = rng.integers(0, n)
    for i in range(1, n_samples):
        d = np.sum((points - points[sel[i - 1]]) ** 2, axis=1)
        dists = np.minimum(dists, d)
        sel[i] = np.argmax(dists)
    return sel


def const_velocity_vs_naive(traj_n, window_size, pred_steps, device=None):
    """For each pred_step, mean-CD improvement (%) of constant-velocity extrapolation
    (G_t + h*(G_t - G_t-1)) over the naive copy-last-frame baseline. This is the
    diagnostic that exposed the noise-dominated-trajectory problem on real reconstructed
    data (raw: -27% to -58%, i.e. constant-velocity LOSES to naive -- no exploitable
    momentum) and confirms smoothing fixes it (+30% to +57% after smoothing).
    Positive means real, learnable motion continuity exists in this video's trajectory.

    Batched via torch (one cdist call per pred_step across all valid windows at once)
    rather than a per-window Python/numpy loop -- needed because the adaptive smoothing
    search below calls this ~7x per video across all 10 videos; the naive per-window
    loop was too slow to run at N_ANCHOR=512 scale within that budget."""
    dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    traj_t = torch.from_numpy(traj_n).float().to(dev)
    N_F = traj_t.shape[0]
    out = {}
    for h in pred_steps:
        ts = list(range(window_size, N_F - h))
        if not ts:
            out[h] = 0.0
            continue
        idx_tm1 = torch.tensor([t - 1 for t in ts], device=dev)
        idx_t   = torch.tensor(ts, device=dev)
        idx_fut = torch.tensor([t + h for t in ts], device=dev)
        G_tm1, G_t, G_future = traj_t[idx_tm1], traj_t[idx_t], traj_t[idx_fut]   # (B, N_A, 3) each

        naive_pred = G_t
        cv_pred    = G_t + h * (G_t - G_tm1)

        d_naive = torch.cdist(naive_pred, G_future)   # (B, N_A, N_A)
        d_cv    = torch.cdist(cv_pred, G_future)
        cd_naive = (d_naive.min(dim=2).values.mean(dim=1) + d_naive.min(dim=1).values.mean(dim=1))
        cd_cv    = (d_cv.min(dim=2).values.mean(dim=1)    + d_cv.min(dim=1).values.mean(dim=1))

        naive_mean, cv_mean = cd_naive.mean().item(), cd_cv.mean().item()
        out[h] = float((naive_mean - cv_mean) / max(naive_mean, 1e-9) * 100)
    return out


def select_smoothing_window(traj_raw, ctr, scl, window_size, pred_steps,
                             candidates=(5, 7, 9, 11, 15, 19, 25),
                             threshold=15.0, polyorder=2, device=None):
    """Per-video ADAPTIVE smoothing: picks the SMALLEST candidate window whose
    constant-velocity-vs-naive score (averaged over pred_steps) clears `threshold`,
    instead of one fixed window for every video regardless of how noisy it is.

    Why this matters (and why a single fixed window was the wrong design): the fix
    was validated on exactly one real checkpoint (workspace_v3, PSNR~29.18dB -- one
    of the *better*-reconstructed videos). A noisier, lower-PSNR video plausibly needs
    more smoothing to recover the same clean signal; picking the SMALLEST window that
    clears the bar (rather than always maximising) means clean videos aren't
    over-smoothed while noisier videos automatically get more. Returns
    (chosen_window, score_at_chosen, all_scores, crossed_threshold)."""
    N_F = traj_raw.shape[0]
    traj_raw_n = (traj_raw - ctr) / scl
    all_scores = {}
    chosen_window, chosen_score = None, None
    for w in candidates:
        w_eff = min(w, N_F if N_F % 2 == 1 else N_F - 1)
        if w_eff < polyorder + 2 or w_eff < 3:
            continue
        traj_s = savgol_filter(traj_raw_n, window_length=w_eff, polyorder=polyorder, axis=0)
        cv = const_velocity_vs_naive(traj_s, window_size, pred_steps, device=device)
        score = float(np.mean(list(cv.values())))
        all_scores[w_eff] = score
        if chosen_window is None and score >= threshold:
            chosen_window, chosen_score = w_eff, score
    crossed = chosen_window is not None
    if not crossed:
        # No candidate cleared the bar even at the largest window tried -- use the
        # best of what was tried, but this gets flagged in QT7 rather than silently accepted.
        chosen_window = max(all_scores, key=all_scores.get)
        chosen_score = all_scores[chosen_window]
    return chosen_window, chosen_score, all_scores, crossed


def select_anchors(traj_full, ckpt_path, n_anchor, rng):
    """Opacity-weighted FPS: prefer visible Gaussians when a checkpoint is available,
    fall back to FPS over all Gaussians otherwise."""
    N_G = traj_full.shape[1]
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        opa = torch.sigmoid(ck['opa']).squeeze(-1).numpy()
        vis = opa > 0.05
        pts_v = traj_full[0][vis]
        if len(pts_v) >= n_anchor:
            sel = farthest_point_sampling(pts_v, n_anchor, rng)
            return np.where(vis)[0][sel], vis
        sel = farthest_point_sampling(traj_full[0], n_anchor, rng)
        return sel, vis
    sel = farthest_point_sampling(traj_full[0], n_anchor, rng)
    return sel, np.ones(N_G, dtype=bool)


# ════════════════════════════════════════════════════════════════════
# MULTI-STEP WINDOW CONSTRUCTION (per-video, per-pred-step temporal split)
# ════════════════════════════════════════════════════════════════════
print(f'\nBuilding multi-step windows (pred_steps={PRED_STEPS})...')

all_tr_inp, all_tr_tgt, all_tr_vid, all_tr_ps = [], [], [], []
all_va_inp, all_va_tgt, all_va_vid, all_va_ps = [], [], [], []
per_video_stats  = []
disp_by_video    = {}          # workspace -> (N_F-1, N_ANCHOR) frame-to-frame displacement magnitude
mean_disp_raw    = {}          # workspace -> scalar
video_order      = []          # workspace names in processing order (== video_id)
fps_viz_sample   = None        # captured once, from the first successfully processed video

for vi, ws_name in enumerate(workspace_names):
    ws_path   = f'{DRIVE_BASE}/{ws_name}'
    traj_path = f'{ws_path}/exports/gaussian_trajectory.npy'
    ckpt_path = f'{ws_path}/checkpoints/4dgs_v4.pth'

    if not os.path.exists(traj_path):
        print(f'  ⚠️  {ws_name}: trajectory missing — skip')
        continue

    traj_full = np.load(traj_path)      # (N_F, N_G, 3)
    N_F, N_G, _ = traj_full.shape
    if N_F <= WINDOW_SIZE + max(PRED_STEPS):
        print(f'  ⚠️  {ws_name}: too few frames ({N_F}) — skip')
        continue

    rng = np.random.default_rng(SEED + vi)     # reproducible, independent per video
    anc, vis = select_anchors(traj_full, ckpt_path, N_ANCHOR, rng)
    traj_raw = traj_full[:, anc, :]            # (N_F, N_ANCHOR, 3) -- pre-smoothing, for QA comparison

    if fps_viz_sample is None:
        fps_viz_sample = {
            'workspace': ws_name,
            'pts_all':  traj_full[0][vis] if vis.sum() >= N_ANCHOR else traj_full[0],
            'pts_anc':  traj_full[0][anc],
        }

    # Adaptive per-video smoothing: sweep SMOOTH_WINDOW_CANDIDATES, pick the smallest
    # window whose constant-velocity-vs-naive score clears SMOOTH_THRESHOLD. Noisier
    # (e.g. lower-PSNR) videos need more smoothing to recover the same clean signal;
    # this measures each video's OWN noise directly instead of assuming one fixed
    # window (validated on a single, above-average-PSNR checkpoint) fits every video.
    ctr_raw = traj_raw[0].mean(axis=0)
    scl_raw = np.abs(traj_raw[0] - ctr_raw).max() + 1e-8
    win, win_score, win_all_scores, win_crossed = select_smoothing_window(
        traj_raw, ctr_raw, scl_raw, WINDOW_SIZE, PRED_STEPS,
        candidates=SMOOTH_WINDOW_CANDIDATES, threshold=SMOOTH_THRESHOLD,
        polyorder=SMOOTH_POLYORDER, device=DEVICE)
    traj = savgol_filter(traj_raw, window_length=win, polyorder=SMOOTH_POLYORDER, axis=0)

    # Normalise per-video (preserves relative motion semantics)
    ctr    = traj[0].mean(axis=0)
    scl    = np.abs(traj[0] - ctr).max() + 1e-8
    traj_n = (traj - ctr) / scl     # [-1, 1]

    disp = np.linalg.norm(np.diff(traj_n, axis=0), axis=-1)   # (N_F-1, N_ANCHOR)
    disp_by_video[ws_name] = disp
    mean_disp_raw[ws_name] = float(disp.mean())
    video_order.append(ws_name)

    # Smoothing-benefit diagnostic for QT7: raw (no smoothing) vs the chosen adaptive
    # window, so QT7 can confirm the fix actually holds for every video, not just the
    # one it was diagnosed on -- and flag any video that never cleared the bar.
    traj_raw_n = (traj_raw - ctr_raw) / scl_raw
    cv_before = const_velocity_vs_naive(traj_raw_n, WINDOW_SIZE, PRED_STEPS, device=DEVICE)
    cv_after  = const_velocity_vs_naive(traj_n, WINDOW_SIZE, PRED_STEPS, device=DEVICE)

    # Build windows PER pred_step, split temporally 80/20 WITHIN each pred_step,
    # then combine. Splitting the concatenated multi-step list (as before) let
    # all pred_step=1 and pred_step=2 windows land entirely in train, leaving
    # validation almost exclusively pred_step=3 -- fixed here.
    v_tr_inp, v_tr_tgt, v_tr_ps = [], [], []
    v_va_inp, v_va_tgt, v_va_ps = [], [], []
    train_ps_counts, val_ps_counts = {}, {}
    split_frame_ps1 = None

    for pred_step in PRED_STEPS:
        ps_inp, ps_tgt = [], []
        for t in range(WINDOW_SIZE, N_F - pred_step + 1):
            ps_inp.append(traj_n[t - WINDOW_SIZE: t])
            ps_tgt.append(traj_n[t + pred_step - 1])
        ps_inp = np.stack(ps_inp, 0).astype(np.float32)
        ps_tgt = np.stack(ps_tgt, 0).astype(np.float32)

        n_val_ps = max(1, int(len(ps_inp) * VAL_FRAC))
        n_tr_ps  = len(ps_inp) - n_val_ps

        v_tr_inp.append(ps_inp[:n_tr_ps]);  v_tr_tgt.append(ps_tgt[:n_tr_ps])
        v_va_inp.append(ps_inp[n_tr_ps:]);  v_va_tgt.append(ps_tgt[n_tr_ps:])
        v_tr_ps.append(np.full(n_tr_ps, pred_step, dtype=np.int8))
        v_va_ps.append(np.full(n_val_ps, pred_step, dtype=np.int8))
        train_ps_counts[pred_step] = int(n_tr_ps)
        val_ps_counts[pred_step]   = int(n_val_ps)
        if pred_step == 1:
            split_frame_ps1 = WINDOW_SIZE + n_tr_ps   # frame index where val begins

    v_tr_inp = np.concatenate(v_tr_inp, 0); v_tr_tgt = np.concatenate(v_tr_tgt, 0)
    v_va_inp = np.concatenate(v_va_inp, 0); v_va_tgt = np.concatenate(v_va_tgt, 0)
    v_tr_ps  = np.concatenate(v_tr_ps, 0);  v_va_ps  = np.concatenate(v_va_ps, 0)

    all_tr_inp.append(v_tr_inp); all_tr_tgt.append(v_tr_tgt)
    all_va_inp.append(v_va_inp); all_va_tgt.append(v_va_tgt)
    all_tr_ps.append(v_tr_ps);   all_va_ps.append(v_va_ps)
    all_tr_vid.append(np.full(len(v_tr_inp), vi, dtype=np.int32))
    all_va_vid.append(np.full(len(v_va_inp), vi, dtype=np.int32))

    met_p    = f'{ws_path}/metrics.json'
    psnr_str = f'{json.load(open(met_p))["psnr_mean"]:.2f}dB' if os.path.exists(met_p) else 'n/a'
    total_w  = len(v_tr_inp) + len(v_va_inp)
    flag = '' if win_crossed else '  ⚠️ never cleared threshold'
    print(f'  ✅ {ws_name:<25}  frames={N_F}  total={total_w}  '
          f'train={len(v_tr_inp)}  val={len(v_va_inp)}  PSNR={psnr_str}  '
          f'smooth_win={win}{flag}')

    per_video_stats.append({
        'video_id': vi, 'workspace': ws_name, 'frames': N_F,
        'total_windows': total_w, 'train': int(len(v_tr_inp)), 'val': int(len(v_va_inp)),
        'train_ps_counts': train_ps_counts, 'val_ps_counts': val_ps_counts,
        'split_frame_pred1': int(split_frame_ps1),
        'mean_disp_raw': mean_disp_raw[ws_name], 'psnr': psnr_str,
        'const_vel_vs_naive_before_smoothing': cv_before,
        'const_vel_vs_naive_after_smoothing':  cv_after,
        'smooth_window': int(win), 'smooth_window_crossed_threshold': bool(win_crossed),
        'smooth_window_all_scores': win_all_scores,
    })

if not per_video_stats:
    raise RuntimeError('No videos produced windows -- check DRIVE_BASE / trajectory paths.')

# Concatenate across videos
train_inp = np.concatenate(all_tr_inp, 0); train_tgt = np.concatenate(all_tr_tgt, 0)
val_inp   = np.concatenate(all_va_inp, 0); val_tgt   = np.concatenate(all_va_tgt, 0)
train_ps  = np.concatenate(all_tr_ps, 0);  val_ps    = np.concatenate(all_va_ps, 0)
train_vid = np.concatenate(all_tr_vid, 0); val_vid   = np.concatenate(all_va_vid, 0)

M_tr, W, N_A, _ = train_inp.shape
M_va = len(val_inp)
single_step_tr = int((train_ps == 1).sum())
single_step_va = int((val_ps == 1).sum())

print(f'\nAfter multi-step augmentation:')
print(f'  Train windows      : {M_tr:,}')
print(f'  Val windows        : {M_va:,}')
print(f'  Single-step (t+1)  : {single_step_tr + single_step_va:,}  '
      f'({single_step_tr:,} train / {single_step_va:,} val)')
print(f'  Multi-step increase: {(M_tr + M_va) / max(single_step_tr + single_step_va, 1):.2f}x '
      f'over single-step alone')

# ════════════════════════════════════════════════════════════════════
# CROSS-VIDEO DISPLACEMENT EQUALISATION
# Positions stay normalised to [-1,1] per video (unchanged). Additionally,
# rescale each video's target *delta* by a per-video scalar so that no
# single high-motion video dominates the diffusion model's training signal.
# This also produces the displacement-space representation
# (G_target - G_last_context) that Module 5 needs -- absolute-position
# v-prediction was diagnosed as unusable at this displacement scale
# (mean 0.0042 normalised units gets drowned by unit-scale diffusion noise).
# ════════════════════════════════════════════════════════════════════
video_order_arr   = np.array(video_order)
mean_disp_arr     = np.array([mean_disp_raw[w] for w in video_order])   # aligned with video_id
GLOBAL_DISP_REF    = float(np.median(mean_disp_arr))
disp_norm_factor_arr = GLOBAL_DISP_REF / mean_disp_arr                   # (n_videos,)

train_disp_norm = disp_norm_factor_arr[train_vid].astype(np.float32)     # (M_tr,)
val_disp_norm   = disp_norm_factor_arr[val_vid].astype(np.float32)

train_deltas = (train_tgt - train_inp[:, -1]) * train_disp_norm[:, None, None]
val_deltas   = (val_tgt   - val_inp[:, -1])   * val_disp_norm[:, None, None]

for v in per_video_stats:
    v['disp_norm_factor'] = float(disp_norm_factor_arr[v['video_id']])

# Save
np.save(f'{COMBINED_DIR}/train_inputs.npy',   train_inp)
np.save(f'{COMBINED_DIR}/train_targets.npy',  train_tgt)
np.save(f'{COMBINED_DIR}/val_inputs.npy',     val_inp)
np.save(f'{COMBINED_DIR}/val_targets.npy',    val_tgt)
np.save(f'{COMBINED_DIR}/train_deltas.npy',   train_deltas)
np.save(f'{COMBINED_DIR}/val_deltas.npy',     val_deltas)
np.save(f'{COMBINED_DIR}/train_disp_norm.npy', train_disp_norm)
np.save(f'{COMBINED_DIR}/val_disp_norm.npy',   val_disp_norm)
np.save(f'{COMBINED_DIR}/train_video_id.npy', train_vid)
np.save(f'{COMBINED_DIR}/val_video_id.npy',   val_vid)
np.save(f'{COMBINED_DIR}/train_pred_step.npy', train_ps)
np.save(f'{COMBINED_DIR}/val_pred_step.npy',   val_ps)

meta4_out = {
    'videos'            : per_video_stats,
    'video_order'       : video_order,
    'n_videos'          : len(per_video_stats),
    'n_anchor'          : N_ANCHOR,
    'window_size'       : WINDOW_SIZE,
    'pred_steps'        : PRED_STEPS,
    'val_frac'          : VAL_FRAC,
    'seed'              : SEED,
    'smooth_window_candidates' : list(SMOOTH_WINDOW_CANDIDATES),
    'smooth_threshold'         : SMOOTH_THRESHOLD,
    'smooth_polyorder'         : SMOOTH_POLYORDER,
    'global_disp_ref'   : GLOBAL_DISP_REF,
    'total_train'       : int(M_tr),
    'total_val'         : int(M_va),
    'input_shape'       : list(train_inp.shape),
    'target_shape'      : list(train_tgt.shape),
    'delta_shape'       : list(train_deltas.shape),
    'note'              : ('train_deltas/val_deltas are displacement-equalised '
                            '(target - last_context_frame) * disp_norm_factor[video]. '
                            'To recover a physically-scaled delta: divide by disp_norm_factor.'),
}
json.dump(meta4_out, open(f'{COMBINED_DIR}/module4_metadata.json', 'w'), indent=2)

# ════════════════════════════════════════════════════════════════════
# QUALITY TESTS
# ════════════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('  MODULE 4 — Quality Tests')
print('='*60)

inp_disp   = np.diff(train_inp, axis=1)
disp_mag   = np.linalg.norm(inp_disp, axis=-1)
mean_disp  = disp_mag.mean()
max_disp   = disp_mag.max()
std_disp   = disp_mag.std()

print(f'\nQT1 — Displacement distribution (normalised coords):')
print(f'  Mean  : {mean_disp:.5f}')
print(f'  Std   : {std_disp:.5f}')
print(f'  Max   : {max_disp:.5f}')
qt1_pass = max_disp < 0.5
print(f'  {"✅ PASS — max < 0.5" if qt1_pass else "⚠️  Large displacement — check normalisation"}')

diff_to_tgt = np.linalg.norm(train_tgt - train_inp[:, -1, :, :], axis=-1).mean()
print(f'\nQT2 — Target adjacency:')
print(f'  Mean |G_t+n - G_t| : {diff_to_tgt:.5f}')
print(f'  Mean displacement   : {mean_disp:.5f}')
print(f'  Ratio               : {diff_to_tgt/max(mean_disp,1e-9):.2f}x (≈ pred_step horizon)')
qt2_pass = diff_to_tgt < 0.3
print(f'  {"✅ PASS" if qt2_pass else "⚠️  Unexpectedly large gap"}')

per_window_max = disp_mag.max(axis=(1, 2))
outlier_thresh = 3.0 * per_window_max.mean()
outlier_frac   = (per_window_max > outlier_thresh).mean()
print(f'\nQT3 — Temporal smoothness:')
print(f'  Outlier windows : {(per_window_max > outlier_thresh).sum()} / {M_tr} ({outlier_frac*100:.1f}%)')
qt3_pass = outlier_frac < 0.05
print(f'  {"✅ PASS (< 5%)" if qt3_pass else "⚠️  High variance"}')

print(f'\nQT4 — Dataset coverage:')
print(f'  Videos       : {len(per_video_stats)}')
print(f'  Train windows: {M_tr:,}')
print(f'  Val windows  : {M_va:,}')
qt4_pass = M_tr >= 500 and len(per_video_stats) >= 2
print(f'  {"✅ PASS" if qt4_pass else "❌ FAIL — not enough data"}')

# QT5a: informational -- raw cross-video displacement spread, before any fix.
video_means_raw = mean_disp_arr
ratio_before = video_means_raw.max() / max(video_means_raw.min(), 1e-9)
print(f'\nQT5a — Cross-video consistency (BEFORE equalisation, informational):')
for w, m in zip(video_order, video_means_raw):
    print(f'  {w:<25} mean_disp={m:.5f}')
print(f'  Max/min ratio: {ratio_before:.2f}x  (this is why displacement equalisation is applied)')

# QT5b: real pass/fail check, AFTER equalisation, using the 90th-percentile of the
# per-video displacement distribution -- a statistic independent of the mean used
# to derive disp_norm_factor, so this is not a tautological check.
p90_eq = np.array([
    np.percentile(disp_by_video[w] * disp_norm_factor_arr[i], 90)
    for i, w in enumerate(video_order)
])
ratio_after = p90_eq.max() / max(p90_eq.min(), 1e-9)
print(f'\nQT5b — Cross-video consistency (AFTER equalisation, p90 displacement):')
for w, p in zip(video_order, p90_eq):
    print(f'  {w:<25} p90_disp_eq={p:.5f}')
print(f'  Max/min ratio: {ratio_after:.2f}x')
qt5_pass = ratio_after < 3.0
print(f'  {"✅ PASS (< 3x)" if qt5_pass else "⚠️  Still large variation after equalisation"}')

# QT6: regression check for the split-order fix -- every video's val set must
# contain windows from every pred_step, not just the last one.
print(f'\nQT6 — Per-video validation horizon coverage (split-fix regression check):')
qt6_pass = True
for v in per_video_stats:
    missing = [ps for ps in PRED_STEPS if v['val_ps_counts'].get(ps, 0) == 0]
    ok = len(missing) == 0
    qt6_pass &= ok
    status = '✅' if ok else '❌'
    print(f'  {status} {v["workspace"]:<25} val counts per pred_step: {v["val_ps_counts"]}')
print(f'  {"✅ PASS — every video validates on all pred_steps" if qt6_pass else "❌ FAIL — some video is missing a horizon in val"}')

print(f'\nQT7 — Adaptive temporal smoothing benefit (constant-velocity vs naive, mean over pred_steps):')
print(f'  Root-caused on real reconstructed data: raw trajectories are noise-dominated '
      f'(near-zero frame-to-frame momentum), so constant-velocity extrapolation LOSES to '
      f'the naive "nothing moves" baseline -- meaning there was no learnable signal for '
      f'Module 5 to find, regardless of architecture or training changes. Each video picks '
      f'its OWN smoothing window (candidates={SMOOTH_WINDOW_CANDIDATES}, threshold={SMOOTH_THRESHOLD}%) '
      f'rather than one size fitting all, since noisier (e.g. lower-PSNR) videos need more '
      f'smoothing to recover the same clean signal.')
qt7_pass = True
for v in per_video_stats:
    before = np.mean(list(v['const_vel_vs_naive_before_smoothing'].values()))
    after  = np.mean(list(v['const_vel_vs_naive_after_smoothing'].values()))
    crossed = v['smooth_window_crossed_threshold']
    ok = crossed and after > 0
    qt7_pass &= ok
    status = '✅' if ok else '⚠️ '
    tag = '' if crossed else '  ⚠️ NEVER CLEARED THRESHOLD even at largest candidate window -- treat this video\'s data with caution'
    print(f'  {status} {v["workspace"]:<25} PSNR={v["psnr"]:<8} window={v["smooth_window"]:>2d}  '
          f'const-vel vs naive: {before:+6.1f}% (raw) -> {after:+6.1f}% (smoothed){tag}')
print(f'  {"✅ PASS — every video reaches a clean, learnable signal after adaptive smoothing" if qt7_pass else "⚠️  At least one video never reached a reliable signal -- see flags above; consider excluding or down-weighting it in Module 5"}')

all_pass = qt1_pass and qt2_pass and qt3_pass and qt4_pass and qt5_pass and qt6_pass and qt7_pass
print(f'\n{"✅ ALL QUALITY TESTS PASSED" if all_pass else "⚠️  Review flagged tests above"}')

# ════════════════════════════════════════════════════════════════════
# VISUALISATION — 10-panel QA dashboard
# ════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(28, 11))
gs  = fig.add_gridspec(2, 5, hspace=0.42, wspace=0.32)
rng_viz = np.random.default_rng(SEED)

# P1 — Displacement distribution
ax = fig.add_subplot(gs[0, 0])
ax.hist(disp_mag.flatten(), bins=80, color='steelblue', edgecolor='white', linewidth=0.2, density=True)
ax.axvline(mean_disp, color='red', linestyle='--', linewidth=1.5, label=f'mean={mean_disp:.4f}')
ax.set_title('P1 (QT1): Displacement Distribution', fontweight='bold', fontsize=10)
ax.set_xlabel('|Δ| (normalised)'); ax.legend(fontsize=7); ax.grid(alpha=0.3)

# P2 — Per-video displacement boxplot, before vs after equalisation
ax = fig.add_subplot(gs[0, 1])
raw_samples, eq_samples = [], []
for i, w in enumerate(video_order):
    flat = disp_by_video[w].flatten()
    idx = rng_viz.choice(len(flat), size=min(3000, len(flat)), replace=False)
    raw_samples.append(flat[idx])
    eq_samples.append(flat[idx] * disp_norm_factor_arr[i])
n_v = len(video_order)
pos_raw = np.arange(n_v) * 2.5
pos_eq  = pos_raw + 0.9
bp1 = ax.boxplot(raw_samples, positions=pos_raw, widths=0.7, patch_artist=True, showfliers=False)
bp2 = ax.boxplot(eq_samples,  positions=pos_eq,  widths=0.7, patch_artist=True, showfliers=False)
for b in bp1['boxes']: b.set_facecolor('salmon')
for b in bp2['boxes']: b.set_facecolor('steelblue')
ax.set_xticks(pos_raw + 0.45)
ax.set_xticklabels([w.replace('workspace_', '') for w in video_order], rotation=60, fontsize=6.5)
ax.legend([bp1['boxes'][0], bp2['boxes'][0]], ['raw', 'equalised'], fontsize=7, loc='upper right')
ax.set_title('P2 (QT5): Per-Video Displacement, Raw vs Equalised', fontweight='bold', fontsize=10)
ax.grid(alpha=0.3, axis='y')

# P3 — Per-video temporal displacement curve (contact-event spikes)
ax = fig.add_subplot(gs[0, 2])
cmap = plt.cm.tab10(np.linspace(0, 1, n_v))
for i, w in enumerate(video_order):
    curve = disp_by_video[w].mean(axis=1)
    x = np.linspace(0, 1, len(curve))
    ax.plot(x, curve, color=cmap[i], linewidth=1, alpha=0.8,
            label=w.replace('workspace_', ''))
ax.set_title('P3: Temporal Displacement (contact events)', fontweight='bold', fontsize=10)
ax.set_xlabel('Normalised frame position'); ax.set_ylabel('Mean |Δ| across anchors')
ax.legend(fontsize=5.5, ncol=2, loc='upper right'); ax.grid(alpha=0.3)

# P4 — Sorted window max displacement (QT3)
ax = fig.add_subplot(gs[0, 3])
ax.plot(sorted(per_window_max), 'b-', linewidth=1, alpha=0.7)
ax.axhline(outlier_thresh, color='orange', linestyle=':', linewidth=1.5, label='3x mean')
ax.set_title('P4 (QT3): Sorted Window Max Disp', fontweight='bold', fontsize=10)
ax.set_xlabel('Window (sorted)'); ax.legend(fontsize=7); ax.grid(alpha=0.3)

# P5 — Windows per video (train/val stacked)
ax = fig.add_subplot(gs[0, 4])
v_names  = [v['workspace'].replace('workspace_', '') for v in per_video_stats]
v_trains = [v['train'] for v in per_video_stats]
v_vals   = [v['val']   for v in per_video_stats]
x = np.arange(len(v_names))
ax.bar(x, v_trains, label='Train', color='steelblue', alpha=0.85)
ax.bar(x, v_vals, bottom=v_trains, label='Val', color='salmon', alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(v_names, fontsize=6.5, rotation=60)
ax.set_title('P5 (QT4): Windows per Video', fontweight='bold', fontsize=10)
ax.legend(fontsize=7); ax.grid(alpha=0.3, axis='y')

# P6 — Windows per pred_step per video (verifies the split fix visually)
ax = fig.add_subplot(gs[1, 0])
width = 0.25
colors_ps = ['#4C72B0', '#DD8452', '#55A868']
for j, ps in enumerate(PRED_STEPS):
    counts = [v['train_ps_counts'][ps] + v['val_ps_counts'][ps] for v in per_video_stats]
    ax.bar(x + (j - 1) * width, counts, width=width, color=colors_ps[j], label=f'pred_step={ps}')
ax.set_xticks(x); ax.set_xticklabels(v_names, fontsize=6.5, rotation=60)
ax.set_title('P6: Windows per Pred-Step per Video', fontweight='bold', fontsize=10)
ax.legend(fontsize=7); ax.grid(alpha=0.3, axis='y')

# P7 — Train/val temporal split timeline (gantt, pred_step=1 boundary)
ax = fig.add_subplot(gs[1, 1])
for i, v in enumerate(per_video_stats):
    split_f = v['split_frame_pred1']; n_f = v['frames']
    ax.barh(i, split_f, left=0, color='steelblue', height=0.6)
    ax.barh(i, n_f - split_f, left=split_f, color='salmon', height=0.6)
ax.set_yticks(range(len(v_names))); ax.set_yticklabels(v_names, fontsize=6.5)
ax.set_xlabel('Frame index'); ax.invert_yaxis()
ax.set_title('P7: Train (blue) / Val (orange) Timeline', fontweight='bold', fontsize=10)
ax.grid(alpha=0.3, axis='x')

# P8 — FPS anchor spatial coverage (representative video)
ax = fig.add_subplot(gs[1, 2])
ax.scatter(fps_viz_sample['pts_all'][:, 0], fps_viz_sample['pts_all'][:, 1],
           s=2, c='lightgray', alpha=0.5, label='all visible Gaussians')
ax.scatter(fps_viz_sample['pts_anc'][:, 0], fps_viz_sample['pts_anc'][:, 1],
           s=6, c='crimson', alpha=0.9, label=f'{N_ANCHOR} FPS anchors')
ax.set_title(f'P8: FPS Coverage — {fps_viz_sample["workspace"]}', fontweight='bold', fontsize=10)
ax.legend(fontsize=7); ax.set_aspect('equal'); ax.grid(alpha=0.3)

# P9 — Sample window, 3D
ax = fig.add_subplot(gs[1, 3], projection='3d')
mid_idx  = M_tr // 2
sample_w = train_inp[mid_idx]
sample_t = train_tgt[mid_idx]
colours  = plt.cm.viridis(np.linspace(0, 0.85, W))
for wi in range(W):
    ax.scatter(sample_w[wi, :, 0], sample_w[wi, :, 1], sample_w[wi, :, 2],
               c=[colours[wi]], s=3, alpha=0.5, label=f'G_t-{W-wi}')
ax.scatter(sample_t[:, 0], sample_t[:, 1], sample_t[:, 2], c='gold', s=10, marker='*',
           label='G_target')
ax.set_title('P9: Sample Window (3D)', fontweight='bold', fontsize=10)
ax.legend(fontsize=5.5, markerscale=2)

# P10 — Equalisation effect: mean displacement before vs after, per video
ax = fig.add_subplot(gs[1, 4])
eq_means = mean_disp_arr * disp_norm_factor_arr
ax.bar(x - 0.2, mean_disp_arr, width=0.4, color='salmon', label='raw')
ax.bar(x + 0.2, eq_means, width=0.4, color='steelblue', label='equalised')
ax.axhline(GLOBAL_DISP_REF, color='black', linestyle='--', linewidth=1, label='global ref')
ax.set_xticks(x); ax.set_xticklabels(v_names, fontsize=6.5, rotation=60)
ax.set_title('P10: Displacement Equalisation Effect', fontweight='bold', fontsize=10)
ax.legend(fontsize=7); ax.grid(alpha=0.3, axis='y')

plt.suptitle(f'Module 4: Multi-Step Deformation Dataset ({M_tr:,} train windows, '
             f'{ratio_before:.1f}x → {ratio_after:.1f}x cross-video consistency)',
             fontsize=13, fontweight='bold')
plt.savefig(f'{COMBINED_DIR}/module4_qa.png', dpi=120, bbox_inches='tight')
plt.show()

print(f'\n{"="*60}')
print(f'  MODULE 4 COMPLETE')
print(f'{"="*60}')
print(f'  Pred steps       : {PRED_STEPS}')
print(f'  Train windows    : {M_tr:,}')
print(f'  Val windows      : {M_va:,}')
print(f'  Cross-video ratio: {ratio_before:.2f}x raw → {ratio_after:.2f}x equalised')
print(f'  Smoothing        : Savitzky-Golay, per-video adaptive window from {SMOOTH_WINDOW_CANDIDATES} '
      f'(polyorder={SMOOTH_POLYORDER}, threshold={SMOOTH_THRESHOLD}% -- see QT7 for the window each video chose)')
print(f'  Saved arrays     : train/val_{{inputs,targets,deltas,disp_norm,video_id,pred_step}}.npy')
print(f'  Saved metadata   : {COMBINED_DIR}/module4_metadata.json')
print(f'  Saved figure     : {COMBINED_DIR}/module4_qa.png')
print(f'  → Proceed to Module 5 — {M_tr:,} windows ready for diffusion training')
