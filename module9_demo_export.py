import os, json, math, numpy as np, torch

# ════════════════════════════════════════════════════════════════════
# MODULE 9 — DEMO EXPORT
#
# Writes everything the frontend viewer needs into one JSON file. Runs as a cell
# directly after Module 6, in the same runtime, and deliberately reuses Module 6's
# objects in memory (VIDEOS, predict_direct, base_pred, BUNDLE, WINDOW_SIZE) rather
# than re-deriving them. Re-deriving would mean a second implementation of the
# prediction path that could drift from the one every metric was measured with, and
# then the demo would be showing something the paper never evaluated.
#
# WHAT THE VIEWER SHOWS, AND WHY EACH FIELD IS HERE
#   gt     ground-truth geometry at t+k, from the held-out part of the video
#   pred   our conditional diffusion prediction at t+k
#   cv     constant-velocity extrapolation at t+k -- the baseline that beats us at
#          k=1 and loses by k=3. Showing it is the honest move: the crossover IS
#          the finding, and a demo that hides the baseline is a demo nobody believes
#   sigma  per-point spread across independent sampled futures -- the model's own
#          uncertainty, and the one quantity no deterministic baseline provides
#
# The export is deliberately lossy: points and origins are subsampled and values are
# rounded, because this has to open in a browser rather than survive a re-analysis.
# Every number it contains is reproducible from the checkpoint; nothing here is the
# system of record. Module 6's JSON remains that.
# ════════════════════════════════════════════════════════════════════

DEMO_DIR      = f'{DRIVE_BASE}/module9_demo'
DEMO_POINTS   = 256     # of N_ANCHOR, evenly strided -- a browser canvas cannot usefully
                         # draw 512 overlapping points at 30fps, and the strided subset is
                         # unbiased because Module 3's anchor order carries no spatial sort
DEMO_ORIGINS  = 80      # frames the viewer can step through
DEMO_HORIZONS = [1, 3]  # k=1 where constant velocity wins, k=3 where it has fallen apart
DEMO_SAMPLES  = 8       # hypotheses per origin for the per-point spread. Higher than the
                         # metric path's 4: this feeds a visual, where a noisy radius reads
                         # as a bug, and it costs one origin's sampling time
DEMO_ROUND    = 4

os.makedirs(DEMO_DIR, exist_ok=True)


def _subsample(n_total, n_want):
    return np.linspace(0, n_total - 1, min(n_want, n_total)).astype(int)


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

    origins = _subsample(hi - lo, DEMO_ORIGINS) + lo
    pts = _subsample(tn.shape[1], DEMO_POINTS)
    frames = []

    for t in origins:
        ctx = tn[t - WINDOW_SIZE:t].unsqueeze(0)              # (1, W, N_A, 3)
        row = {'t': int(t), 'ctx': tn[t - 1][pts].cpu().numpy()}
        for k in DEMO_HORIZONS:
            hs = torch.full((1,), k, device=DEVICE, dtype=torch.long)
            row[f'gt{k}']   = tn[t + k - 1][pts].cpu().numpy()
            row[f'pred{k}'] = predict_direct(ctx, k, disp_norm=dn)[0][pts].cpu().numpy()
            row[f'cv{k}']   = base_pred(ctx, hs)[0][pts].cpu().numpy()

        # Per-point uncertainty: independent futures from the identical context, so the
        # disagreement between them is the model's own, not a difference in what it saw.
        hyps = torch.stack([predict_direct(ctx, DEMO_HORIZONS[0], disp_norm=dn,
                                           eta=1.0, n_samples=1)[0]
                            for _ in range(DEMO_SAMPLES)])      # (S, N_A, 3)
        row['sigma'] = hyps.std(dim=0).norm(dim=-1)[pts].cpu().numpy()
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

    out = {
        'video': name,
        'n_points': int(len(pts)),
        'n_frames': int(len(frames)),
        'horizons': DEMO_HORIZONS,
        'window': int(WINDOW_SIZE),
        'centre': centre,
        'radius': radius,
        'target_mode': BUNDLE.get('mode', 'delta'),
        'checkpoint_epoch': BUNDLE.get('epoch', '?'),
        'mean_error': err,
        'frames': [
            dict({'t': f['t'], 'ctx': enc(f['ctx']), 'sigma': enc(f['sigma'])},
                 **{f'{tag}{k}': enc(f[f'{tag}{k}'])
                    for k in DEMO_HORIZONS for tag in ('gt', 'pred', 'cv')})
            for f in frames
        ],
    }
    path = f'{DEMO_DIR}/demo_data.json'
    with open(path, 'w') as fh:
        json.dump(out, fh, separators=(',', ':'))
    mb = os.path.getsize(path) / 1e6
    print(f'  {name}: {len(frames)} origins x {len(pts)} points -> {path}  ({mb:.1f} MB)')
    for k in DEMO_HORIZONS:
        e = err[str(k)]
        best = min(e, key=e.get)
        print(f'    k={k}  ours {e["ours"]:.5f}   const-vel {e["cv"]:.5f}   '
              f'no-motion {e["still"]:.5f}   -> {best} wins here')
    return out


print('=' * 60)
print('  MODULE 9 — Demo export for the frontend viewer')
print('=' * 60)
# Module 6 builds `videos`, a list of per-video dicts. Pick the one with the longest
# held-out tail: more origins to step through, and the least chance of the assert above
# firing on a short clip.
_vid = max(videos, key=lambda v: v['n_frames'] - int(v['split_frame']))
_name = str(_vid.get('video_id') or _vid.get('workspace') or 'video')
print(f'  Chosen video  : {_name}  ({_vid["n_frames"]} frames, held-out tail starts at '
      f'{_vid["split_frame"]})')
export_demo(_vid, _name)
print('\n  Copy demo_data.json next to demo/index.html and open it in a browser.')
