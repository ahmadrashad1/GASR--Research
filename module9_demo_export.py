import os, json, math, numpy as np, torch

# ════════════════════════════════════════════════════════════════════
# MODULE 9 — DEMO EXPORT
#
# Writes everything the clinical demonstrator needs into one payload. Runs as a cell
# directly after Module 6, in the same runtime, and deliberately reuses Module 6's
# objects in memory (videos, predict_direct, base_pred, BUNDLE, WINDOW_SIZE, alpha,
# timing_by_ws) rather than re-deriving them. Re-deriving would mean a second
# implementation of the prediction path that could drift from the one every metric was
# measured with, and then the demonstrator would be showing something the paper never
# evaluated.
#
# WHAT THE VIEWER SHOWS, AND WHY EACH FIELD IS HERE
#   gt{k}     ground-truth geometry k frames ahead, from the held-out part of the video
#   pred{k}   our conditional diffusion prediction at that horizon
#   cv{k}     constant-velocity extrapolation at that horizon -- the baseline this
#             project has to clear. Showing it is the honest move: a demonstrator that
#             hides the baseline is a demonstrator nobody should believe. Which method
#             wins, at which horizon, is a question for the numbers below; this file
#             exports the comparison and asserts nothing about its outcome.
#   ctx       the last OBSERVED frame, which doubles as the freeze-in-place baseline
#   sigma{k}  per-point spread across independent sampled futures AT THAT HORIZON --
#             the model's own uncertainty, and the one quantity no deterministic
#             baseline provides. Sampled per horizon, never reused across horizons:
#             k=1's spread drawn around a k=3 prediction would understate the
#             uncertainty, which is the flattering direction and therefore the one
#             mistake this export must not make.
#
# The export is deliberately lossy: points and origins are subsampled and values are
# rounded, because this has to open in a browser rather than survive a re-analysis.
# Every number it contains is reproducible from the checkpoint; nothing here is the
# system of record. Module 6's JSON remains that.
# ════════════════════════════════════════════════════════════════════

# ── What this cell cannot do without ─────────────────────────────────
# Everything below comes from Module 6 and only from Module 6. Checked up front,
# because the alternative -- which is what this cell used to do -- is a bare
# `NameError: name 'videos' is not defined` a hundred lines in, after the config has
# printed and with nothing to say about the cause.
_NEEDS = {
    'videos':         'the per-video rebuilt trajectories',
    'predict_direct': 'the prediction path every metric was measured with',
    'base_pred':      'the constant-velocity / no-motion baseline',
    'to_physical':    'the inverse of Module 4/5 scaling',
    'BUNDLE':         'the loaded checkpoint',
    'WINDOW_SIZE':    'the context window',
    'DRIVE_BASE':     'the Drive mount',
    'DEVICE':         'the torch device',
}
_absent = [f'      {k:<16} {why}' for k, why in _NEEDS.items() if k not in globals()]
if _absent:
    raise SystemExit(
        '\n  MODULE 9 NEEDS MODULE 6 IN THE SAME RUNTIME.\n'
        '    Not defined here:\n' + '\n'.join(_absent) + '\n'
        '    Module 9 reuses Module 6\'s prediction path on purpose, so that the\n'
        '    demonstrator shows the predictor the paper reports and not a second\n'
        '    implementation of it. Run the Module 6 cell first, in this runtime, and\n'
        '    then run this cell. If the runtime disconnected mid-session, Module 6\n'
        '    has to be re-run: its objects live in memory, not on Drive.')

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

# Module 6 fits one magnitude scalar per horizon on TRAINING origins and reports its
# headline numbers with it applied. predict_direct defaults to alpha=1.0, so an export
# that does not pass it would quietly show a different predictor than the paper
# describes -- and the spread in sigma{k} would be on a different scale than the
# prediction it surrounds. Read from Module 6 if it ran the fit; 1.0 otherwise, which is
# the uncalibrated predictor and is recorded as such in the payload.
_cal_on = bool(globals().get('CALIBRATE', False) and globals().get('APPLY_CALIBRATION', False))
DEMO_ALPHA = {k: float(globals().get('alpha', {}).get(k, 1.0)) if _cal_on else 1.0
              for k in DEMO_HORIZONS}


# ── Pure helpers ─────────────────────────────────────────────────────
# No torch in these bodies, deliberately: tests/test_module9_export.py lifts them out of
# this file by AST and exercises them on a laptop with neither torch nor a Drive mount.

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
    camera space -- "World (= camera) points" -- so no extrinsics are involved.

    CALIBRATION IS NOT ASSUMED FROM THE FILE'S EXISTENCE. Module 1 always writes
    camera_meta.json, and always with fx derived from a nominal 75-degree laparoscope
    FOV: a textbook figure for a 10mm scope, not a measurement of the one that shot
    this video. Reading "the file is there" as "the camera is calibrated" marks every
    sequence calibrated and suppresses the warning that exists to stop an assumption
    reaching a projector with the confidence of a measurement.

    So the assumption is detected rather than trusted: metadata whose fx matches the
    FOV formula -- with or without the fov_deg field that Module 1 also writes -- is
    reported as 'assumed-fov'.

    This does NOT mean the overlay is misaligned. The reconstruction was itself built
    with these intrinsics, so projecting with them is self-consistent and the points
    land where the pipeline believes the tissue is. What is unfounded is any claim that
    the field of view matches the real scope."""
    H_, W_ = int(frame_hw[0]), int(frame_hw[1])
    if meta:
        fx, fy = float(meta['fx']), float(meta['fy'])
        W = int(meta.get('W', W_)); H = int(meta.get('H', H_))
        assumed = (W / 2) / math.tan(math.radians(float(meta.get('fov_deg', 75.0))) / 2)
        looks_assumed = ('fov_deg' in meta) or (abs(fx - assumed) <= 1e-6 * max(fx, 1.0))
        return {'fx': fx, 'fy': fy,
                'cx': float(meta['cx']), 'cy': float(meta['cy']),
                'W': W, 'H': H,
                'calibrated': not looks_assumed,
                'source': 'assumed-fov' if looks_assumed else 'measured'}
    fx = (W_ / 2) / math.tan(math.radians(75.0) / 2)
    return {'fx': fx, 'fy': fx, 'cx': W_ / 2.0, 'cy': H_ / 2.0,
            'W': W_, 'H': H_, 'calibrated': False, 'source': 'assumed-fov'}


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


print('=' * 60)
print('  MODULE 9 — Demo export for the clinical demonstrator')
print('=' * 60)
# Module 6 builds `videos`, a list of per-video dicts. Pick the one with the longest
# held-out tail: more origins to step through, and the least chance of the assert above
# firing on a short clip.
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
