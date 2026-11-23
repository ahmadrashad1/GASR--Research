# ════════════════════════════════════════════════════════════════════════════════
#  LIVE BUNDLE — the geometry the local inference server needs
#
#  Run this as a Colab cell AFTER the Module 9 export cell, in the same session, so
#  `videos`, `alpha`, `WINDOW_SIZE`, `camera_from`, `_frame_hw` and `_export_frames`
#  are already defined. It trains nothing and predicts nothing: it takes about a
#  second.
#
#  WHY IT IS NEEDED AT ALL
#  The exported payload carries, per origin, only the LAST observed frame (`ctx`).
#  The network needs the whole W=32 window. So a server that predicts at origins the
#  export did not choose has to be handed the tracked trajectory itself. That is one
#  float32 array of (T, 512, 3) -- a few megabytes, not the 261 MB checkpoint and not
#  Module 4's full workspace.
#
#  WHAT IT DELIBERATELY DOES NOT CARRY
#  No predictions. The point of the live server is that the prediction happens on the
#  laptop, in front of the panel. Pre-computing any of it here would make the live
#  mode theatre, which is worse than not having it.
#
#  It also carries `split_frame`, so the server can refuse origins that contributed
#  gradient steps. A panel member scrubbing freely must not be able to land on a
#  training frame and see the model at its most flattering for the one reason that
#  proves nothing.
# ════════════════════════════════════════════════════════════════════════════════
import json
import os

import numpy as np

OUT_DIR = globals().get('DEMO_DIR', '/content')
EXPORT_FRAMES = True       # also write the stills for the whole held-out range, so a
                           # live origin outside the exported 80 still has an image to
                           # draw the overlay on. Geometry works without them.

_required = ['videos', 'WINDOW_SIZE', 'camera_from', '_frame_hw']
_absent = [n for n in _required if n not in globals()]
if _absent:
    raise RuntimeError(
        f'{_absent} not defined. Run this cell after the Module 9 demo-export cell in '
        f'the same runtime -- it reuses that cell\'s helpers rather than copying them.')

print('=' * 60)
print('  LIVE BUNDLE — context windows for the local inference server')
print('=' * 60)

# The same video Module 9 chose, by the same rule, so the live server and the exported
# payload describe one sequence. Choosing differently here would put the page's static
# numbers and its live numbers on two different videos.
_vid = max(videos, key=lambda v: v['n_frames'] - int(v['split_frame']))
_name = str(_vid.get('video_id') or _vid.get('workspace') or 'video')
_horizons = list(globals().get('DEMO_HORIZONS', [1, 3]))

_tn = _vid['traj_n']
_traj = (_tn.detach().cpu().numpy() if hasattr(_tn, 'detach') else np.asarray(_tn))
_traj = np.ascontiguousarray(_traj, dtype=np.float32)

_dn = _vid['disp_norm']
_dn = float(np.asarray(_dn.detach().cpu() if hasattr(_dn, 'detach') else _dn).reshape(-1)[0])

_cam_path = f'{DRIVE_BASE}/{_vid["workspace"]}/camera_meta.json'
_meta = None
if os.path.exists(_cam_path):
    with open(_cam_path) as fh:
        _meta = json.load(fh)
_cam = camera_from(_meta, _frame_hw(_vid['workspace']))

# The calibration Module 6 fitted on TRAINING origins. Carried, not refitted: refitting
# at serve time on the origins a panel happens to visit would fit it to the held-out
# data it is meant to be evaluated on.
_cal_on = bool(globals().get('_cal_on', 'alpha' in globals()))
_alpha = {k: (float(globals().get('alpha', {}).get(k, 1.0)) if _cal_on else 1.0)
          for k in _horizons}

_payload = {
    'traj_n': _traj,
    'disp_norm': np.float32(_dn),
    'scl': np.asarray(_vid['scl'], dtype=np.float32),
    'ctr': np.asarray(_vid['ctr'], dtype=np.float32),
    'split_frame': np.int64(int(_vid['split_frame'])),
    'window': np.int64(int(WINDOW_SIZE)),
    'horizons': np.asarray(_horizons, dtype=np.int64),
    'video': np.asarray(_name),
    'workspace': np.asarray(str(_vid['workspace'])),
    'camera': np.asarray(json.dumps(_cam)),
    'calibrated': np.bool_(_cal_on),
}
for k in _horizons:
    _payload[f'alpha_{int(k)}'] = np.float32(_alpha[k])

_out = f'{OUT_DIR}/live_bundle.npz'
np.savez_compressed(_out, **_payload)

_lo = max(int(WINDOW_SIZE), int(_vid['split_frame']))
_hi = _traj.shape[0] - max(_horizons)
print(f'  video            : {_name}  ({_traj.shape[0]} tracked frames)')
print(f'  trajectory       : {_traj.shape}  float32')
print(f'  held-out origins : {_lo}..{_hi - 1}   ({_hi - _lo} the server will serve)')
print(f'  disp_norm        : {_dn:.6f}')
print(f'  calibration      : ' + ('  '.join(f'k{k}={_alpha[k]:.4f}' for k in _horizons)
                                  if _cal_on else 'none (alpha = 1)'))
print(f'  camera           : {"measured" if not _cam.get("calibrated", False) is False else _cam.get("source", "?")}'
      f"  ({_cam.get('source', '?')})")
print(f'\n  {_out}  ({os.path.getsize(_out) / 1e6:.1f} MB)')

if EXPORT_FRAMES and '_export_frames' in globals():
    _idx = sorted({i for t in range(_lo, _hi)
                   for i in [t - 1] + [t + k - 1 for k in _horizons]}
                  & set(range(_traj.shape[0])))
    _written = _export_frames(_vid['workspace'], _idx)
    print(f'  stills           : {len(_written)} frames for the full live range')

print('\n  Copy live_bundle.npz into serve/ beside server.py, then on the laptop:')
print('      python3 serve/server.py --warm')
print('  and open http://localhost:8765')
print('\n  live_bundle.npz holds a patient video\'s tracked geometry. It is gitignored')
print('  for the same reason frames/ is.')
