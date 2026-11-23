#!/usr/bin/env python3
"""The local inference server for the clinical demonstrator.

    python3 serve/server.py          then open http://localhost:8765

WHY IT SERVES THE PAGE TOO
--------------------------
The demonstrator's hard constraint is that it opens by double-click from `file://`
with no server at all. That still holds and is still the fallback. But a page on the
`file://` origin has an opaque origin, and a cross-origin request from it to
http://localhost runs into CORS and, in current Chrome, the local-network access
prompt -- a dialog nobody wants to meet in a viva.

So the server serves `demo/clinical/` itself. Opened through the server the page is
SAME-ORIGIN with the API: no CORS header, no preflight, no permission prompt.
Double-clicked, it is the same file with no server behind it and runs exactly as it
does today. One artifact, two ways in, and the live path cannot break the static one.

NO DEPENDENCIES
---------------
stdlib `http.server` only -- no FastAPI, no uvicorn, no pip install. The demonstrator
exists so that nothing has to work on the day except Python and a browser, and a
framework that must be installed is one more thing that can be missing at 9 a.m.

BOUND TO LOCALHOST
------------------
127.0.0.1 by design. This process holds a 261 MB research checkpoint and a patient
video's tracked geometry; it has no business listening on a university network.
"""
import argparse
import ast
import json
import os
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import DDIM_SAMPLES, DDIM_STEPS, Engine  # noqa: E402
from gasr_arch import REPO_ROOT  # noqa: E402

PAGE_DIR = os.path.join(REPO_ROOT, 'demo', 'clinical')

# The overlay's image coordinates and the display point subset are lifted from the
# exporter rather than rewritten, for the same reason the model is: the live overlay
# and the exported overlay must be the same projection or they will disagree on screen.
_EXPORT_SRC = os.path.join(REPO_ROOT, 'module9_demo_export.py')


def _lift(names, path=_EXPORT_SRC):
    with open(path) as fh:
        text = fh.read()
    lines = text.splitlines(keepends=True)
    ns = {'np': np, 'numpy': np, 'math': __import__('math'), 'json': json, 'os': os}
    found = set()
    for node in ast.parse(text).body:
        name = getattr(node, 'name', None)
        if name in names and isinstance(node, ast.FunctionDef):
            exec(''.join(lines[node.lineno - 1:node.end_lineno]), ns)
            found.add(name)
    missing = set(names) - found
    if missing:
        raise RuntimeError(f'{os.path.basename(path)} no longer defines {sorted(missing)}')
    return ns


def _lift_int(name, path=_EXPORT_SRC):
    """A module-level integer constant from the exporter.

    DEMO_POINTS is how many of the 512 anchors the page draws. It belongs to the
    exporter, and a second copy of it here would mean the live arrays and the exported
    arrays could be different lengths -- read off the end of the payload's triples,
    which is wrong numbers rather than an error.
    """
    with open(path) as fh:
        for node in ast.parse(fh.read()).body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name) and tgt.id == name:
                        return int(node.value.value)
    raise RuntimeError(f'{os.path.basename(path)} no longer defines {name}')


_EXPORT = _lift({'project_norm', '_subsample'})
project_norm = _EXPORT['project_norm']
subsample = _EXPORT['_subsample']
DEMO_POINTS = _lift_int('DEMO_POINTS')


class Live:
    """The engine plus the one lock that keeps it honest.

    Inference is single-threaded by choice: torch already uses every core for one
    request, so two concurrent predictions would halve each other's speed and double
    the memory. Requests queue.
    """

    def __init__(self, engine, demo_points=DEMO_POINTS):
        self.e = engine
        self.lock = threading.Lock()
        self.demo_points = demo_points
        self.pts = None
        if engine.ready:
            self.pts = subsample(engine.n_anchor, demo_points)

    def health(self):
        e = self.e
        info = {
            'ok': True,
            'live': bool(e.ready),
            'checkpoint_epoch': e.epoch,
            'params': e.n_params,
            'window': e.window,
            'n_anchor': e.n_anchor,
            'target_mode': e.target_mode,
            'steps': DDIM_STEPS,
            'samples': DDIM_SAMPLES,
            'threads': torch.get_num_threads(),
        }
        if e.ready:
            b = e.bundle
            lo, hi = e.origin_bounds()
            info.update({
                'video': str(np.asarray(b.get('video', '?')).reshape(-1)[0]),
                'n_frames_total': int(b['traj_n'].shape[0]),
                'split_frame': int(np.asarray(b.get('split_frame', 0)).reshape(-1)[0]),
                'origin_min': int(lo),
                'origin_max': int(hi - 1),
                'n_points': int(len(self.pts)),
                'horizons': e.horizons,
                'calibration': {str(k): e.alpha_for(k) for k in e.horizons},
            })
        return info

    def predict(self, origin, k, hypotheses=0):
        with self.lock:
            r = self.e.predict_origin(origin, k, hypotheses=hypotheses)
        pts = self.pts
        cam = self.e.bundle.get('camera')
        cam = json.loads(str(np.asarray(cam).reshape(-1)[0])) if cam is not None else None

        def ser(t):
            if t is None:
                return None
            a = t.detach().cpu().numpy()[pts]
            return [round(float(v), 7) for v in a.reshape(-1)]

        def px(t):
            # Normalised -> camera space -> image coordinates, exactly as the exporter
            # does it: project_norm(to_world(x), cam). Skipping to_world would place
            # every dot wrongly while still drawing a convincing overlay.
            if t is None or cam is None:
                return None
            a = t.detach().cpu().numpy()[pts]
            return [round(float(v), 7)
                    for v in project_norm(self.e.to_world(a), cam).reshape(-1)]

        out = {
            'origin': r['origin'], 'k': r['k'], 'alpha': r['alpha'],
            'seconds': round(r['seconds'], 2),
            'steps': r['steps'], 'samples': r['samples'],
            'ctx': ser(r['ctx']), 'pred': ser(r['pred']),
            'cv': ser(r['cv']), 'gt': ser(r['gt']),
            'px_ctx': px(r['ctx']), 'px_pred': px(r['pred']),
            'px_cv': px(r['cv']), 'px_gt': px(r['gt']),
        }
        # The still for this origin, but only if the file is actually beside the page.
        # Module 9 names them frames/NNNNN.jpg; a live origin outside the exported set
        # has one only if export_live_bundle.py wrote the whole range. The page draws
        # geometry alone when this is null rather than showing a broken image.
        img_idx = r['origin'] - 1
        rel = f'frames/{img_idx:05d}.jpg'
        out['img'] = int(img_idx)
        out['img_src'] = rel if os.path.exists(os.path.join(PAGE_DIR, rel)) else None

        if 'sigma' in r:
            s = r['sigma'].detach().cpu().numpy()[pts]
            out['sigma'] = [round(float(v), 9) for v in s.reshape(-1)]
        # Errors computed here, from these arrays, so the page never has to be told
        # who won -- it is handed both numbers and says what they are.
        if r['gt'] is not None:
            def mpe(a, b):
                d = (a - b).detach().cpu().numpy()
                return float(np.linalg.norm(d, axis=-1).mean())
            out['mean_error'] = {
                'ours': mpe(r['pred'], r['gt']),
                'cv': mpe(r['cv'], r['gt']),
                'still': mpe(r['ctx'].unsqueeze(0)[0], r['gt']),
            }
        return out


class Handler(SimpleHTTPRequestHandler):
    live = None

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=PAGE_DIR, **kw)

    def log_message(self, fmt, *args):
        if '/api/' in (self.path or ''):
            sys.stderr.write('  %s\n' % (fmt % args))

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.split('?')[0] == '/api/health':
            return self._json(self.live.health())
        return super().do_GET()

    def do_POST(self):
        if self.path.split('?')[0] != '/api/predict':
            return self._json({'error': 'no such endpoint'}, 404)
        if not self.live.e.ready:
            return self._json({'error': 'no live bundle loaded; see '
                                        'serve/export_live_bundle.py'}, 503)
        try:
            n = int(self.headers.get('Content-Length') or 0)
            req = json.loads(self.rfile.read(n) or b'{}')
            origin = int(req['origin'])
            k = int(req['k'])
            hyp = int(req.get('hypotheses', 0))
        except Exception as exc:
            return self._json({'error': f'bad request: {exc}'}, 400)
        try:
            return self._json(self.live.predict(origin, k, hypotheses=hyp))
        except (IndexError, RuntimeError) as exc:
            return self._json({'error': str(exc)}, 400)
        except Exception as exc:                     # noqa: BLE001
            sys.stderr.write(f'  predict failed: {exc!r}\n')
            return self._json({'error': 'inference failed; the page should fall back '
                                        'to the exported data'}, 500)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--bundle', default=None)
    ap.add_argument('--threads', type=int, default=None)
    ap.add_argument('--warm', action='store_true',
                    help='run one prediction at startup so the first click is not the '
                         'one that pays for lazily allocated kernels')
    args = ap.parse_args(argv)

    kw = {}
    if args.ckpt:
        kw['ckpt_path'] = args.ckpt
    if args.bundle:
        kw['bundle_path'] = args.bundle
    print('loading checkpoint ...', flush=True)
    eng = Engine(threads=args.threads, **kw)
    live = Live(eng)
    h = live.health()
    print(f"  epoch {h['checkpoint_epoch']}  {h['params']:,} parameters  "
          f"W={h['window']}  N={h['n_anchor']}  {h['threads']} threads")
    if eng.ready:
        print(f"  live bundle: video {h['video']}  frames 0..{h['origin_max']}  "
              f"origins {h['origin_min']}..{h['origin_max']}")
        if args.warm:
            print('  warming ...', flush=True)
            r = live.predict(h['origin_min'], 1)
            print(f"  first prediction took {r['seconds']} s")
    else:
        print('  NO LIVE BUNDLE -- the page will serve the exported payload only.')
        print('  Run serve/export_live_bundle.py in Colab to produce serve/live_bundle.npz')

    Handler.live = live
    try:
        srv = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    except OSError as exc:
        # Almost always this server already running from earlier. A traceback here,
        # five minutes before a viva, reads as "the demo is broken". It is not.
        print(f'\n  cannot listen on port {args.port}: {exc}')
        print(f'  Something already has it -- most likely this server, already up.')
        print(f'  Open  http://localhost:{args.port}  first, or use '
              f'--port {args.port + 1}.')
        return 1
    print(f'\n  open  http://localhost:{args.port}\n  ctrl-c to stop\n', flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print('\nstopped')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
