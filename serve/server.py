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
import auth  # noqa: E402
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


COOKIE = 'gasr_session'


class Handler(SimpleHTTPRequestHandler):
    live = None
    accounts = None
    sessions = None

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=PAGE_DIR, **kw)

    def log_message(self, fmt, *args):
        if '/api/' in (self.path or ''):
            sys.stderr.write('  %s\n' % (fmt % args))

    def _json(self, obj, code=200, cookie=None, clear_cookie=False):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        if cookie:
            # HttpOnly so page scripts cannot read the token; SameSite=Strict so another
            # site cannot ride the session. No Secure flag: this is http on loopback, and
            # setting it would stop the cookie being stored at all.
            self.send_header('Set-Cookie',
                             f'{COOKIE}={cookie}; HttpOnly; SameSite=Strict; Path=/; '
                             f'Max-Age={auth.SESSION_HOURS * 3600}')
        if clear_cookie:
            self.send_header('Set-Cookie',
                             f'{COOKIE}=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0')
        self.end_headers()
        self.wfile.write(body)

    # ── sign-in ──────────────────────────────────────────────────────────────
    def _token(self):
        raw = self.headers.get('Cookie') or ''
        for part in raw.split(';'):
            name, _, value = part.strip().partition('=')
            if name == COOKIE:
                return value
        return None

    def _signed_in(self):
        """The roll number of the signed-in doctor, or None.

        With no account store configured the server is open, which is the documented
        way to run it when a demonstration must not be able to fail on a password.
        """
        if self.accounts is None:
            return '(no sign-in required)'
        return self.sessions.get(self._token())

    def _body(self):
        n = int(self.headers.get('Content-Length') or 0)
        return json.loads(self.rfile.read(n) or b'{}')

    def _login(self):
        try:
            req = self._body()
            roll = str(req.get('roll', ''))
            password = str(req.get('password', ''))
        except Exception as exc:
            return self._json({'error': f'bad request: {exc}'}, 400)
        if self.accounts is None:
            return self._json({'ok': True, 'roll': None,
                               'note': 'this server was started with --no-auth'})
        ok, why = self.accounts.check(roll, password)
        if not ok:
            sys.stderr.write(f'  sign-in refused for {roll!r}: {why}\n')
            return self._json({'error': why}, 401)
        token = self.sessions.new(roll)
        sys.stderr.write(f'  signed in: {roll}\n')
        return self._json({'ok': True, 'roll': roll,
                           'name': self.accounts.name_of(roll)}, cookie=token)

    def do_GET(self):
        path = self.path.split('?')[0]
        if path == '/api/health':
            who = self._signed_in()
            # Unauthenticated, health says only that a server is here and wants a
            # sign-in. The bundle metadata -- which video, which frames are held out,
            # the checkpoint's epoch -- is behind the gate with everything else.
            if who is None:
                return self._json({'ok': True, 'live': False, 'auth_required': True,
                                   'signed_in': False})
            out = dict(self.live.health())
            out['auth_required'] = self.accounts is not None
            out['signed_in'] = True
            if self.accounts is not None:
                out['doctor'] = self.accounts.name_of(who)
            return self._json(out)
        if path == '/api/session':
            who = self._signed_in()
            return self._json({'signed_in': who is not None,
                               'auth_required': self.accounts is not None,
                               'doctor': (self.accounts.name_of(who)
                                          if who and self.accounts else None)})
        # Chrome asks every origin for a favicon. There isn't one, and the 404 it gets
        # is the only error in the console on an otherwise clean run -- which is worth
        # a line of code to remove, because a panel member who opens the developer
        # console during a viva should find nothing there to ask about.
        if path == '/favicon.ico':
            self.send_response(204)
            self.send_header('Content-Length', '0')
            self.end_headers()
            return
        return super().do_GET()

    def do_POST(self):
        path = self.path.split('?')[0]
        if path == '/api/login':
            return self._login()
        if path == '/api/logout':
            self.sessions.drop(self._token())
            return self._json({'ok': True}, clear_cookie=True)
        if path != '/api/predict':
            return self._json({'error': 'no such endpoint'}, 404)
        # The gate. Running the model against a patient video's geometry is the one
        # thing this process offers that is not already on the USB stick.
        if self._signed_in() is None:
            return self._json({'error': 'sign in to run the model'}, 401)
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
    ap.add_argument('--accounts', default=None,
                    help='doctor account file (default serve/doctors.json, created '
                         'with one seeded account on first run)')
    ap.add_argument('--no-auth', action='store_true',
                    help='serve live inference without a sign-in. The escape hatch for '
                         'a demonstration that must not be able to fail on a password')
    ap.add_argument('--add-doctor', metavar='ROLL',
                    help='create a doctor account, print a generated password, exit')
    ap.add_argument('--reset-password', metavar='ROLL',
                    help='set a new generated password for a doctor, print it, exit')
    ap.add_argument('--list-doctors', action='store_true',
                    help='print the roll numbers that can sign in, and exit')
    args = ap.parse_args(argv)

    # Account administration runs without loading 261 MB of weights.
    if args.add_doctor or args.reset_password or args.list_doctors:
        acc = auth.Accounts(args.accounts or auth.DEFAULT_STORE)
        if args.list_doctors:
            if not acc.rolls():
                print('  no accounts yet')
            for roll in acc.rolls():
                print(f'  {roll}    {acc.name_of(roll)}')
            return 0
        if args.add_doctor:
            pw = acc.add(args.add_doctor)
            print(f'\n  added {args.add_doctor}\n  password: {pw}\n')
            print('  Stored hashed. This is the only time it is shown.\n')
            return 0
        try:
            pw = acc.set_password(args.reset_password)
        except KeyError:
            print(f'  no such doctor: {args.reset_password}')
            print(f'  known: {", ".join(acc.rolls()) or "(none)"}')
            return 1
        print(f'\n  new password for {args.reset_password}: {pw}\n')
        return 0

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
    if args.no_auth:
        Handler.accounts = None
        print('\n  --no-auth: live inference is open to anyone who can reach this port.')
        print('  Fine on loopback for a rehearsal; not how it is meant to run.')
    else:
        accounts = auth.Accounts(args.accounts or auth.DEFAULT_STORE)
        Handler.accounts = accounts
        Handler.sessions = auth.Sessions()
        print(f'\n  sign-in required to run the model '
              f'({len(accounts.rolls())} doctor account(s))')
        if accounts.seeded:
            roll, pw = accounts.seeded
            print(f'\n  Created {accounts.path} with one account:')
            print(f'      roll number : {roll}')
            print(f'      password    : {pw}')
            print('\n  Shown once. Stored hashed. Change it with:')
            print(f'      python3 serve/server.py --reset-password {roll}')
        else:
            print(f'  accounts: {", ".join(accounts.rolls())}')
            print('  Forgotten it? python3 serve/server.py --reset-password <roll>')
        print('  The exported page and its measurements are NOT behind this; only '
              'live\n  inference is. Double-clicking the page still needs nothing.')

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
