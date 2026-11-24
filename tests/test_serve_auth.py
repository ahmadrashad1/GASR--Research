"""Doctor sign-in on the local inference server.

These run the real HTTP handler against a stub engine, so they need no checkpoint and
no bundle: what is under test is the gate, not the model.

The property that matters is the negative one. A page that has not signed in must not
be able to get a prediction out of the checkpoint, and must not be told which video
the bundle holds or where its held-out frames start -- and the refusal must not reveal
whether a roll number exists.
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'serve'))

import auth  # noqa: E402

# Hashing at 600k iterations costs ~0.25 s, which is the point of it; the tests would
# spend minutes on that for no benefit, so they use a low count where the cost is not
# what is being checked.
FAST = 1000


class StubEngine:
    ready = True


class StubLive:
    """Stands in for Live. Returns the shape the handler expects and nothing real."""

    def __init__(self):
        self.e = StubEngine()
        self.calls = []

    def health(self):
        return {'ok': True, 'live': True, 'checkpoint_epoch': 100, 'params': 16318467,
                'window': 32, 'n_anchor': 512, 'target_mode': 'cv_residual',
                'steps': 20, 'samples': 4, 'threads': 1, 'video': '3',
                'n_frames_total': 320, 'split_frame': 240, 'origin_min': 240,
                'origin_max': 316, 'n_points': 256, 'horizons': [1, 3],
                'calibration': {'1': 0.92, '3': 1.02}}

    def predict(self, origin, k, hypotheses=0):
        self.calls.append((origin, k, hypotheses))
        return {'origin': origin, 'k': k, 'seconds': 0.0}


class ServerFixture:
    """A real ThreadingHTTPServer on a loopback port, with the real Handler."""

    def __init__(self, accounts, store_path=None):
        import server as srv_mod
        self.mod = srv_mod
        self.live = StubLive()
        srv_mod.Handler.live = self.live
        srv_mod.Handler.accounts = accounts
        srv_mod.Handler.sessions = auth.Sessions()
        self.srv = ThreadingHTTPServer(('127.0.0.1', 0), srv_mod.Handler)
        self.port = self.srv.server_address[1]
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.thread.start()

    def url(self, path):
        return f'http://127.0.0.1:{self.port}{path}'

    def request(self, path, data=None, cookie=None, method=None):
        body = json.dumps(data).encode() if data is not None else None
        req = urllib.request.Request(self.url(path), data=body,
                                     method=method or ('POST' if body else 'GET'))
        if body:
            req.add_header('Content-Type', 'application/json')
        if cookie:
            req.add_header('Cookie', cookie)
        try:
            with urllib.request.urlopen(req, timeout=10) as res:
                return res.status, json.loads(res.read() or b'{}'), res.headers
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read() or b'{}'), exc.headers

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


def accounts_in(tmpdir, roll='demo-doctor', password='correct-horse'):
    path = os.path.join(tmpdir, 'doctors.json')
    acc = auth.Accounts(path)
    acc.data['doctors'] = {}
    acc.add(roll, password)
    return acc, path


class TestHashing(unittest.TestCase):
    def test_the_password_is_never_stored(self):
        record = auth.hash_password('correct-horse', iterations=FAST)
        blob = json.dumps(record)
        self.assertNotIn('correct-horse', blob,
                         'the password itself appears in the stored record')
        self.assertEqual(record['algorithm'], 'pbkdf2_sha256')

    def test_the_same_password_hashes_differently_each_time(self):
        # No salt means identical passwords share a hash, which tells an attacker who
        # reads the file which doctors chose the same one.
        a = auth.hash_password('same', iterations=FAST)
        b = auth.hash_password('same', iterations=FAST)
        self.assertNotEqual(a['salt'], b['salt'])
        self.assertNotEqual(a['hash'], b['hash'])
        self.assertTrue(auth.verify_password('same', a))
        self.assertTrue(auth.verify_password('same', b))

    def test_wrong_password_is_rejected(self):
        record = auth.hash_password('correct-horse', iterations=FAST)
        self.assertFalse(auth.verify_password('Correct-horse', record))
        self.assertFalse(auth.verify_password('', record))
        self.assertFalse(auth.verify_password('correct-horse ', record))

    def test_a_corrupt_record_fails_closed(self):
        # A truncated or hand-edited doctors.json must refuse everyone, not admit them.
        for bad in ({}, {'salt': 'zz', 'hash': 'zz'}, {'salt': 'ab'}, None):
            self.assertFalse(auth.verify_password('anything', bad), bad)

    def test_the_iteration_count_is_not_quietly_weak(self):
        # The cost IS the protection. If someone lowers this to make tests faster, the
        # stored hashes become cheap to grind, and nothing else would notice.
        self.assertGreaterEqual(auth.ITERATIONS, 600_000)


class TestAccounts(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()
        self.saved = auth.ITERATIONS
        auth.ITERATIONS = FAST

    def tearDown(self):
        auth.ITERATIONS = self.saved

    def test_first_run_seeds_an_account_with_a_random_password(self):
        path = os.path.join(self.tmp, 'doctors.json')
        a = auth.Accounts(path)
        self.assertIsNotNone(a.seeded, 'first run should seed an account')
        roll, password = a.seeded
        self.assertTrue(auth.Accounts(path).check(roll, password)[0])
        # A seeded account with a seeded password is not an account.
        b = auth.Accounts(os.path.join(self.tmp, 'other.json'))
        self.assertNotEqual(password, b.seeded[1],
                            'the seeded password must not be a constant')

    def test_the_store_is_not_world_readable(self):
        path = os.path.join(self.tmp, 'doctors.json')
        auth.Accounts(path)
        self.assertEqual(os.stat(path).st_mode & 0o077, 0,
                         'doctors.json is readable by other users')

    def test_an_unknown_roll_is_refused_exactly_like_a_wrong_password(self):
        # Different wording for "no such doctor" and "wrong password" hands an attacker
        # a list of valid roll numbers for free.
        acc, _ = accounts_in(self.tmp, 'doc-1', 'right')
        _, unknown = acc.check('nobody-at-all', 'right')
        _, wrong = acc.check('doc-1', 'wrong')
        self.assertEqual(unknown, wrong)

    def test_repeated_failures_lock_the_account(self):
        acc, _ = accounts_in(self.tmp, 'doc-1', 'right')
        for _ in range(auth.MAX_FAILURES):
            self.assertFalse(acc.check('doc-1', 'wrong')[0])
        ok, why = acc.check('doc-1', 'right')
        self.assertFalse(ok, 'the correct password still worked while locked out')
        self.assertIn('too many failed attempts', why)
        self.assertGreater(acc.locked_for('doc-1'), 0)

    def test_resetting_a_password_clears_the_lockout(self):
        acc, _ = accounts_in(self.tmp, 'doc-1', 'right')
        for _ in range(auth.MAX_FAILURES):
            acc.check('doc-1', 'wrong')
        fresh = acc.set_password('doc-1')
        self.assertEqual(acc.locked_for('doc-1'), 0)
        self.assertTrue(acc.check('doc-1', fresh)[0])

    def test_reloading_the_file_keeps_the_accounts(self):
        acc, path = accounts_in(self.tmp, 'doc-1', 'right')
        again = auth.Accounts(path)
        self.assertIsNone(again.seeded, 'an existing store must not be re-seeded')
        self.assertTrue(again.check('doc-1', 'right')[0])


class TestSessions(unittest.TestCase):
    def test_tokens_are_unguessable_and_distinct(self):
        s = auth.Sessions()
        tokens = {s.new('doc-1') for _ in range(50)}
        self.assertEqual(len(tokens), 50)
        self.assertTrue(all(len(t) >= 32 for t in tokens))

    def test_a_token_resolves_then_can_be_dropped(self):
        s = auth.Sessions()
        t = s.new('doc-1')
        self.assertEqual(s.get(t), 'doc-1')
        s.drop(t)
        self.assertIsNone(s.get(t))

    def test_an_expired_token_is_refused(self):
        s = auth.Sessions(hours=0)
        t = s.new('doc-1')
        time.sleep(0.01)
        self.assertIsNone(s.get(t))

    def test_an_unknown_token_is_refused(self):
        s = auth.Sessions()
        for junk in (None, '', 'not-a-token', 'x' * 64):
            self.assertIsNone(s.get(junk), junk)


class TestTheGate(unittest.TestCase):
    """The HTTP layer, which is what actually stands between a browser and the model."""

    @classmethod
    def setUpClass(cls):
        import tempfile
        cls.saved = auth.ITERATIONS
        auth.ITERATIONS = FAST
        cls.tmp = tempfile.mkdtemp()
        cls.acc, _ = accounts_in(cls.tmp, 'doc-1', 'right-password')
        cls.fx = ServerFixture(cls.acc)

    @classmethod
    def tearDownClass(cls):
        cls.fx.close()
        auth.ITERATIONS = cls.saved

    def sign_in(self, roll='doc-1', password='right-password'):
        status, body, headers = self.fx.request(
            '/api/login', {'roll': roll, 'password': password})
        cookie = headers.get('Set-Cookie') or ''
        token = cookie.split(';')[0] if cookie else None
        return status, body, token

    def test_health_says_almost_nothing_until_a_doctor_signs_in(self):
        status, body, _ = self.fx.request('/api/health')
        self.assertEqual(status, 200)
        self.assertIs(body['live'], False)
        self.assertIs(body['auth_required'], True)
        self.assertIs(body['signed_in'], False)
        # The bundle metadata is behind the gate: which patient video, and which
        # frames the model never saw, are not for an unauthenticated caller.
        for leak in ('video', 'split_frame', 'origin_min', 'origin_max',
                     'checkpoint_epoch', 'n_frames_total'):
            self.assertNotIn(leak, body, f'health leaked {leak} before sign-in')

    def test_predict_is_refused_without_a_session(self):
        before = len(self.fx.live.calls)
        status, body, _ = self.fx.request('/api/predict', {'origin': 250, 'k': 1})
        self.assertEqual(status, 401)
        self.assertIn('sign in', body['error'].lower())
        self.assertEqual(len(self.fx.live.calls), before,
                         'the model ran for a caller that had not signed in')

    def test_a_wrong_password_is_refused_and_sets_no_cookie(self):
        status, body, token = self.sign_in(password='wrong')
        self.assertEqual(status, 401)
        self.assertIsNone(token, 'a failed sign-in handed out a session cookie')
        self.assertNotIn('right-password', json.dumps(body))

    def test_signing_in_opens_the_gate(self):
        status, _, token = self.sign_in()
        self.assertEqual(status, 200)
        self.assertIsNotNone(token)

        status, body, _ = self.fx.request('/api/health', cookie=token)
        self.assertIs(body['live'], True)
        self.assertIs(body['signed_in'], True)
        self.assertEqual(body['video'], '3')

        before = len(self.fx.live.calls)
        status, body, _ = self.fx.request('/api/predict', {'origin': 250, 'k': 1},
                                          cookie=token)
        self.assertEqual(status, 200)
        self.assertEqual(len(self.fx.live.calls), before + 1)

    def test_the_session_cookie_is_httponly_and_samesite(self):
        _, _, headers = self.fx.request(
            '/api/login', {'roll': 'doc-1', 'password': 'right-password'})
        raw = headers.get('Set-Cookie') or ''
        self.assertIn('HttpOnly', raw, 'page scripts can read the session token')
        self.assertIn('SameSite=Strict', raw, 'another site could ride the session')
        self.assertIn('Path=/', raw)

    def test_a_forged_cookie_does_not_work(self):
        status, _, _ = self.fx.request('/api/predict', {'origin': 250, 'k': 1},
                                       cookie='gasr_session=' + 'a' * 43)
        self.assertEqual(status, 401)

    def test_signing_out_closes_the_gate_again(self):
        _, _, token = self.sign_in()
        status, _, _ = self.fx.request('/api/logout', {}, cookie=token)
        self.assertEqual(status, 200)
        status, _, _ = self.fx.request('/api/predict', {'origin': 250, 'k': 1},
                                       cookie=token)
        self.assertEqual(status, 401, 'the session survived signing out')

    def test_the_page_itself_is_served_without_a_sign_in(self):
        # Deliberate. The same bytes open by double-click from a USB stick with no
        # server at all, so a password in front of them on localhost would protect
        # nothing and imply a confidentiality this project does not claim.
        req = urllib.request.Request(self.fx.url('/index.html'))
        with urllib.request.urlopen(req, timeout=10) as res:
            self.assertEqual(res.status, 200)


class TestNoAuthEscapeHatch(unittest.TestCase):
    """--no-auth exists so a demonstration cannot be killed by a forgotten password."""

    def test_without_an_account_store_the_model_runs(self):
        fx = ServerFixture(accounts=None)
        try:
            status, body, _ = fx.request('/api/health')
            self.assertIs(body['live'], True)
            self.assertIs(body['auth_required'], False)
            status, _, _ = fx.request('/api/predict', {'origin': 250, 'k': 1})
            self.assertEqual(status, 200)
        finally:
            fx.close()


class TestNothingSecretIsCommitted(unittest.TestCase):
    def test_the_account_store_is_gitignored(self):
        out = subprocess.run(['git', 'check-ignore', 'serve/doctors.json'],
                             cwd=REPO, capture_output=True, text=True)
        self.assertEqual(out.returncode, 0,
                         'serve/doctors.json is not gitignored; a committed hash is a '
                         'hash somebody can grind offline at their leisure')

    def test_no_password_hash_is_tracked_in_the_repo(self):
        out = subprocess.run(['git', 'ls-files'], cwd=REPO,
                             capture_output=True, text=True)
        self.assertNotIn('doctors.json', out.stdout)

    def test_the_built_page_carries_no_credential(self):
        page = os.path.join(REPO, 'demo', 'clinical', 'index.html')
        with open(page, encoding='utf-8') as fh:
            html = fh.read()
        self.assertNotIn('pbkdf2', html.lower())
        # A client-side password check would mean the gate is theatre. The page may
        # only ever POST to /api/login and let the server decide.
        self.assertNotRegex(html, re.compile(r'password\s*===\s*["\'`]'))
        self.assertIn('/api/login', html)


if __name__ == '__main__':
    unittest.main(verbosity=2)
