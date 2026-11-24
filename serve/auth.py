"""Doctor sign-in for the local inference server.

WHAT THIS PROTECTS, AND WHAT IT DOES NOT
----------------------------------------
It gates live inference: `/api/predict`, and the bundle metadata in `/api/health`.
Those are the things a running server exposes that are not already in the examiner's
hands -- the 261 MB checkpoint, and the ability to run it against a patient video's
tracked geometry at any origin.

It does NOT protect the exported demonstrator. `demo/clinical/index.html` and
`demo_data.js` are handed out on a USB stick and open by double-click with no server
at all, which is the whole point of them. Putting a password in front of the same
bytes on localhost would protect nothing and would imply a confidentiality this
project has never claimed. The page's sign-in screen is therefore a gate on the live
model, not on the exported measurement, and the README says so in those words.

NO DEPENDENCIES
---------------
stdlib only, like the rest of the server. PBKDF2-HMAC-SHA256 from hashlib rather than
bcrypt or argon2, because both would have to be installed and the demonstrator's rule
is that nothing has to work on the day except Python and a browser. PBKDF2 at this
iteration count is a sound choice, just not the fashionable one.
"""
import hashlib
import hmac
import json
import os
import secrets
import threading
import time

# OWASP's floor for PBKDF2-HMAC-SHA256 is 600,000 iterations. Measured on the target
# laptop this costs roughly a quarter of a second per attempt, which is unnoticeable
# on a login and ruinous on a dictionary.
ITERATIONS = 600_000
SALT_BYTES = 16
SESSION_HOURS = 12

# A sign-in that can be retried without limit is a password of whatever length the
# attacker has patience for. Five wrong answers buys a minute of silence, per account.
MAX_FAILURES = 5
LOCKOUT_SECONDS = 60

DEFAULT_STORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'doctors.json')
SEED_ROLL = 'demo-doctor'


def hash_password(password, salt=None, iterations=ITERATIONS):
    """Return the stored record for a password. The password itself is never kept."""
    if not isinstance(password, str) or not password:
        raise ValueError('a password is required')
    salt = secrets.token_bytes(SALT_BYTES) if salt is None else salt
    digest = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, iterations)
    return {
        'algorithm': 'pbkdf2_sha256',
        'iterations': iterations,
        'salt': salt.hex(),
        'hash': digest.hex(),
    }


def verify_password(password, record):
    """Constant-time check of a password against a stored record.

    hmac.compare_digest rather than `==`: a byte-by-byte comparison returns sooner for
    a wrong first byte than a wrong last one, and that difference is measurable over
    enough requests. It costs nothing to not leak it.
    """
    if not isinstance(password, str) or not record:
        return False
    try:
        salt = bytes.fromhex(record['salt'])
        want = bytes.fromhex(record['hash'])
        iterations = int(record.get('iterations', ITERATIONS))
    except (KeyError, ValueError, TypeError):
        return False
    got = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, iterations)
    return hmac.compare_digest(got, want)


def new_password(words=3):
    """A readable password that can be typed off a screen by someone under pressure."""
    alphabet = 'abcdefghijkmnopqrstuvwxyz23456789'     # no l/1/0/o
    return '-'.join(''.join(secrets.choice(alphabet) for _ in range(4))
                    for _ in range(words))


class Accounts:
    """Doctor accounts, in a JSON file beside the server.

    The file holds roll numbers and password hashes. It is NOT in the repository: a
    committed credential is a credential that has leaked, even a demo one, and a hash
    in a public repo is a hash somebody can grind offline at their leisure.
    """

    def __init__(self, path=DEFAULT_STORE):
        self.path = path
        self._lock = threading.Lock()
        self._fail = {}                 # roll -> [count, locked_until]
        self.data = {'doctors': {}}
        self.seeded = None              # (roll, password) when this run created it
        self._load()

    # ── storage ───────────────────────────────────────────────────────────────
    def _load(self):
        if os.path.exists(self.path):
            with open(self.path) as fh:
                self.data = json.load(fh)
            self.data.setdefault('doctors', {})
            return
        # First run: seed one account so the server is usable immediately, and hand
        # the password to whoever started it. A seeded ACCOUNT with a seeded PASSWORD
        # would be no account at all, so the password is random and printed once.
        password = new_password()
        self.data = {'doctors': {SEED_ROLL: {
            'name': 'Demonstration doctor',
            'password': hash_password(password),
            'created': int(time.time()),
        }}}
        self._save()
        self.seeded = (SEED_ROLL, password)

    def _save(self):
        tmp = self.path + '.tmp'
        with open(tmp, 'w') as fh:
            json.dump(self.data, fh, indent=2)
            fh.write('\n')
        os.replace(tmp, self.path)
        try:
            os.chmod(self.path, 0o600)   # hashes, so: not world-readable
        except OSError:
            pass

    # ── accounts ──────────────────────────────────────────────────────────────
    def rolls(self):
        return sorted(self.data['doctors'])

    def add(self, roll, password=None, name=None):
        roll = (roll or '').strip()
        if not roll:
            raise ValueError('a roll number is required')
        password = password or new_password()
        with self._lock:
            self.data['doctors'][roll] = {
                'name': name or roll,
                'password': hash_password(password),
                'created': int(time.time()),
            }
            self._save()
        return password

    def set_password(self, roll, password=None):
        roll = (roll or '').strip()
        if roll not in self.data['doctors']:
            raise KeyError(roll)
        password = password or new_password()
        with self._lock:
            self.data['doctors'][roll]['password'] = hash_password(password)
            self._save()
            self._fail.pop(roll, None)
        return password

    def name_of(self, roll):
        return (self.data['doctors'].get(roll) or {}).get('name') or roll

    # ── sign-in ───────────────────────────────────────────────────────────────
    def locked_for(self, roll):
        """Seconds remaining on a lockout, or 0."""
        with self._lock:
            entry = self._fail.get(roll)
            if not entry:
                return 0
            return max(0, int(entry[1] - time.time()))

    def check(self, roll, password):
        """(ok, message). The message never says which half was wrong."""
        roll = (roll or '').strip()
        wait = self.locked_for(roll)
        if wait:
            return False, (f'too many failed attempts; try again in {wait} s')

        record = (self.data['doctors'].get(roll) or {}).get('password')
        # Verify even when the roll number is unknown, against a throwaway hash of the
        # same cost. Otherwise an unknown roll returns in microseconds and a known one
        # in a quarter of a second, which tells an attacker who exists.
        if record is None:
            verify_password(password or '', hash_password('x'))
            ok = False
        else:
            ok = verify_password(password or '', record)

        with self._lock:
            if ok:
                self._fail.pop(roll, None)
            else:
                count, _ = self._fail.get(roll, (0, 0))
                count += 1
                until = time.time() + LOCKOUT_SECONDS if count >= MAX_FAILURES else 0
                self._fail[roll] = [0 if until else count, until]
        if ok:
            return True, ''
        return False, 'that roll number and password do not match'


class Sessions:
    """In-memory sessions. Restarting the server signs everyone out, which is correct:
    nothing here is worth persisting across a process that holds a checkpoint in RAM."""

    def __init__(self, hours=SESSION_HOURS):
        self.ttl = hours * 3600
        self._lock = threading.Lock()
        self._by_token = {}

    def new(self, roll):
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._by_token[token] = (roll, time.time() + self.ttl)
        return token

    def get(self, token):
        if not token:
            return None
        with self._lock:
            entry = self._by_token.get(token)
            if not entry:
                return None
            roll, expires = entry
            if time.time() > expires:
                del self._by_token[token]
                return None
            return roll

    def drop(self, token):
        with self._lock:
            self._by_token.pop(token, None)

    def count(self):
        with self._lock:
            return len(self._by_token)
