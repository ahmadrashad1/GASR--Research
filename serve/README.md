# Live inference on the laptop

The clinical demonstrator normally replays what Module 9 exported. This server lets it
instead run the trained model **in the room, on the machine on the table**, at any
origin in the held-out tail.

It is an addition, never a requirement. Double-clicking `demo/clinical/index.html`
still works exactly as before, with no server, no network and no install — and if this
server dies mid-viva the page falls back to the exported data and keeps going.

## Running it

```bash
python3 serve/server.py --warm      # loads the checkpoint, then open the URL it prints
```

Then open **http://localhost:8765**. The page is served by this process, so the API is
same-origin: no CORS, no browser permission prompt.

It needs two files that are deliberately not in the repository:

| File | Where from | Size |
|---|---|---|
| `diffusion_ckpt_best.pth` | Drive — the epoch-100 checkpoint | 261 MB |
| `serve/live_bundle.npz` | run `serve/export_live_bundle.py` as a Colab cell | ~3 MB |

Without the bundle the server still serves the page; it just reports `live: false` and
the page never offers live mode. Without the checkpoint it refuses to start and says
why.

`python3 serve/server.py` uses **only the Python standard library** plus torch and
numpy. There is no FastAPI, no uvicorn, nothing to `pip install`. The demonstrator
exists so that nothing has to work on the day except Python and a browser.

It binds `127.0.0.1` only. This process holds a research checkpoint and a patient
video's tracked geometry and has no business on a university network.

## Doctor sign-in

Live inference is behind a sign-in. On first start the server creates
`serve/doctors.json` with one account and prints its roll number and a generated
password **once**:

```
  Created serve/doctors.json with one account:
      roll number : demo-doctor
      password    : xxxx-xxxx-xxxx
```

```bash
python3 serve/server.py --list-doctors            # who can sign in
python3 serve/server.py --add-doctor 22i-1175     # new account, prints a password
python3 serve/server.py --reset-password demo-doctor
python3 serve/server.py --no-auth                 # no sign-in at all
```

Passwords are stored as PBKDF2-HMAC-SHA256 at 600,000 iterations with a per-account
random salt, compared with `hmac.compare_digest`, and the file is written `0600` and
gitignored — a committed hash is a hash somebody can grind offline at their leisure.
Five wrong answers lock an account for a minute. An unknown roll number is refused in
the same words, and at the same cost, as a wrong password, so the refusal cannot be
used to discover who exists. Sessions are random 32-byte tokens in memory, in an
`HttpOnly; SameSite=Strict` cookie, and they do not survive a restart.

**What it protects, and what it does not.** It gates `/api/predict` and the bundle
metadata in `/api/health`: without a session the server will not run the model, and
will not say which video it holds or where the held-out frames begin. It does **not**
protect the exported demonstrator. `index.html` and `demo_data.js` are the same bytes
that go on the USB stick and open by double-click with no server and no password, so
a gate in front of them here would protect nothing and imply a confidentiality this
project has never claimed. The sign-in screen says exactly that, and offers
**Continue to the exported demonstrator** — which is also why a forgotten password
cannot stop a demonstration.

Double-clicked, the page is on `file://`, makes no request at all, and never sees any
of this. The sign-in screen is unreachable there by construction.

## Why it is fast enough to be worth having

`ConditionalDenoiser.forward` recomputes `encode_context(ctx)` on every call, and
`ddim_sample` calls it once per diffusion step. But across the 20 steps of one
prediction `ctx` and `hstep` never change — only the noisy target and the timestep do.
So the context encoder, a pointwise MLP plus a 3-layer transformer over
512 × 32 = 16,384 tokens, was running twenty times to produce twenty identical tensors.

Measured on the target machine (Core Ultra 5 225U, 14 threads, N=512, W=32, idle):

| | |
|---|---|
| one full forward pass | 2299 ms |
| of which `encode_context` | 2206 ms — **95.9 %** |
| 20-step DDIM, re-encoding every step | **126.7 s** |
| 20-step DDIM, context encoded once | **9.3 s** |

Hoisting a pure function out of a loop over inputs it does not depend on is an
algebraic identity, not an approximation — and at `n_samples=1` the two outputs are
bit-for-bit equal, `max|a-b| = 0.0`.

**At `n_samples=4`, which is what every published number used, they differ by
7.2e-07.** Not because the hoist is wrong, but because the fast path encodes one
context row and expands it to four, while the original encodes four identical rows: a
matmul over batch 1 and over batch 4 sum in a different order, so the two float32
roundings of the same quantity differ in the last bits. `encode_full_batch=True`
encodes at the full batch and is bit-identical again; the test suite uses it to prove
the expansion is arithmetic rather than a mistake. After denormalisation that
worst-case disagreement is about 0.01 % of a single frame of tissue motion — four
orders of magnitude below anything the demonstrator says out loud.

A prediction takes about 10 s on an idle machine. Measured under load from a competing
job it took 86 s, so **close everything else before the viva.**

## What it will not do

- **Serve an origin that trained the model.** Origins are clamped to Module 9's own
  bound, `max(W, split_frame)`. Module 4 split each video 80/20 temporally, and a
  panel scrubbing freely must not be able to land on a training frame and see the model
  at its most flattering for the one reason that proves nothing. The refusal says so,
  and says something different for an origin past the end, where the video simply runs
  out before the horizon.
- **Restate the paper's numbers from one origin.** A live prediction updates this
  frame's geometry at this horizon and nothing else. The readout table keeps averaging
  the whole held-out sequence from the export, because that is what its sentences are
  about; the live origin's own errors appear separately, labelled as one origin.
- **Precompute anything.** The bundle carries the tracked trajectory and no
  predictions. If the prediction were made in Colab and replayed here, live mode would
  be theatre, which is worse than not having it.
- **Hold its own copy of the model.** `gasr_arch.py` lifts `ConditionalDenoiser` out of
  `module5_diffusion.py` as source text and executes it against the hyperparameters in
  the checkpoint's own `arch_cfg`. There is one definition of the network in this
  repository. A second one is the single bug in this design that would not announce
  itself — weights that load, a server that starts, and every number subtly not the
  paper's. `server.py` lifts the overlay projection out of `module9_demo_export.py` for
  the same reason.

The parameter count is asserted at 16,318,467 on load, which is the figure in the
paper. If that ever fails, the checkpoint is not the one the report describes.

## Files

| | |
|---|---|
| `gasr_arch.py` | extracts the network from the training script |
| `engine.py` | checkpoint, schedule, the context-cached sampler, `predict_direct`'s arithmetic |
| `server.py` | stdlib HTTP: serves the page, `/api/health`, `/api/predict`, sign-in |
| `auth.py` | doctor accounts, password hashing, sessions, lockout |
| `export_live_bundle.py` | the Colab cell that produces `live_bundle.npz` |

## Tests

```bash
python3 -m unittest tests.test_serve_engine -v              # 18 tests, ~80 s
python3 -m unittest tests.test_serve_auth -v                # 27 tests, ~11 s
GASR_SLOW=1 python3 -m unittest tests.test_serve_engine -v  # adds the equivalence proofs
```

The slow three run the unmodified sampler; the whole suite then takes about 19 minutes. They are the only
tests here that matter: they are what stands between "the live demo agrees with the
paper" and a claim nobody checked. Run them before the viva.

`test_serve_auth` runs the real HTTP handler against a stub engine, so it needs
neither the checkpoint nor the bundle. What it pins is the negative: that a caller
who has not signed in cannot get a prediction, cannot learn which video the bundle
holds, and cannot tell a wrong password from an unknown roll number.

The client half is tested in `demo/clinical-src` (`npm test`), including the one
guarantee this whole design rests on: that on the `file://` origin the page makes **no
request at all**, asserted by counting calls against a stubbed `fetch`.
