# Clinical Demonstration App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A self-contained, offline, arrow-key-driven web app that demonstrates the trained deformation-prediction model to clinicians at the FYP viva, plus the Module 9 export changes it needs.

**Architecture:** One HTML page in `demo/clinical/` loading three classic scripts — pure logic, scene definitions, placeholder data. All comparative claims are computed at runtime from the loaded data; none are written into the page. Module 9 gains a `demo_data.js` sibling output (a `<script>`-loadable copy of the same payload), exported endoscopic JPEGs, per-horizon uncertainty, and recovered frame timing.

**Tech Stack:** Vanilla JS, canvas 2D, no build step, no package manager, no CDN. Tests: `node --test` (Node v22.23.2, built in) for JS; `python3 -m unittest` (stdlib) for Python. No pytest, torch or cv2 exists locally.

**Spec:** `docs/superpowers/specs/2026-11-17-clinical-demo-app-design.md` — read it first; this plan argues from it.

## Global Constraints

- **Must open from `file://` by double-click.** No server, no network, no install.
- **No `<script type="module">` and no `fetch()` of local data.** Both are blocked from `file://` by CORS. Classic `<script src>` and `<img src>` are permitted and are the only load mechanisms allowed.
- **No comparative claim, verdict or ranking written into the page.** Every one is computed from the loaded `mean_error`. The repo's only MPE-style comparison (`paper/measured.json` → `comparison`) is from the pre-fix epoch-60 W=4 checkpoint, `read_from_figure: true`, and must not shape this app.
- **No absolute physical units, ever.** No millimetres. No single averaged millisecond figure. Horizons are stated in frames; elapsed time only as a range, only when `timing_ok` is true.
- **Errors lead with the motion-relative figure** — "N% of the distance the tissue actually travelled". Span-relative appears once, as secondary context, labelled.
- **Placeholder mode shows no endoscopic imagery at all**, and labels every screen `PLACEHOLDER DATA`.
- **Never commit a real surgical frame into the repo.**
- Palette, fixed: `--blood:#D62828`, `--crimson:#5C0A0A`, `--flesh:#C98276`, `--white:#F5F5F7`.
- `demo/index.html` must end byte-identical to its current state.
- Module 9 keeps running as a notebook cell after Module 6, reusing `videos`, `predict_direct`, `BUNDLE`, `WINDOW_SIZE` from memory.

## Review Focus

Five conditions the spec implies that no task's happy path exercises. Each has its test pinned to the task that owns the code.

1. **`mean_error` is null** (placeholder mode). Verdict and ranking functions must return "unknown", never fabricate or crash. — Task 1.
2. **Motion over the horizon is zero.** The skill-score denominator divides by distance travelled; a static window makes it 0. Must return null, not `Infinity` or `NaN`. — Task 1.
3. **A horizon listed in `horizons` has no `sigma{k}` or `gt{k}`** — an old `demo_data.js` loaded into the new app. Must degrade to what is present, not throw mid-render. — Task 3.
4. **`timing_ok` false, or `frame_gap` missing.** Must silently fall back to frames-only. Printing any time here is the millisecond version of the millimetre anti-goal. — Task 1.
5. **`frames/` directory absent** while real data is loaded. Scenes 1–2 must show the stated "imagery comes from the export" panel, not a broken-image icon. — Task 3.

---

## File Structure

| File | Responsibility |
|---|---|
| `demo/clinical/core.js` | Create. Pure maths and all computed verdicts. Dual-mode: sets `window.GASR`, and `module.exports` for Node. No DOM. |
| `demo/clinical/scenes.js` | Create. The seven scenes: title, on-screen prose, which layers each shows. All clinician-facing wording lives here and nowhere else, so it can be reviewed in one small file. |
| `demo/clinical/synth.js` | Create. Placeholder geometry generator. Geometry only — never imagery. |
| `demo/clinical/index.html` | Create. Shell, canvas rendering, keyboard and on-screen controls, render loop. |
| `demo/clinical/README.md` | Create. How to run it, what placeholder mode means, what to copy off Drive. |
| `tests/core.test.js` | Create. `node --test`. |
| `tests/app.test.js` | Create. `node --test`. Static assertions on `index.html` that pin the `file://` constraints. |
| `tests/test_module9_export.py` | Create. `python3 -m unittest`. Extracts Module 9's pure helpers by AST and tests them without torch. |
| `module9_demo_export.py` | Modify. New outputs and fields. |
| `FYP_MultiVideo_Pipeline_(11).ipynb` | Modify. Cell 47 resynced from the modified Module 9. |

---

### Task 1: Pure logic and computed verdicts

**Files:**
- Create: `demo/clinical/core.js`
- Test: `tests/core.test.js`

**Interfaces:**
- Consumes: nothing.
- Produces, all on `window.GASR` and `module.exports`:
  - `METHOD_LABELS` — `{ours: 'GASR (ours)', cv: 'Straight-line motion', still: 'No movement'}`
  - `unflat(flat: number[], n: number) -> [number,number,number][]`
  - `meanPointError(a: Point[], b: Point[]) -> number`
  - `motionTravelled(ctx: Point[], gt: Point[]) -> number`
  - `skillPercent(err: number, motion: number) -> number | null`
  - `spanPercent(err: number, span: number) -> number | null`
  - `rankAtHorizon(meanError: object | null, k: number) -> {key: string, err: number}[] | null`
  - `verdict(meanError: object | null, k: number) -> {winner, winnerLabel, text, oursBeatsAll, oursBeatsStatic} | null`
  - `horizonLabel(k: number, timing: object | null) -> string`

- [ ] **Step 1: Write the failing tests**

```js
const t = require('node:test'), a = require('node:assert');
const G = require('../demo/clinical/core.js');

t.test('unflat groups triples', () => {
  a.deepStrictEqual(G.unflat([1,2,3,4,5,6], 2), [[1,2,3],[4,5,6]]);
});

t.test('meanPointError averages euclidean distance', () => {
  a.strictEqual(G.meanPointError([[0,0,0],[0,0,0]], [[3,4,0],[0,0,0]]), 2.5);
});

t.test('motionTravelled measures ctx to gt distance', () => {
  a.strictEqual(G.motionTravelled([[0,0,0]], [[0,0,2]]), 2);
});

t.test('skillPercent is error as percent of motion', () => {
  a.strictEqual(G.skillPercent(0.5, 2), 25);
});

// Review Focus 2
t.test('skillPercent returns null when nothing moved', () => {
  a.strictEqual(G.skillPercent(0.5, 0), null);
  a.strictEqual(G.skillPercent(0.5, -1), null);
});

t.test('rankAtHorizon orders methods by error ascending', () => {
  const me = {'3': {ours: 0.010, cv: 0.014, still: 0.016}};
  a.deepStrictEqual(G.rankAtHorizon(me, 3).map(r => r.key), ['ours','cv','still']);
});

t.test('verdict names ours when ours is lowest', () => {
  const me = {'3': {ours: 0.010, cv: 0.014, still: 0.016}};
  const v = G.verdict(me, 3);
  a.strictEqual(v.winner, 'ours');
  a.strictEqual(v.oursBeatsAll, true);
  a.match(v.text, /closest/i);
});

t.test('verdict names the baseline honestly when a baseline wins', () => {
  const me = {'3': {ours: 0.0225, cv: 0.0187, still: 0.0222}};
  const v = G.verdict(me, 3);
  a.strictEqual(v.winner, 'cv');
  a.strictEqual(v.oursBeatsAll, false);
  a.strictEqual(v.oursBeatsStatic, false);
  a.match(v.text, /Straight-line motion/);
});

// Review Focus 1
t.test('verdict and rank return null when there is no measurement', () => {
  a.strictEqual(G.verdict(null, 3), null);
  a.strictEqual(G.rankAtHorizon(null, 3), null);
  a.strictEqual(G.verdict({'1': {}}, 3), null);
});

t.test('horizonLabel states frames only without timing', () => {
  a.strictEqual(G.horizonLabel(3, null), '3 frames ahead');
});

// Review Focus 4
t.test('horizonLabel refuses to state time when timing is unusable', () => {
  a.strictEqual(G.horizonLabel(3, {timing_ok: false, fps: 30, gap_min: 2, gap_max: 10}),
                '3 frames ahead');
  a.strictEqual(G.horizonLabel(3, {timing_ok: true}), '3 frames ahead');
});

t.test('horizonLabel gives a range, never a single figure, when timing is good', () => {
  const s = G.horizonLabel(3, {timing_ok: true, fps: 30, gap_min: 2, gap_max: 10});
  a.match(s, /3 frames ahead/);
  a.match(s, /0\.2.*1\.0\s*s/);
});
```

- [ ] **Step 2: Run to verify they fail**

Run: `node --test tests/core.test.js`
Expected: FAIL — cannot find module `../demo/clinical/core.js`

- [ ] **Step 3: Implement `demo/clinical/core.js`**

Plain object of the functions above, ending with the dual-mode footer so the same file is a classic browser script and a Node CommonJS module:

```js
if (typeof module !== 'undefined' && module.exports) module.exports = GASR;
if (typeof window !== 'undefined') window.GASR = GASR;
```

`verdict` composes its sentence from `rankAtHorizon` and `METHOD_LABELS`; when a baseline wins it names that baseline and adds that the model is the only method reporting a per-point spread. `horizonLabel` computes its range as `k * gap_min / fps` to `k * gap_max / fps`, and returns frames-only unless `timing_ok` is true and `fps`, `gap_min`, `gap_max` are all present and positive.

- [ ] **Step 4: Run to verify they pass**

Run: `node --test tests/core.test.js`
Expected: PASS, 12 tests

- [ ] **Step 5: Commit**

```bash
git add demo/clinical/core.js tests/core.test.js
git commit -m "Compute every comparative claim from the data, not the page"
```

---

### Task 2: Scenes and placeholder geometry

**Files:**
- Create: `demo/clinical/scenes.js`, `demo/clinical/synth.js`
- Test: append to `tests/core.test.js`

**Interfaces:**
- Consumes: `METHOD_LABELS` from Task 1.
- Produces:
  - `window.SCENES` / `module.exports` — array of exactly 7 `{id, title, body, layers, horizon}`. `layers` is a subset of `['frame','ctx','gt','pred','cv','still','sigma']`. `horizon` is `1`, `3`, or `null` for "whatever the user has selected".
  - `window.synth() -> payload` matching the real schema, with `synthetic: true`, `mean_error: null`, `frame_files: {}`, and `sigma1`/`sigma3` per frame.

- [ ] **Step 1: Write the failing tests**

```js
const S = require('../demo/clinical/scenes.js');
const synth = require('../demo/clinical/synth.js');

t.test('there are exactly seven scenes in spec order', () => {
  a.strictEqual(S.length, 7);
  a.deepStrictEqual(S.map(s => s.id),
    ['sees','reconstructs','why','prediction','baselines','uncertainty','standing']);
});

t.test('every scene declares only known layers', () => {
  const ok = new Set(['frame','ctx','gt','pred','cv','still','sigma']);
  for (const s of S) {
    a.ok(s.title && s.body, `${s.id} needs title and body`);
    for (const l of s.layers) a.ok(ok.has(l), `${s.id}: unknown layer ${l}`);
  }
});

t.test('only the first two scenes use endoscopic imagery', () => {
  a.deepStrictEqual(S.filter(s => s.layers.includes('frame')).map(s => s.id),
                    ['sees','reconstructs']);
});

t.test('no scene hard-codes a comparative claim', () => {
  const banned = /\b(outperform|beats|better than|best|wins|superior|state of the art)\b/i;
  for (const s of S) a.doesNotMatch(s.body, banned, `${s.id} states a verdict`);
});

t.test('no scene states an absolute physical unit', () => {
  const banned = /\b\d+(\.\d+)?\s*(mm|cm|millimet|ms|millisecond)/i;
  for (const s of S) a.doesNotMatch(s.body, banned, `${s.id} invents a unit`);
});

t.test('synth is shaped like the real payload and claims nothing', () => {
  const d = synth();
  a.strictEqual(d.synthetic, true);
  a.strictEqual(d.mean_error, null);
  a.deepStrictEqual(d.horizons, [1, 3]);
  a.strictEqual(d.frames.length, d.n_frames);
  for (const k of d.horizons)
    for (const tag of ['gt','pred','cv'])
      a.strictEqual(d.frames[0][tag+k].length, d.n_points*3, `${tag}${k} wrong length`);
  a.strictEqual(d.frames[0].sigma1.length, d.n_points);
  a.strictEqual(d.frames[0].sigma3.length, d.n_points);
});
```

- [ ] **Step 2: Run to verify they fail**

Run: `node --test tests/core.test.js`
Expected: FAIL — cannot find `scenes.js`

- [ ] **Step 3: Write `scenes.js` and `synth.js`**

Scene bodies are the clinician-facing prose, written per the spec's seven claims: no verdicts, no units, plain language, two or three sentences each. Scene `standing` describes *what* is being measured and its scope — mean point error over the exported points of this one sequence, not the paper's ten-video table — and leaves the numbers to the runtime.

`synth.js` adapts the generator already in `demo/index.html` (lines 168–198): same breathing-surface construction, extended to emit `sigma1` and `sigma3` and flat arrays under the real field names. Keep its existing property that error is roughly flat in `k` while constant velocity's grows, so the placeholder does not teach a lesson the real data may contradict.

- [ ] **Step 4: Run to verify they pass**

Run: `node --test tests/core.test.js`
Expected: PASS, all tests

- [ ] **Step 5: Commit**

```bash
git add demo/clinical/scenes.js demo/clinical/synth.js tests/core.test.js
git commit -m "Put every clinician-facing sentence in one reviewable file"
```

---

### Task 3: The app shell

**Files:**
- Create: `demo/clinical/index.html`
- Test: `tests/app.test.js`

**Interfaces:**
- Consumes: `window.GASR` (Task 1), `window.SCENES`, `window.synth` (Task 2).
- Produces: nothing other tasks consume.

- [ ] **Step 1: Write the failing tests**

Static assertions, because the `file://` constraints are exactly what a browser test could not check offline.

```js
const fs = require('node:fs');
const html = fs.readFileSync('demo/clinical/index.html', 'utf8');

t.test('loads no module scripts and fetches no local data', () => {
  a.doesNotMatch(html, /type\s*=\s*["']module["']/);
  a.doesNotMatch(html, /\bfetch\s*\(/);
  a.doesNotMatch(html, /XMLHttpRequest/);
});

t.test('pulls nothing from the network', () => {
  a.doesNotMatch(html, /src\s*=\s*["']https?:/);
  a.doesNotMatch(html, /@import|cdn\.|unpkg|jsdelivr/);
});

t.test('every script it references exists on disk', () => {
  const srcs = [...html.matchAll(/<script[^>]+src=["']([^"']+)["']/g)].map(m => m[1]);
  a.ok(srcs.includes('core.js') && srcs.includes('scenes.js') && srcs.includes('synth.js'));
  for (const s of srcs)
    if (s !== 'demo_data.js') a.ok(fs.existsSync('demo/clinical/' + s), `missing ${s}`);
});

t.test('demo_data.js is optional, so its tag cannot block the page', () => {
  a.match(html, /<script[^>]+src=["']demo_data\.js["'][^>]*>/);
  a.match(html, /onerror/, 'the optional data script needs an onerror fallback');
});

t.test('uses the fixed palette', () => {
  for (const c of ['#D62828','#5C0A0A','#C98276','#F5F5F7']) a.match(html, new RegExp(c, 'i'));
});

t.test('carries the placeholder banner and the no-imagery panel', () => {
  a.match(html, /PLACEHOLDER DATA/);
  a.match(html, /id=["']noimg["']/);
});

t.test('the research viewer is untouched by this task', () => {
  a.doesNotMatch(fs.readFileSync('demo/index.html','utf8'), /clinical/);
});
```

- [ ] **Step 2: Run to verify they fail**

Run: `node --test tests/app.test.js`
Expected: FAIL — `demo/clinical/index.html` does not exist

- [ ] **Step 3: Build `demo/clinical/index.html`**

Script tags in order: `core.js`, `scenes.js`, `synth.js`, then `demo_data.js` with an `onerror` that leaves `window.DEMO_DATA` undefined so the page falls back to `synth()` and shows the placeholder banner.

Projector-first layout: scene title and prose in large type, the geometry canvas as the main stage, the endoscopic frame beside it on scenes 1–2 only. Reuse the orbit/zoom projection from `demo/index.html` (lines 200–296) — same maths, larger point radii and heavier strokes for projection distance. One fixed colour per series across all scenes.

Controls, both keyboard and on-screen: `←`/`→` scene, `space` play/pause, `1`/`3` horizon, `R` reset view. Every control stays live in every scene.

Per the Review Focus, the render path must tolerate a horizon whose `gt{k}`/`sigma{k}` is missing by drawing what exists and labelling the rest unavailable, and must show the `#noimg` panel when a frame's JPEG is absent — including when `frame_files` is empty under real data.

- [ ] **Step 4: Run tests, then open it**

Run: `node --test tests/app.test.js`
Expected: PASS, 7 tests

Then open `demo/clinical/index.html` by double-clicking — not through a server — and confirm: the placeholder banner shows, all seven scenes advance with the arrow keys, orbit and zoom work, and the browser console is clean.

- [ ] **Step 5: Commit**

```bash
git add demo/clinical/index.html tests/app.test.js
git commit -m "Build the viva demonstrator as a double-clickable page"
```

---

### Task 4: Module 9 export

**Files:**
- Modify: `module9_demo_export.py`
- Test: `tests/test_module9_export.py`

**Interfaces:**
- Consumes: Module 6's in-memory `videos`, `predict_direct`, `BUNDLE`, `WINDOW_SIZE`, `timing_by_ws`.
- Produces, in `{DEMO_DIR}`: `demo_data.json` (unchanged consumers), `demo_data.js`, `frames/<index>.jpg`.
- New payload fields: `span`, `frame_files`, `timing`, per-frame `img`/`img{k}`, per-frame `sigma{k}` replacing `sigma`.

The pure helpers below must be defined at module top level with no torch in their bodies, so the test can extract and exercise them without torch installed:

- `needed_frame_indices(origins: list[int], horizons: list[int]) -> list[int]` — sorted unique absolute frame indices: `t-1` (last observed) and `t+k-1` (ground truth at horizon `k`) for every origin `t`. These are Module 9's own existing conventions, read off its loop; do not re-derive them.
- `frame_file_map(indices: list[int]) -> dict[int, str]`
- `motion_by_horizon(frames: list[dict], horizons: list[int]) -> dict[str, float]`
- `js_payload(obj: dict) -> str`

- [ ] **Step 1: Write the failing test**

```python
import ast, json, unittest, numpy as np

SRC = 'module9_demo_export.py'
NAMES = ['needed_frame_indices', 'frame_file_map', 'motion_by_horizon', 'js_payload']

def load_helpers():
    """Module 9 is a notebook cell: it imports torch and reads DRIVE_BASE at import
    time, so it cannot be imported here. Extract just the pure helpers by AST."""
    tree = ast.parse(open(SRC).read())
    want = {n.name: n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in NAMES}
    missing = set(NAMES) - set(want)
    assert not missing, f'helpers not at module level: {missing}'
    ns = {'np': np, 'json': json}
    exec(compile(ast.Module(body=[want[n] for n in NAMES], type_ignores=[]),
                 SRC, 'exec'), ns)
    return ns

class TestHelpers(unittest.TestCase):
    def setUp(self):
        self.h = load_helpers()

    def test_needed_indices_are_sorted_and_unique(self):
        # Module 9's convention, from its own loop: ctx is tn[t-W:t] so the last
        # OBSERVED frame is t-1, and gt{k} is tn[t+k-1]. Origins 10 and 12 at
        # horizons 1 and 3 therefore need 9,10,12 and 11,12,14 -> 5 unique indices.
        got = self.h['needed_frame_indices']([10, 12], [1, 3])
        self.assertEqual(got, [9, 10, 11, 12, 14])

    def test_frame_file_map_is_zero_padded(self):
        self.assertEqual(self.h['frame_file_map']([7, 1234])[7], 'frames/00007.jpg')

    def test_motion_by_horizon_measures_distance_travelled(self):
        frames = [{'ctx': np.zeros((2, 3)),
                   'gt1': np.array([[0., 0., 1.], [0., 0., 3.]]),
                   'gt3': np.array([[0., 0., 2.], [0., 0., 4.]])}]
        m = self.h['motion_by_horizon'](frames, [1, 3])
        self.assertAlmostEqual(m['1'], 2.0)
        self.assertAlmostEqual(m['3'], 3.0)

    def test_js_payload_is_an_assignment_the_browser_can_load(self):
        s = self.h['js_payload']({'video': 'v1', 'n': 2})
        self.assertTrue(s.startswith('window.DEMO_DATA = '))
        self.assertTrue(s.rstrip().endswith(';'))
        self.assertEqual(json.loads(s[len('window.DEMO_DATA = '):].rstrip().rstrip(';')),
                         {'video': 'v1', 'n': 2})

class TestExportContract(unittest.TestCase):
    def test_no_authored_verdict_survives_in_the_comments(self):
        """The old header claimed constant velocity 'beats us at k=1 and loses by
        k=3' -- a finding from the pre-fix checkpoint that the design forbids."""
        src = open(SRC).read()
        for phrase in ('beats us', 'loses by'):
            self.assertNotIn(phrase, src)

    def test_sigma_is_exported_per_horizon(self):
        src = open(SRC).read()
        self.assertIn("f'sigma{k}'", src)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m unittest discover -s tests -v`
Expected: FAIL — `helpers not at module level`

- [ ] **Step 3: Modify `module9_demo_export.py`**

Add the four helpers at module top level. Then in `export_demo`:

- sample `sigma` once per horizon rather than only at `DEMO_HORIZONS[0]`, batching the `DEMO_SAMPLES` passes, and store under `sigma{k}`;
- write `frames/<index>.jpg` for each index from `needed_frame_indices`, reading `{DRIVE_BASE}/{workspace}/frames/{index:05d}.png` and downscaling to 320 px wide at JPEG quality 75 with `cv2`;
- add `span` (state in a comment that it is `2 * radius`), `frame_files`, `motion` from `motion_by_horizon`, and `timing` taken from Module 6's recovered `timing_by_ws` entry for this workspace — carrying `timing_ok`, `fps`, `gap_min`, `gap_max`, and nothing derived;
- write `demo_data.js` beside `demo_data.json` using `js_payload`;
- rewrite the header comment that currently asserts the k=1/k=3 crossover as settled, and the closing print, which should now name both files and the `frames/` directory.

- [ ] **Step 4: Run tests and preflight**

Run: `python3 -m unittest discover -s tests -v`
Expected: PASS, 6 tests

Run: `python3 tools/preflight.py`
Expected: `PASS: 0 suspect name(s)`

- [ ] **Step 5: Commit**

```bash
git add module9_demo_export.py tests/test_module9_export.py
git commit -m "Export what the clinical demonstrator needs, per horizon"
```

---

### Task 5: Notebook sync and operator docs

**Files:**
- Modify: `FYP_MultiVideo_Pipeline_(11).ipynb` (cell 47)
- Create: `demo/clinical/README.md`
- Modify: `RUNBOOK.md` (step 5)

**Interfaces:**
- Consumes: the finished `module9_demo_export.py` from Task 4.
- Produces: nothing other tasks consume.

- [ ] **Step 1: Write the failing test**

```python
class TestNotebookSync(unittest.TestCase):
    def test_cell_47_matches_module9_on_disk(self):
        import io, json
        nb = json.load(io.open('FYP_MultiVideo_Pipeline_(11).ipynb', encoding='utf-8'))
        cell = ''.join(nb['cells'][47]['source'])
        self.assertEqual(cell.strip(), open(SRC).read().strip())
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m unittest discover -s tests -v`
Expected: FAIL — cell 47 still holds the pre-Task-4 source

- [ ] **Step 3: Resync the cell and write the docs**

Replace cell 47's source with the file verbatim, preserving the notebook's `indent=1` JSON formatting and trailing newline — the same procedure used for cells 36/42/44 earlier in this project's history.

`demo/clinical/README.md`: what the app is and who it is for, that it opens by double-clicking with no server, which three files to copy out of `{DRIVE_BASE}/module9_demo` (`demo_data.js` and the `frames/` directory; `demo_data.json` belongs to the other viewer), what placeholder mode looks like and why it carries no imagery, and the keyboard controls.

`RUNBOOK.md` step 5: note that Module 9 now writes `demo_data.js` and `frames/` as well, and add the clinical app beside the research viewer in the copy instructions.

- [ ] **Step 4: Run the full suite**

Run: `python3 -m unittest discover -s tests -v && node --test tests/`
Expected: PASS throughout

Run: `git diff --stat HEAD -- demo/index.html`
Expected: no output — the research viewer is untouched

- [ ] **Step 5: Commit**

```bash
git add FYP_MultiVideo_Pipeline_\(11\).ipynb demo/clinical/README.md RUNBOOK.md
git commit -m "Sync the Module 9 cell and document how to run the demonstrator"
```

---

## Self-Review

**Spec coverage.** Intent, goals, non-goals and anti-goals → Tasks 1–3 (computed verdicts, no units, no imagery in placeholder mode) and the Global Constraints. Data delivery → Tasks 3 and 4. New Module 9 fields, all five → Task 4. Expressing error → Task 1 (`skillPercent`, `spanPercent`) and Task 2's prose. Seven scenes → Task 2. Interaction → Task 3. Look → Task 3. Where it goes → the file structure. "Left to the builder" items are deliberately unconstrained here.

**Known gap, accepted:** `motion_by_horizon` lands in the payload in Task 4 but the app's scenes consume it through `skillPercent`, which Task 1 tests against literal values rather than against a real export. Nothing can close that until Module 6 and Module 9 have run on the fixed checkpoint — which is the same reason no verdict is authored. Task 3's step 4 ends in a manual open against placeholder data; the first run against real data is the user's, after Colab.

**Type consistency.** `mean_error` keys are strings (`'1'`, `'3'`) throughout, matching the existing exporter. `sigma{k}` is consistent across Tasks 2, 3 and 4. Method keys `ours`/`cv`/`still` match `demo_data.json`'s existing `mean_error` block. `timing` carries only `timing_ok`/`fps`/`gap_min`/`gap_max`, consumed by `horizonLabel` with exactly those names.

**Proportion.** Five tasks, roughly 420 lines of plan against an app of perhaps 900. Code blocks are tests and signatures; no function bodies are transcribed.
