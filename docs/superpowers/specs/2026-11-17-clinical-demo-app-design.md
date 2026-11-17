# Clinical demonstration app — design

**Date:** 17 November 2026
**Project:** S26-017-R-GASR — GASR: 4D Surgical Scene Reconstruction & Deformation Prediction
**Status:** awaiting approval

---

## Intent and why

Ahmad will drive this app on his own laptop at the FYP viva, with Dr Jamil and the
examination panel watching. Its job is to make the trained diffusion model's behaviour
legible to clinicians and examiners in a few minutes, without Ahmad having to explain
Chamfer distance or denoising schedules to get there.

The existing viewer, `demo/index.html`, is a research console: layer checkboxes, a
mean-point-error table, orbit hints. It is the right instrument for a reviewer checking the
paper's Stage 7 claim and it stays exactly as it is. It is the wrong instrument for a room
of clinicians, because every label in it is a research term.

This app is a second reader of the same exported data, pitched at the room instead of the
reviewer.

## Goals

- A surgeon with no machine-learning background understands, by watching, what the system
  predicts and how far ahead.
- The panel can see the model's prediction set against what actually happened, and against
  the obvious alternative guesses, on held-out frames.
- **One sequence, not ten.** Module 9 exports the single video with the longest held-out
  tail, and the app shows that one. "Scrub to the case they asked about" below means a
  *moment* within that sequence, not a different patient. A case-switcher would multiply the
  JPEG budget tenfold and add a control to all seven scenes for no gain in the room.
- Ahmad can drive it start to finish on a projector with the arrow keys, and never has to
  improvise structure under examination.
- When the panel interrupts with a question, the controls are live inside whatever scene is
  on screen, so he can scrub to the case they asked about rather than dead-ending.
- It opens by double-clicking the file. No server, no terminal, no internet, no install.

## Non-goals

- **No live inference.** The model does not run in the browser or behind it. Everything shown
  is pre-exported by Module 9 from the trained checkpoint on held-out frames.
- **No projection of predicted points onto the endoscopic image.** A true overlay needs the
  camera intrinsics and extrinsics out of the splatting workspace. That is the one piece of
  this that could quietly consume the remaining time before submission, and side-by-side
  frame-synced views read just as well to a clinician.
- **No replacement of `demo/index.html`.** The paper's Stage 7 section already describes that
  viewer. This is an addition.
- **No new training, no new metrics.** It reads what Module 9 already computes.

## Anti-goals

Things that would make this a failure even if it technically worked.

- **Stating errors in millimetres.** The trajectories live in a per-video normalised space
  (`traj_n = (traj - centre) / scale`); there is no metric calibration anywhere in this
  pipeline. A surgeon will want millimetres and inventing them would be fabrication. Errors
  are expressed as a percentage of the reconstructed tissue's own span, labelled as such.
- **Implying clinical validation.** This is a feasibility demonstration on ten
  reconstructed endoscopic sequences. The app must not imply it has been tested for
  surgical use, nor suggest a diagnosis or an action.
- **Showing placeholder data as if it were real.** The app must run before Module 9 has been
  exported, so Ahmad can rehearse — and when it does, every screen must say so.
- **Anything that needs a network or a local server at the viva.** A demo that fails in the
  room is worse than no demo.

## The result is not known yet, and the app must not assume it

`paper/measured.json` → `comparison` is the only MPE-style comparison in the repo, and it
reports constant velocity beating this model at every horizon. It must not be used to shape
this app, in either direction:

- its `source` is *"Module 6 QT1, checkpoint epoch 60, full held-out set"* — the pre-fix
  checkpoint, at a 4-frame window with absolute-coordinate context, which is the defect this
  project diagnosed and corrected;
- its `dataset` block beside it records `window_size: 4`;
- it carries `read_from_figure: true`, meaning the values were read off a chart by eye.

The only evidence on the fixed checkpoint is Module 5's own quality tests at epoch 100:
Chamfer distance 0.012984 (ours) against 0.014151 (constant velocity) and 0.016195 (static).
That is a different metric on a validation subset, and it does not establish what mean point
error does at k=1, 2 and 3 on the full held-out set. **Module 6 has not run on this
checkpoint.** Until it does, whether the model beats straight-line extrapolation on position
is genuinely unknown.

**Therefore no verdict, ranking or comparative claim is written into the page.** Every such
statement is computed at runtime from the `mean_error` block of the loaded data, the way
`verdict()` in `demo/index.html` already does. Scenes 4, 5 and 7 must each render correctly
and honestly in both outcomes:

- **if the model wins at a horizon** — it is named as the closest prediction there;
- **if a baseline wins** — the scene says so plainly, and the scene's weight shifts to what
  remains true regardless: the model produces a distribution with calibrated per-point
  spread, which neither baseline can do at all, and which Module 6 measures as a
  spread-to-error correlation.

Fixed prose reads better on a projector, and this costs us some of that. It is the price of
not standing in front of an examination panel making a claim the data on the screen
contradicts.

## Constraints

- **The data does not exist yet.** `demo_data.json` has never been generated. Module 9 is
  RUNBOOK step 5, and Module 6 has not completed since the W=32 fix. The app has to be
  reviewable against the synthetic placeholder and correct against the real export without
  a second build.
- **`file://` blocks `fetch`.** Chrome will not let a page opened from disk `fetch` a JSON
  file next to it. Images via `<img src>` are fine. This dictates the data-delivery
  mechanism below.
- **Module 9 runs inside the notebook**, as a cell after Module 6, reusing that module's
  in-memory `videos`, `predict_direct`, `BUNDLE` and `WINDOW_SIZE`. Any change to it has to
  keep that shape.
- **The endoscopic frames are on Drive** at `{DRIVE_BASE}/{workspace}/frames/00000.png`,
  one PNG per kept frame, already read by Module 6 with `cv2` for frame-timing recovery.
- **One self-contained HTML file** of the kind already in this repo: vanilla JS, canvas, no
  build step, no package manager, no external CDN.

## Approaches considered

**A. A "clinical mode" toggle inside `demo/index.html`.**
Cheapest — one file, one checkbox. Rejected: the layout *is* a research console. The panel
would be reading layer checkboxes and an error table while Ahmad talks about anatomy. A
toggle cannot undo an information architecture built for a different reader.

**B. A guided presentation app that stays live — chosen.**
A new self-contained page driven like a deck: ordered scenes, one claim each, advanced with
the arrow keys, type sized for a projector — but every scene keeps the playback, horizon and
orbit controls active, so a question from the panel is answered by scrubbing rather than by
leaving the demo. Carries the structure Ahmad needs under examination and still survives
interruption.

**C. A free-form clinical workstation.**
Endoscopic video and 3D geometry side by side, scrubber, overlays, no narrative. The better
product if a clinician were exploring it alone, which is explicitly not the setting. At a
viva it means improvising the story while being graded.

**Chosen: B, with C's controls built into every scene.** The narrative carries the demo; the
controls rescue it from questions.

## How it gets built

### Data delivery — settled

Module 9 gains a second output beside the existing `demo_data.json`:

- **`demo_data.js`** — the identical payload, written as `window.DEMO_DATA = {...};`. Loaded
  with `<script src="demo_data.js">`, which `file://` permits. The JSON file stays for
  `demo/index.html`, which is served and already reads it.
- **`frames/<index>.jpg`** — the real endoscopic frames, one per absolute frame index the
  export actually references, downscaled to 320 px wide at JPEG quality 75. Referenced from
  the page by `<img src>`, which `file://` permits. Roughly 200 unique indices at ~20 KB
  each; a few megabytes.

The app loads the real data if `window.DEMO_DATA` is present and falls back to the existing
synthetic generator otherwise, labelling every screen `PLACEHOLDER DATA` when it does.

**In placeholder mode there is no endoscopic imagery at all.** Scenes 1 and 2 show the
geometry panel with a plain note that the surgical frames come from the real export. Two
alternatives were rejected: a drawn or generated "endoscopic" image would be synthetic
clinical imagery on screen, which the anti-goal above forbids even when labelled; and
committing a real surgical frame into the repo as a rehearsal still is a patient-data
decision that is not the builder's to make. Scenes 3 to 7 rehearse fully on synthetic
geometry.

### Existing payload, unchanged

```
video, n_points, n_frames, horizons [1,3], window, centre[3], radius,
target_mode, checkpoint_epoch,
mean_error: { "1": {ours, cv, still}, "3": {...} },
frames[]: { t, ctx, sigma, gt1, pred1, cv1, gt3, pred3, cv3 }
```

Point arrays are flat `[x,y,z,...]` of length `3 * n_points`; `sigma` is length `n_points`.
`ctx` is the last observed frame, so it doubles as the freeze-in-place baseline.

### New fields Module 9 must add

- `frame_files`: map from absolute frame index to its exported JPEG filename.
- Per frame entry: `img` (the index of the last observed frame, which the prediction is made
  *from*) and `img{k}` (the index of the frame `k` ahead, what actually happened).
- `span`: the reconstructed tissue's extent in normalised units. `2 * radius` is the natural
  choice; Module 9 states which it used rather than leaving the app to assume. See
  *Expressing error* below — `span` is not the only denominator the app needs.
- **`sigma{k}` per horizon, replacing the single `sigma`.** Today `sigma` is sampled at
  `DEMO_HORIZONS[0]` only, so at k=3 the app would draw k=1's spread around a k=3
  prediction — wrong, and wrong in the flattering direction. Each horizon gets its own
  spread. Cost is 80 origins x 8 samples x 2 horizons of DDIM sampling; it must be batched,
  given this notebook has already hit one out-of-memory failure during sampling. If that
  export proves too slow in practice, the fallback is for scene 6 to pin k=1 and say so,
  not to reuse the wrong spread.
- **`frame_gap`: the recovered raw-frame span and `timing_ok` for the chosen video.** Kept
  frames are not evenly spaced in time — Module 1 takes every second raw frame and then
  drops blurred ones, so consecutive trajectory frames can be 2 or 10 raw frames apart.
  Module 6's `recover_frame_timing` already computes this and it is in memory when Module 9
  runs.

  The horizon is therefore stated **in frames** everywhere, which is exact. Elapsed time
  appears only as a *range* ("3 frames ahead — between about 0.1 and 0.5 s at this
  sequence's recovered frame rate"), only when `timing_ok` is true, and never as a single
  averaged figure. A single "≈200 ms" over origins whose real gaps vary by five times is the
  same class of fabrication as the millimetre anti-goal, and it is the kind of number a
  panel asks a second question about.

### Expressing error

Two denominators are available and they tell opposite stories.

**Relative to the tissue's span** (`2 * radius`, about 2 normalised units) every method in
this project lands under 0.5%, including "assume the tissue does not move". It is a real
number, it is reassuring, and it distinguishes nothing. Leading with it is the quietest way
this app could break the "no implying validation" anti-goal while satisfying every line of
the verification list — the panel would read "0.4% error" as excellence, for a prediction
that may be no better than freezing the frame.

**Relative to how far the tissue actually moved over that horizon** is the denominator a
prediction claim has to answer to. It is a skill score: at 100% the prediction is no better
than not predicting at all, below 100% it beats standing still, above 100% it is actively
worse. It makes the model's performance legible in one number and it cannot flatter.

**Decision: scenes 4 to 7 lead with the motion-relative figure**, in those words — "the
error is N% of the distance the tissue actually travelled". The span-relative figure appears
once, as secondary context for scale, and is labelled as such. This pairs with the
no-authored-verdict rule above: the same readout that shows a win shows a loss, so the app
stays honest whichever way Module 6 lands.

### Scenes

Seven, in order. Each is one claim, one visual.

1. **What the endoscope sees.** The real footage playing alone. Tissue is never still —
   breathing, pulse, retraction.
2. **What the system reconstructs.** Frame and 3D anchor cloud side by side, synced. The
   pipeline tracks 512 surface points per frame; the viewer draws an evenly strided 256 of
   them, because a browser canvas cannot usefully render more. Both numbers are stated on
   screen rather than quietly conflated.
3. **Why prediction matters.** A robot or an overlay acting on where tissue *is* acts on the
   past. The question is where it will be.
4. **The prediction.** Three frames ahead: the model's answer against what actually
   happened, on frames it never saw in training.
5. **Against the obvious guesses.** The same frames with "assume it keeps moving the same
   way" and "assume it stops". Both are reasonable; both come apart by three frames.
6. **How sure it is.** The per-point spread across sampled hypotheses, drawn as a halo. The
   model reports where it is uncertain instead of committing.
7. **Where it stands.** The numbers for *this sequence*, computed from `mean_error`, with
   the caveat about normalised units and feasibility rather than validation.

   Scene 7 reports mean point error over the 256 exported points of the one video on screen,
   and says so in those words. It is **not** the paper's comparison table, which is Chamfer
   distance over all ten videos and needs Module 7. The two will not agree, and a panel
   member reading one against the other mid-viva is a worse outcome than labelling the scope
   plainly. Importing an aggregate the exporter cannot compute is out of scope.

### Interaction

| Key | Action |
|---|---|
| `←` `→` | previous / next scene |
| `space` | play / pause the sequence |
| `1` `3` | prediction horizon |
| `R` | reset the 3D view |

Mouse: drag to orbit, scroll to zoom, in every scene. On-screen controls mirror all of it so
nothing depends on Ahmad remembering a key.

### Look

Built for a projector at the back of a room, not a laptop at reading distance: large type,
high contrast, the surgical palette already established in `demo/index.html`
(`--blood:#D62828`, `--crimson:#5C0A0A`, `--flesh:#C98276`, `--white:#F5F5F7`). Prediction,
ground truth and each baseline keep one consistent colour across all seven scenes, so the
panel learns the legend once.

### Where it goes

`demo/clinical/index.html`, with `demo_data.js` and `frames/` beside it. Keeps the research
viewer's directory clean and makes the whole app one copyable folder.

## Left to the builder

- Exact scene transitions and easing.
- Canvas drawing specifics: point sizes, depth cueing, halo rendering.
- How the error readout is laid out within each scene.
- Whether scene 5 shows both baselines at once or toggles between them.
- JPEG dimensions and quality, if 320 px / q75 proves wrong in practice.
- Responsive behaviour below projector size.

## Verification

- Runs from `file://` with no server, on a machine with no network.
- Runs with no `demo_data.js` present, labelled as placeholder throughout.
- Runs with the real export and reports the checkpoint epoch and video name from it.
- No error anywhere is stated in millimetres or any absolute physical unit.
- `demo/index.html` byte-identical to its current state.
