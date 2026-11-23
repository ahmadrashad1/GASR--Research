# The clinical demonstrator

A seven-scene, arrow-key-driven walkthrough of what the trained model predicts, pitched
at clinicians and an examination panel rather than at a reviewer checking the paper.

This is a **second reader of the same exported data**, not a replacement for
`demo/index.html`. That one is the research console — layer checkboxes, an error table,
research vocabulary — and it stays exactly as it is. Use it to verify the paper's
Stage 7 claim. Use this one in the room.

## Running it

**Double-click `index.html`.** No server, no terminal, no internet, no install. This
is the path that must work, and it is the one to use if anything is uncertain.

**Optionally**, start the local inference server first and open it over http instead:

```bash
python3 serve/server.py --warm      # then open http://localhost:8765
```

Same page, same file, one addition: a **run the model now** button that predicts on
this laptop in about ten seconds, at any origin in the held-out tail — including ones
the export never visited. Everything else is identical. See `serve/README.md`.

The page never requests anything when opened from `file://`, so the double-click path
is unchanged and cannot be affected by the server being absent, broken, or forgotten.
If the server dies mid-demo the page says so in one line and carries on with the
exported measurement.

That constraint shapes the whole build. A page on the `file://` origin cannot fetch
local data and cannot load a module script, so the deliverable is a single
self-contained HTML file: React, the styles and all the logic are inlined into it as a
classic script. `demo_data.js` and `frames/` stay outside it — they are the measured
export, they are large, and they are not committed.

## Changing it

The page is built, not hand-edited. Source lives in `demo/clinical-src/`
(React 19 + TypeScript + Tailwind 4, bundled by Vite), deliberately *outside* this
folder so `node_modules` never travels onto the USB stick.

```bash
cd demo/clinical-src
npm install
npm run dev      # hot reload at localhost; put demo_data.js in public/ to see real data
npm test         # 39 tests: logic, scenes, and the app mounted in a DOM
npm run check    # TypeScript
npm run build    # bundles, verifies, and installs ../clinical/index.html
```

`npm run build` refuses to install a page that would not open by double-click. It
checks for a module script, an external stylesheet, a network request and any use of
`fetch`, and fails the build rather than shipping a page that works on your laptop and
is blank on the examiner's.

## The overlay

Scenes 4, 5 and 6 draw the tracked points **on the endoscopic frame itself**, beside the
3D view. Nothing is computed in the browser: Module 9 projects the same arrays it
exports, using the sequence's own pinhole intrinsics, and writes image coordinates in
[0,1] alongside the 3D ones. The page just draws where it is told, so the overlay and
the figures can never disagree — they are the same numbers in two spaces.

No extrinsics are involved. The trajectories are already in camera space, which is why
this was cheap enough to build; the earlier design assumed otherwise and ruled it out.

**If the sequence has no `camera_meta.json`**, the export falls back to Module 1's
documented 75° field of view, marks the payload `calibrated: false`, and the app states
on screen that the points are placed plausibly rather than measurably. The geometry and
every figure beside it are unaffected — only the position of the dots on the image is
assumed. Check the export's console line if you want to know which you have.

An export made before the overlay existed has no image coordinates; the app detects
that and simply shows the still without dots.

## Controls

| Key | Button | Action |
|---|---|---|
| `←` `→` | back / next | previous or next scene |
| `space` | play | play or pause the sequence |
| `1` `3` | 1 frame / 3 frames | prediction horizon |
| `R` | reset view | reset the 3D camera |

Drag to orbit, scroll to zoom. Everything stays live in every scene, so when the panel
interrupts you can answer inside the scene you are on instead of dead-ending.

## What it will and will not say

Every comparative sentence — which method came closest, by how much — is computed from
the loaded `mean_error` as the page renders. None of it is written into the page. The
same readout that shows a win shows a loss, which is the point: the alternative is
standing in front of an examination panel asserting something the number on the same
screen contradicts.

Two things it will never do:

- **State an error in millimetres, or any physical unit.** These trajectories live in a
  per-video normalised space and nothing in this pipeline is calibrated to a physical
  scale. A surgeon will ask for millimetres; inventing them would be fabrication.
- **State a single elapsed time.** Kept frames are not evenly spaced — Module 1 takes
  every second raw frame and then drops blurred ones, so consecutive frames can be two
  or ten raw frames apart. Horizons are given in frames, which is exact, and elapsed
  time appears only as a range, and only when the export recovered usable timing.

Errors lead with the share of **how far the tissue actually travelled**, where 100% means
a prediction was no better than not predicting at all. The share of the tissue's overall
span appears second, labelled, for scale only — against that denominator every method in
this project, including assuming the tissue never moves, looks excellent, so it
distinguishes nothing.

Scene 7 reports the mean point error over the exported points of **one sequence**. It is
not the paper's comparison table, which is Chamfer distance over all ten videos. The two
will not agree, and saying so plainly beats a panel member noticing mid-viva.

## Live inference, and what it is not allowed to do

Live mode adds a prediction; it never restates a result.

- The readout table keeps averaging **the whole held-out sequence, from the export**,
  because that is what its sentences are about. A live origin's own errors appear in a
  separate panel, labelled as one origin, which is a single frame and not a result.
- The server refuses origins before Module 4's `split_frame`. Those frames trained the
  model, and predicting one would flatter it for the one reason that proves nothing.
  It refuses origins past the end too, for the unrelated reason that the video runs out
  before the horizon, and says which.
- Nothing is precomputed. The bundle the server reads carries the tracked trajectory
  and no predictions at all, so what appears on screen was computed in the room.
- A live prediction uses the same checkpoint, the same 20 denoising steps and the same
  4 averaged trajectories as every number in the report.

## Tests

```bash
cd demo/clinical-src && npm test      # logic, components, live fallback (62)
node --test tests/app.test.js         # the built page, from the repo root (8)
python3 -m unittest tests.test_serve_engine   # the inference server (17)
```

`core.test.js` covers the maths and the computed verdicts, including what happens with
no measurement and with a horizon where nothing moved. `app.test.js` pins the `file://`
constraints — no module scripts, no local data requests, nothing from the network — and
boots the page's own script against a stub DOM to walk every scene. That boot test
cannot see layout or colour, so **open the page and look at it before the viva.**
