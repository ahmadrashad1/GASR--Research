# The clinical demonstrator

A seven-scene, arrow-key-driven walkthrough of what the trained model predicts, pitched
at clinicians and an examination panel rather than at a reviewer checking the paper.

This is a **second reader of the same exported data**, not a replacement for
`demo/index.html`. That one is the research console — layer checkboxes, an error table,
research vocabulary — and it stays exactly as it is. Use it to verify the paper's
Stage 7 claim. Use this one in the room.

## Running it

**Double-click `index.html`.** No server, no terminal, no network, no install.

That constraint is the reason for the whole file layout. A page opened from disk runs on
the `file://` origin, where the browser refuses both module scripts and local data
requests, so the data arrives as a classic script that assigns `window.DEMO_DATA`.
Nothing here is loaded over a network, so it works on a locked-down laptop in an
examination room with the wifi off.

## Getting the real data in

Copy two things out of `{DRIVE_BASE}/module9_demo/` to sit beside `index.html`:

| Copy | What it is |
|---|---|
| `demo_data.js` | the payload, as a `<script>`-loadable file |
| `frames/` | the exported endoscopic stills, referenced by `<img src>` |

Do **not** copy `demo_data.json` here — that is the same payload for the other viewer,
which is served over HTTP and can request it. This page cannot.

```
demo/clinical/
├── index.html      ← double-click this
├── core.js
├── scenes.js
├── synth.js
├── demo_data.js    ← from Drive
└── frames/         ← from Drive
```

Neither copied file is committed. `frames/` holds real endoscopic imagery, and putting a
surgical frame into a public repository is a patient-data decision that is not the
builder's to make.

## The badge, and what placeholder mode means

The badge in the top right reads **MEASURED** or **PLACEHOLDER DATA**. There is no third
state and no silent fallback.

Without `demo_data.js` the app runs on invented geometry so the walkthrough can be
rehearsed. In that mode:

- every screen is labelled `PLACEHOLDER DATA`;
- **no comparison is shown at all** — placeholder data carries no measurement, so there
  is nothing for the app to read, and it cannot state a verdict even by accident;
- **there is no endoscopic imagery.** Scenes 1 and 2 show a panel saying the imagery
  comes from the export. A generated "endoscopic" picture would be synthetic clinical
  imagery on a projector, which is worse than an honest blank.

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

## Tests

```bash
node --test tests/core.test.js tests/app.test.js      # from the repo root
```

`core.test.js` covers the maths and the computed verdicts, including what happens with
no measurement and with a horizon where nothing moved. `app.test.js` pins the `file://`
constraints — no module scripts, no local data requests, nothing from the network — and
boots the page's own script against a stub DOM to walk every scene. That boot test
cannot see layout or colour, so **open the page and look at it before the viva.**
