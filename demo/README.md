# Deformation prediction viewer

The Module 9 deliverable: a browser viewer for the self-supervised tissue
deformation prediction, running on held-out frames of a real endoscopic video.

## Running it

```bash
python3 -m http.server -d demo 8000   # then open http://localhost:8000
```

Opening `index.html` directly with `file://` also works, except that the browser
will refuse the `fetch` of `demo_data.json`; the viewer falls back to synthetic
data and says so in the header. Serve it over HTTP to see measured results.

## Getting real data into it

1. Run Modules 1-6 as usual.
2. Run `module9_demo_export.py` as the next cell, in the same runtime. It reuses
   Module 6's `videos`, `predict_direct` and `base_pred` rather than re-deriving
   them, so what the viewer draws is what the metrics measured.
3. Copy `module9_demo/demo_data.json` from Drive into this folder.

The header badge reads **measured** once it loads, and **synthetic** otherwise.
There is no third state and no silent fallback: a viewer that invents results is
worse than one that refuses to start.

## What is on screen

| Layer | Meaning |
|---|---|
| White | ground truth at t+k |
| Red | our conditional diffusion prediction |
| Flesh | constant-velocity extrapolation |
| Dark halo | per-point uncertainty — the spread across independently sampled futures |
| Error vectors | prediction to ground truth, per point |

Only frames at or past Module 4's `split_frame` are exported, so every origin is
one the model never trained on.

## Why constant velocity is on screen

Because it wins at one frame ahead. Hiding the baseline the method loses to is
the fastest way to lose a reviewer, and the crossover — physics ahead at k=1,
the learned prior ahead by k=3 — is the actual finding. The horizon selector
exists so it can be demonstrated rather than asserted.
