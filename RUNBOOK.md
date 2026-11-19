# Runbook — finishing the FYP

Everything that does not need a GPU is done. This is the remaining sequence, in order,
with what to check at each step and what to do when a step disagrees with the plan.

Nothing here is exploratory. The configuration is settled; see §6 for why each number is
what it is, and change none of them without a reason written down.

---

## Where this stands — 13 November 2026

**Step 2 is done and it worked.** The retrain under the corrected context encoder
completed 100/100 epochs in 11h15m at ~409 s/epoch, cosine annealed to 1.29e-06.

| On the 649-window subset | CD |
|---|---|
| Ours, 4-sample mean | **0.012984** |
| Constant velocity | 0.014151 |
| Zero motion | 0.016195 |

**+19.8% against the static baseline and +8.2% against constant velocity.** The
predictor beats the estimator it previously lost to. The 0.74% input-scale defect was
the cause, and feeding motion directly removed it.

Two notes on that run, both since fixed in this repository:

- It ended with a CUDA OOM in **QT4**, the DDIM step-count diagnostic — the one sampling
  call that did not batch. QT1 to QT3 had already passed. Nothing was lost that Module 6
  does not recompute, and the batching is fixed.
- It reported *"run is incomplete"* because the session budget and the final epoch fired
  together, so a finished run was labelled partial and scored on the subset rather than
  the full validation set. Also fixed.

**Re-running the Module 5 cell now trains nothing** (`start_epoch >= TOTAL_EPOCHS`) and
goes straight to the quality tests on the **full** validation set — about 20 minutes for
the headline number the paper quotes. Worth doing once, with the fixes above.

> **Do not change `WINDOW_SIZE`, `MICRO_BATCH` or `TOTAL_EPOCHS` in Module 5.** They now
> match the checkpoint exactly, confirmed by its parameter count of 16,318,467. Changing
> the window alters `temporal_pe`'s shape, which retires the checkpoint and restarts
> training from epoch 0 — eleven hours gone, silently, with no error.

Next: step 3.

---

## 0. Before the first run — 2 minutes

```bash
# in Colab, after the Drive mount cell
cp /content/drive/MyDrive/EndoGaussian/module5/diffusion_ckpt_best.pth \
   /content/drive/MyDrive/EndoGaussian/module5/diffusion_ckpt_best_delta_ep60.pth
```

**Why:** that checkpoint is the only thing that makes the current Module 7 comparison
table reproducible. The retrain overwrites it. The architecture change retires it
automatically, so nothing will stop you losing it except doing this.

Then get the current notebook into Colab. The repository copy is the source of truth;
Colab reads its own copy from Drive and will not pick up these changes by itself.
Upload `FYP_MultiVideo_Pipeline_(11).ipynb`, or paste cells **34, 36, 42, 44, 46, 47**.

> Paste all of them or none. Modules 5 to 8 share the model definition and are fed the
> same checkpoint. A partial update does not error — it silently evaluates the weights
> on an input representation they never saw, which is the exact defect being fixed.

Pre-flight before spending GPU time:

```bash
python3 tools/preflight.py module5_diffusion.py module6_prediction.py \
    module7_baselines.py module8_hpo.py module9_demo_export.py \
    "FYP_MultiVideo_Pipeline_(11).ipynb"
```
Expect `PASS: 0 suspect name(s)`.

---

## 1. Module 4 — rebuild the dataset — ~15 min

**Cell 32 first, then cell 34 — and never cell 32 alone.**

This ordering is not optional and it is the one trap in the whole pipeline. Cell 32 is the
original Module 4; cell 34 is `module4_improved`. Cell 34 reads its workspace list from the
`metadata.json` that cell 32 writes, so in a fresh runtime cell 32 has to run first. Cell 32
used to write `train_inputs.npy`, `val_inputs.npy` and the two target arrays itself, at
`W=4` — the same filenames cell 34 writes at `W=32`. So running cell 32 after cell 34, or
running cell 32 and stopping, silently reverted the dataset to a 4-frame window while
`module4_metadata.json` still said 32.

That is fixed: cell 32 now writes those four arrays under a `legacy_` prefix, and Modules 5,
6 and 7 each refuse to start if the context arrays and the metadata disagree. If you ever
see `INCONSISTENT MODULE 4 ARRAYS`, the fix is to re-run cell 34. Nothing is lost — the
trained checkpoint never reads these files.

**Check:** the printed window count is around 6,500 total, down ~11% from 7,371. A much
larger drop means a video is shorter than `W + max(k)` and was skipped — find out which.

---

## 2. Module 5 — train — already done, do not repeat

Cell 36. `WINDOW_SIZE = 32`, `MICRO_BATCH = 2`, `TOTAL_EPOCHS = 100`,
`TARGET_MODE = 'cv_residual'`. **These match the trained checkpoint exactly and must not be
changed** — see the warning in *Where this stands* above.

Training finished on 13 November 2026 at epoch 100/100. Re-running the cell trains nothing
(`start_epoch >= TOTAL_EPOCHS`) and goes straight to the quality tests, which is worth ~20
minutes once: a completed run scores the **full** validation set rather than the 649-window
subset, and that full-set number is the one the paper quotes.

**Check at the top of any re-run:**

| Line | Expect |
|---|---|
| `Context` | `last 32 of the 32 frames Module 4 built` — or the line absent, which is also correct at W=32 |
| `CTX_SCALE` | a number in the tens or low hundreds, not ~1 |
| `Target` | `cv_residual` |
| checkpoint | **resumed at epoch 100** — if it says *retired and restarted from epoch 0*, stop the cell immediately: the config no longer matches the checkpoint |
| epoch time | ~200 s. At ~800 s, `MICRO_BATCH` did not take; stop and fix |

**Check at the end:** the run reaches epoch 150 and the cosine schedule anneals fully.
`CKPT_BEST` keeps the best epoch regardless of where it occurred.

**If it OOMs:** halve `MICRO_BATCH` to 4 and restart. Do not lower `BATCH_SIZE`;
accumulation preserves the effective batch and the training dynamics with it.

**Do not change `TOTAL_EPOCHS` partway.** The resume path refuses a longer horizon,
because lengthening re-heats already-annealed weights. Shortening is allowed.

---

## 3. Modules 6 and 7 — evaluate — ~1 h

Cells 42 and 44. Module 6 writes the metrics and the calibration; Module 7 writes the
comparison against all ten baselines.

### The pass/fail bar

Module 7 prints a Mean Point Error for `Ours — diffusion`. Compare it against these,
which are measured, not estimated:

| Threshold | Meaning |
|---|---|
| **> 0.009117** | still worse than assuming nothing moves. The fix did not work |
| **< 0.007980** | beats constant velocity. The model works |
| **< 0.007579** | beats the deterministic Transformer. The model leads the table |

A degenerate model under `cv_residual` now reproduces constant velocity, so anything
above 0.008 means something is wrong rather than merely weak.

**If it lands above 0.009117:** stop. Do not retrain again on instinct. Go to step 4,
which is cheap and tells you which of the remaining variables is responsible.

---

## 4. Module 8 — ablations

**Studies A and B are done** (19 November 2026, 14.3 h). Their results were recovered by
transcription from the console, because Study C hit an out-of-memory error before the
cell's single `json.dump` could run. Before re-running the cell:

```bash
# copy results/module8_hpo.json from this repo to Drive, once
cp results/module8_hpo.json "$DRIVE_BASE/module8/module8_hpo.json"
```

Module 8 now reloads that file at startup and writes each study back **as it finishes**, so
a crash costs one study rather than all of them. `RUN_A` and `RUN_B` are therefore `False`
in the committed cell; the loaded results still appear in the output and on the figure.

### What A and B found

| Study | Result |
|---|---|
| **A** — sampling cost vs accuracy | Best **0.007702 MPE** at 5 DDIM steps × 8 samples. The knee is **2×8 at 16 forward passes**, 2.5× cheaper for 0.8% more error. More steps is *worse*: every 50-step cell is beaten by every 2-step cell. |
| **B** — context length | **W=8 is best (0.008204). W=32 is worst (0.009442)** — W=4 → W=32 is **9.2% worse**, not better. |

Against the bars in step 3, over Study A's 20 cells: **20/20 beat the static baseline**,
**12/20 beat constant velocity** (including the best and the knee), **0/20 beat the
deterministic Transformer**.

So the fix worked and the model is a real predictor — but on this evidence it does not lead
the table. Report that.

### Study B contradicts the deployed configuration

The trained model is W=32, and the cheap backbone says W=8 would have been better. Two
honest caveats: the backbone is not the diffusion model, and it does not use the relative
context encoding that made W=32 trainable at all. It is evidence, not proof. **Do not
retrain on it** — there is no time, and Study A already shows the deployed checkpoint
beating constant velocity. Report it as the ablation result it is.

### Study C — the one still to run

Cell 46 with `RUN_A=False, RUN_B=False, RUN_C=True`. It trains 16 backbones. Study B
measured the per-run cost exactly, so the budget is arithmetic rather than a guess:

| Window | 40 epochs (Study B) | 16 runs × 12 epochs |
|---|---|---|
| W=32 (deployed) | 10,242 s | **~13.7 h** |
| W=8 (best) | 3,389 s | ~4.5 h |
| **W=4 (used)** | 2,156 s | **~2.9 h** |

So it runs at `C_WINDOW = 4`, `C_EPOCHS = 12`. The window is cut harder than the schedule
on purpose: a short schedule systematically favours fast-converging settings, which is
precisely what the optimiser and learning-rate rows are measuring, so shrinking epochs
would bias the very ranking the study exists to produce. Study C reports sensitivity **at a
fixed budget**; its absolute MPE is not comparable with Study A or Module 7, and the output
JSON records that. The real estimate is printed before any training starts — read it.

`RUN_D` retrains the diffusion model once per ablation. Leave it `False` unless there is a
spare day.

---

## 5. Module 9 and the viewer — ~15 min

Cell 47, immediately after Module 6 **in the same runtime** — it reuses Module 6's
objects rather than re-deriving the prediction path.

```bash
# copy module9_demo/demo_data.json from Drive into demo/
python3 -m http.server -d demo 8000
```

The header badge must read **measured**. If it reads **synthetic**, the JSON is not
beside `index.html` or the page was opened over `file://`.

---

## 6. The paper — minutes

```bash
./paper/build.sh /content/drive/MyDrive/EndoGaussian    # or your Drive path
```

Every table is generated from the result JSONs. Anything not yet produced prints as
`[pending]`, so the PDF always compiles. After steps 1–5 there should be no `[pending]`
left except in the reconstruction table if Module 2 was not re-run.

The headline sentence in the results section is **generated** from
`module7_comparison.json`, not typed: whether the predictor beats the static baseline,
leads the table, or does neither is derived from the file. You cannot ship a paper whose
text contradicts its own table, and you do not have to remember to update it.

**One consistency obligation.** Section 5 now describes the relative-context encoder,
so the results must come from a run that used it. Do not ship a PDF whose method
section and tables disagree — regenerate the tables, or say in the caption which
formulation produced them.

---

## Submission checklist

- [ ] `module7_comparison.json` exists and the paper's Table 1 is populated
- [ ] No `[pending]` in `paper/main.pdf` other than knowingly accepted ones
- [ ] `demo/demo_data.json` present; the viewer header reads **measured**
- [ ] `paper/refs.bib` — the two entries marked `NEEDS-CHECK` have real authors, not
      `{Anonymous}`. Both are recent arXiv preprints and must be verified against the
      actual PDF before submission
- [ ] `ROADMAP.md` reflects what was actually run
- [ ] Repository pushed

---

## Why the configuration is what it is

| Setting | Value | Reason |
|---|---|---|
| Module 4 `WINDOW_SIZE` | 32 | Built wide once. Lets Study B sweep 4–32 with no rebuild |
| Module 5 `WINDOW_SIZE` | 8 | `MICRO_BATCH × W = 64` holds activation memory at the footprint already known to fit a T4 |
| `MICRO_BATCH` | 8 | 8 × 512 × 8 = 32,768 token positions, the same as the old W=4 setting |
| `TOTAL_EPOCHS` | 150 | ~200 s/epoch → ~8.3 h. A complete, fully annealed run inside one session beats a better one spread over four |
| `TARGET_MODE` | `cv_residual` | Degenerate output reproduces constant velocity instead of zero |
| Context encoding | relative + anchor | The defect. Displacement was 0.74% of the coordinates carrying it |

The model truncates its own context, so changing the trained window means changing one
number in Module 5 and nothing else.
