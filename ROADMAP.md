# From the current system to live intraoperative prediction

**Project S26-017-R-GASR · assessment and plan · 20 October 2026**

This document answers two questions. How far is the system from reconstructing and
predicting tissue deformation inside a live operating theatre, and what is the shortest
honest route there.

> **Status, 11 November 2026.** Module 7 has since run, and it changed the diagnosis in
> §1.1. The predictor did not merely mispredict magnitude — it had collapsed to the
> conditional mean, scoring 1.8% *worse* than assuming nothing moves and losing to a
> 136k-parameter MLP. The cause was found and is not a tuning problem: the context
> encoder was fed absolute coordinates, in which one frame of displacement is 0.74% of
> the input magnitude. It now receives motion directly. **The execution sequence lives in
> [RUNBOOK.md](RUNBOOK.md); this document remains the longer-range assessment, and §2
> onward still stands.**

Everything below is grounded in what the repository actually contains. Where a number is
measured it is cited; where it is an estimate it says so.

---

## 1. Where the system actually stands

The pipeline is complete end to end and produces measured results. That is a real
achievement and it is not what limits deployment. Six axes matter, and they are not
equally far along.

### 1.1 Prediction quality — not yet usable, but diagnosed

| | |
|---|---|
| Accuracy at k = 1–3 | **Loses to constant velocity**, on all ten videos |
| Accuracy past rollout step ~4 | **Beats constant velocity** — QT2 crossover |
| Direction | Learned: cosine +0.57 at t+1, decaying to zero by t+5 |
| Magnitude | Not learned — this is the whole of the deficit |
| Uncertainty | Calibrated, r = 0.69 against realised error |

Two corrections are implemented and have never been evaluated: magnitude calibration at
inference (no retraining needed) and the constant-velocity residual target (one retrain).

**Superseded, 11 November.** Module 7 placed the model 8th of 11 methods at MPE
0.009281 against a zero-motion baseline of 0.009117, tracking that baseline to within
2% at every horizon — the signature of collapse to the conditional mean, not of a
scale error. Neither correction above addresses the cause. The cause was the input
representation, and all four modules now feed the encoder motion relative to the most
recent frame at unit variance, with the absolute frame retained as an anchor. The
residual target still matters, for a different reason: it moves the degenerate floor
from the static baseline up to constant velocity.

**The important asymmetry:** the system already knows where it wins. Constant velocity is
better for one to three frames; the diffusion model is better from roughly four frames
out, because straight-line extrapolation diverges and the model does not. A clinically
useful horizon is probably longer than three frames, not shorter. This should shape the
product, rather than being treated as a problem to eliminate.

### 1.2 Reconstruction latency — the critical path

Reconstruction is **20,000 optimisation iterations, about 15 minutes per scene on a T4**
(`fyp-1/EndoGaussian_4D_Colab.ipynb`, Step 5).

This is not a tuning problem. Per-scene optimisation is structurally incompatible with
live operation: the scene must already exist before anything can be predicted into it, and
it cannot exist until the video has been observed and optimised over. No amount of faster
hardware converts a fifteen-minute per-scene fit into an intraoperative capability.

**This, not prediction accuracy, is the single thing standing between the project and its
stated goal.** It requires a different class of method — feed-forward or amortised
reconstruction that infers Gaussians in one pass rather than optimising them per scene.

### 1.3 Causality — the predictor is ready, the pipeline is not

| Stage | Can it run forward in time? |
|---|---|
| M1 frames + masks | Yes, per frame |
| M2 4D Gaussian reconstruction | **No** — needs the whole video |
| M3 trajectory extraction | **No** — needs the finished reconstruction |
| M4 windowing | **No** — needs the whole trajectory |
| M5/M6 prediction | **Yes** — four frames in, k frames out |

The predictor is already causal. Everything upstream is batch. The system as composed
cannot be run forward in time even if every component were instant.

### 1.4 Frame timing — a known correctness defect

`module6_prediction.py:1075` records it plainly: Module 1 discards blurry frames, so
consecutive trajectory frames are **not evenly spaced in real time, yet every module
downstream treats them as if they were.**

Timing recovery has already failed in practice — the Module 6 timeline figure reports
*"frame timing unavailable: saved frames do not align with Video10.mp4 (matched 35/269)."*

This matters twice over. It may be suppressing accuracy now, and in a live system you have
true timestamps, so the model must accept a real time delta rather than an index. The code
already names the fix: resample to uniform time, or feed the gaps to the model.

### 1.5 Data scale — clinically trivial

Ten videos, roughly 219 frames each: on the order of 2,200 frames, one procedure type, one
institution, no patient diversity. The overfitting measured after epoch 60 is exactly what
a corpus this size predicts. Any clinical claim needs two to three orders of magnitude
more, across sites and surgeons.

### 1.6 Clinical and regulatory — not started

No IRB approval, no prospective data, no latency budget agreed with surgeons, no failure
mode and effects analysis, no software-as-a-medical-device classification. A predictive
intraoperative display is plausibly Class IIa/IIb in the EU and Class II in the US. This
axis is measured in years and is not shortened by engineering.

---

## 2. The honest summary

There are three gaps, not one, and they are independent:

1. **Accuracy** — months. The diagnosis is done and two fixes are written.
2. **Live reconstruction** — a research programme. Different architecture, not a faster
   version of this one.
3. **Clinical validation** — years, and gated on people and process rather than code.

A project that fixes only the first still cannot run in an operating theatre. A project
that fixes the first two has a real-time research prototype, which is a legitimate and
publishable end point and is the right target for this work.

**Recommended end goal for the project as scoped: a real-time research prototype on
recorded surgical video, running causally at interactive rate, with calibrated
uncertainty.** Live human use is the goal of the programme, not of the FYP.

---

## 3. The plan

Ordered by cost. Each phase has a kill criterion, so effort stops when evidence says it
should rather than when enthusiasm runs out.

### Phase 0 — Spend the compute you have already paid for
**Effort: days. Cost: GPU hours only. No new code.**

Run Modules 7 and 8. Both are written, tested and idle. This fills four empty tables,
makes the paper's abstract true, and tells you which hyperparameters actually matter
before you spend time tuning the wrong ones.

*Kill criterion: none. This is pure upside and blocks the paper.*

### Phase 1 — Cash in the two implemented corrections
**Effort: 1–2 weeks.**

Evaluate magnitude calibration (no retraining) and the constant-velocity residual (one
retrain). The mathematics says any non-zero directional agreement should produce an
improvement over naive once the scale is solved for, and you measure +0.57.

*Kill criterion: if the calibrated model still loses to constant velocity at k = 1, the
formulation is wrong rather than under-tuned. Stop polishing it and go to Phase 2 and 4.*

### Phase 2 — Fix frame timing
**Effort: 2–3 weeks.**

Resample trajectories to uniform time, or pass the real time delta to the model as a
conditioning input alongside the horizon. The second is strictly better and is what a live
system will need anyway, because frames arrive on a real clock.

Do this *before* collecting more data — otherwise the new data inherits the defect.

*Kill criterion: none; this is a correctness fix regardless of whether it recovers
accuracy.*

### Phase 3 — Make the evaluation honest at scale
**Effort: 1 month, mostly waiting on data.**

Leave-one-video-out, then grow the corpus. Prioritise *variety* over volume: a second
procedure type is worth more than ten more clips of the same one.

### Phase 4 — Replace per-scene reconstruction
**Effort: the real research. One to two quarters.**

This is where live is won or lost. The direction is feed-forward reconstruction: infer the
Gaussian field in a single pass from the stream rather than optimising it per scene.
Related lines to evaluate before building anything — feed-forward 4D Gaussian methods from
the autonomous-driving literature, and the surgical Gaussian forecasting work that
appeared this year.

Run a two-week literature and feasibility spike first and write down the latency budget
you are targeting. Do not start implementing until that number exists.

*Kill criterion: if no feed-forward method reaches usable fidelity on endoscopic data
within the spike, the live goal is out of scope for this project and should be stated as
future work rather than pursued.*

#### Phase 4 spike — first findings (21 Oct 2026)

Two results change the shape of this phase. Neither is yet verified against a paper PDF;
both come from published abstracts and should be confirmed before any work is committed.

**1. Our reconstruction is far off the published operating point.**
EndoGaussian — the method this pipeline is built on — reports training *within two minutes
per scene* and rendering at 195 FPS. We run 20,000 iterations for about fifteen minutes.
EndoPlanar (MICCAI 2025) reports 307.5 FPS at 34.51 dB.

That is roughly a sevenfold gap on reconstruction time, and closing it is configuration
work rather than research. **This becomes Phase 4a and should happen before any
feed-forward work is considered**, because it is cheap, it is measurable in days, and it
changes what "live" would even require of the rest of the system.

**2. The monocular constraint is what makes feed-forward hard, and it may be
self-imposed.**
The feed-forward endoscopic reconstruction that exists is largely stereo. Endo-E2E-GS
builds Gaussian fields from *a single stereo pair* with no scene-specific optimisation,
reporting 38.874 dB on ENDONERF and SCARED — exactly the capability Phase 4 needs, but
from stereo input. (Its page returned 403 and a follow-up search did not surface the
preprint, so treat these numbers as unconfirmed.)

This forces a decision the project has not yet made explicitly:

| | Monocular | Stereo |
|---|---|---|
| Deployment target | conventional laparoscopy | robotic surgery (da Vinci is stereo) |
| Feed-forward reconstruction | little published work | available now |
| Our current novelty claim | rests on this | weakened |
| Route to live | research programme | engineering |

Robotic platforms already provide stereo. If the intended destination is robotic surgery,
the monocular constraint is self-imposed, and relaxing it converts Phase 4 from a research
programme into an integration task. If the destination is conventional laparoscopy, the
constraint is real and must be paid for.

**This is a question for the supervisor, not a technical choice to be made quietly.** It
determines whether "live" is reachable inside this project or belongs to the programme
that follows it.

### Phase 5 — Causal re-architecture
**Effort: 1 quarter, and only meaningful after Phase 4.**

Convert M3 and M4 to sliding-window streaming, hold a bounded history, drop the replay
gate in favour of an online integrity check. The predictor needs no change — it is already
causal.

### Phase 6 — Clinical
**Effort: years. Start the paperwork during Phase 3, not after Phase 5.**

IRB, prospective recording, a latency and accuracy budget agreed with surgeons in advance,
FMEA, regulatory classification. The long pole is approval, not code, which is why it
starts early and runs in parallel.

---

## 4. The efficient path, stated bluntly

If the aim is the shortest route to something that runs live and is genuinely useful:

**Ship the uncertainty, not the accuracy.** The calibrated spread (r = 0.69) is the part of
this work no baseline provides and no competing method offers. A live display that shows
constant-velocity extrapolation *annotated with your confidence estimate* would be useful
today, is cheap to build, and is honest about what works. The diffusion model then earns
its place at the longer horizons where QT2 already shows it wins.

That reframes the remaining work from "make the model beat a two-line formula at one frame"
— which it may never do, and which may not matter — to "establish the horizon at which
learned prediction pays, and deliver confidence everywhere." The second is defensible,
achievable within the project, and publishable whether or not the first ever lands.

---

## 5. What to do now

Superseded by [RUNBOOK.md](RUNBOOK.md), which carries the ordered sequence, the
pass/fail thresholds and the submission checklist. In brief: rebuild Module 4, retrain
Module 5 once under the corrected encoder, evaluate with Modules 6 and 7, run the
Module 8 ablations, export the Module 9 demonstrator, rebuild the paper.

Two items from this document are *not* in the runbook, because they are decisions
rather than steps, and they outlive the FYP:

1. **Put the monocular-versus-stereo question to Dr Jamil** (§4, Phase 4 spike). It
   decides whether live operation is reachable inside this project or belongs to the
   programme that follows it.
2. **Close the reconstruction gap** (Phase 4a). EndoGaussian reports two minutes per
   scene; we take fifteen. Sevenfold, for configuration work rather than research, and
   it remains the single largest obstacle between this system and live operation —
   prediction accuracy never was.
