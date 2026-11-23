# Paper build

IEEE conference manuscript. Every number comes from the pipeline's own JSON output —
nothing is typed into `main.tex`.

## Build

```bash
python3 make_tables.py --drive /content/drive/MyDrive/EndoGaussian
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

`make_tables.py` reads whatever result files exist and writes two files:

| File | Contents | Where it is used |
|---|---|---|
| `generated_macros.tex` | `\newcommand` macros holding individual numbers | preamble, so the abstract can use them |
| `generated_tables.tex` | the five result tables | the Results section |

Missing result files produce a clearly-marked *"Pending: run Module N"* placeholder rather
than a LaTeX error, so the manuscript compiles and can be drafted while experiments run.

## Sources

| Module | Produces | Feeds |
|---|---|---|
| 4 | `module4_metadata.json` | dataset size macros |
| 6 | `module6_metrics.json` | calibration correlation, rollout growth exponent |
| 7 | `module7_comparison.json` | Table I (comparison against baselines) |
| 8 | `module8_hpo.json` | Tables II–V (inference cost, ablations, context, sensitivity) |

## Macros available to the prose

`\numVideos` `\numAnchors` `\ctxWindow` `\predSteps` `\numTrainWindows` `\numValWindows`
`\calibCorr` `\growthExp` — already used in the text.

`\bestMethod` `\bestMPE` `\oursMPE` `\oursCD` `\oursParams` `\naiveMPE` — defined but not
yet used. They exist so the Results prose can state outcomes once Module 7 has run, e.g.
`our method reaches \oursMPE{} against \naiveMPE{} for the zero-motion baseline`. Write
such sentences only after seeing the table; they are deliberately not pre-written.

## Switching to Springer LNCS

Replace the `\documentclass` and package block with `\documentclass{llncs}`, remove
`\IEEEoverridecommandlockouts` and the `IEEEauthorblock` wrappers, and change the
bibliography style to `splncs04`. The body needs no other change.

## Before submission

- `refs.bib` entries were taken from the papers' listing pages. Verify page numbers and
  published venues against the official proceedings; entries noted `NEEDS-CHECK` have a
  detail that could not be confirmed.
- The Results prose describes what each table shows but states no outcome. Fill in the
  findings once the runs are complete.
