# tools

## preflight.py

Static check before spending GPU hours. A `NameError` forty minutes into a
Colab run is expensive and entirely avoidable.

```bash
python3 tools/preflight.py "FYP_MultiVideo_Pipeline_(11).ipynb" \
    module5_diffusion.py module6_prediction.py \
    module7_baselines.py module8_hpo.py
```

Pass the notebook alongside the modules. These modules run as notebook cells,
so names set in an earlier cell — `DRIVE_BASE` and friends — are legitimately
undefined in the `.py` file. The checker harvests those from the notebook
rather than carrying an allowlist that would go stale.

**What it catches:** names read but never bound — typos, renames applied in
one place and not another, helpers deleted while still referenced.

**What it does not catch:** shape mismatches, missing files on Drive, OOM, or
anything that depends on values rather than names. A clean report is weak
evidence; a hit is worth reading.

It is deliberately conservative and prefers silence to a false alarm, since a
checker that cries wolf stops being run. Verified against an injected typo:
renaming `ctx` to `cxt` in one call site is reported at the right line.
