#!/usr/bin/env python3
"""Generate module5_quality_tests.py: a standalone quality-test cell.

Why this is generated rather than written by hand
-------------------------------------------------
A hand-written quality-test cell would be a fifth copy of ConditionalDenoiser and a
second implementation of evaluate_cd, ddim_sample and the baselines. This project has
already paid for that once: module5_eval_sweep.py was missed when the context encoder
moved from 3 absolute coordinates to 6 relative channels, and it sat in the repo unable
to load the current checkpoint at all.

So the model, the sampler, the metrics and the six tests are lifted VERBATIM out of
module5_diffusion.py by source span. Nothing is retyped, so nothing can drift at
generation time. When Module 5 changes, re-run this script.

Usage:  python3 tools/make_qt_cell.py
"""
import ast
import io
import os
import sys

SRC = 'module5_diffusion.py'
OUT = 'module5_quality_tests.py'

# Definitions the six tests transitively need. Computed, not guessed: see the free-name
# analysis in tools/README.md.
DEFS = ['representative_subset_indices', 'GeomMLP', 'sinusoidal_embedding',
        'ConditionalDenoiser', 'cosine_beta_schedule', 'pred_x0_from_v',
        'pred_eps_from_v', 'ddim_sample', 'predict_mean_oneshot', 'chamfer_distance',
        'to_physical', 'const_velocity_pred', 'base_pred', 'evaluate_cd']

# The QT block: QT1's opener through the final verdict print.
QT_FIRST = 'if len(train_losses) >= 10:'
QT_LAST = '✅ ALL QUALITY TESTS PASSED'


def main():
    if not os.path.exists(SRC):
        sys.exit(f'{SRC} not found -- run this from the repository root.')
    src = io.open(SRC, encoding='utf-8').read()
    lines = src.splitlines()
    tree = ast.parse(src)

    found = {n.name: n for n in tree.body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in DEFS}
    missing = [d for d in DEFS if d not in found]
    if missing:
        sys.exit(f'{SRC} no longer defines, at module level: {missing}\n'
                 f'Module 5 has been restructured; update DEFS in this script.')

    def span(node):
        """Source for a def, including the comment block immediately above it."""
        start = node.lineno - 1
        while start > 0 and (lines[start - 1].lstrip().startswith('#')
                             or lines[start - 1].startswith('@')):
            start -= 1
        return '\n'.join(lines[start:node.end_lineno])

    blocks = [span(found[d]) for d in DEFS]

    # QT block, located by content so a shifted line number cannot silently truncate it.
    try:
        qt_start = next(i for i, l in enumerate(lines) if l.rstrip() == QT_FIRST)
        qt_end = next(i for i, l in enumerate(lines) if QT_LAST in l)
    except StopIteration:
        sys.exit(f'Could not locate the quality-test block in {SRC}. Looked for a line '
                 f'equal to {QT_FIRST!r} and a line containing {QT_LAST!r}.')
    if qt_end <= qt_start:
        sys.exit(f'Quality-test block looks inverted: opener at {qt_start + 1}, '
                 f'verdict at {qt_end + 1}.')
    qt_block = '\n'.join(lines[qt_start:qt_end + 1])

    out = PREAMBLE + '\n\n'.join(blocks) + '\n\n' + MIDDLE + qt_block + '\n' + EPILOGUE
    io.open(OUT, 'w', encoding='utf-8').write(out)

    n_def_lines = sum(b.count('\n') + 1 for b in blocks)
    print(f'wrote {OUT}')
    print(f'  {len(DEFS)} definitions lifted verbatim ({n_def_lines} lines)')
    print(f'  quality-test block: {SRC} lines {qt_start + 1}-{qt_end + 1} '
          f'({qt_end - qt_start + 1} lines)')
    print(f'  total: {out.count(chr(10)) + 1} lines')


PREAMBLE = '''# ════════════════════════════════════════════════════════════════════
# MODULE 5 — QUALITY TESTS ONLY  (no training, read-only)
#
# GENERATED FILE -- do not edit. Produced by tools/make_qt_cell.py, which lifts the
# model, the sampler, the metrics and the six tests verbatim out of
# module5_diffusion.py. Edit Module 5 and re-run that script.
#
# Runs QT1-QT6 against the checkpoint already saved on Drive, on the FULL validation
# set. It does not train, and it never writes a checkpoint: there is no torch.save and
# no os.rename anywhere in this file, so it cannot cost you the run that produced the
# weights.
#
# Every architectural and scaling constant is read FROM THE CHECKPOINT, not from
# constants here. That matters: if this file declared WINDOW_SIZE itself and disagreed
# with the checkpoint, the weights would refuse to load -- or worse, load into a model
# whose normalisation differs and report confident nonsense. The checkpoint is the
# single source of truth about the model it contains.
#
# Runtime: roughly 20-25 minutes. QT2 alone samples FINAL_SAMPLES trajectories x
# FINAL_STEPS DDIM steps for every validation window.
# ════════════════════════════════════════════════════════════════════
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, os, math, json, time
from tqdm import tqdm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# DRIVE_BASE and the Drive mount normally come from the setup cell at the top of the
# notebook. Recovered here so this cell runs standalone in a fresh runtime.
if 'DRIVE_BASE' not in globals():
    DRIVE_BASE = '/content/drive/MyDrive/EndoGaussian'
    if not os.path.isdir(DRIVE_BASE):
        from google.colab import drive
        drive.mount('/content/drive')
    assert os.path.isdir(DRIVE_BASE), f'Drive mounted, but {DRIVE_BASE} is absent'
    print(f'DRIVE_BASE recovered locally -> {DRIVE_BASE}')

COMBINED_DIR = f'{DRIVE_BASE}/module4_combined'
MODULE5_DIR  = f'{DRIVE_BASE}/module5'
CKPT_BEST    = f'{MODULE5_DIR}/diffusion_ckpt_best.pth'
CKPT_LATEST  = f'{MODULE5_DIR}/diffusion_ckpt.pth'

# Which checkpoint to score. 'best' is the one the paper should quote; 'latest' is
# wherever training stopped. They differ whenever the best validation score was not on
# the final epoch.
WHICH        = 'best'              # 'best' | 'latest'

FINAL_STEPS   = 20                 # DDIM steps for QT2 -- matches Module 5's final QT2
FINAL_SAMPLES = 4                  # trajectories averaged per window
EVAL_ROWS     = 64                 # windows * n_samples resident at once
SEED          = 2026

torch.manual_seed(SEED)
np.random.seed(SEED)

print('=' * 70)
print('  MODULE 5 — QUALITY TESTS (read-only, no training)')
print('=' * 70)

# ── Checkpoint ───────────────────────────────────────────────────────
_path = CKPT_BEST if WHICH == 'best' else CKPT_LATEST
if not os.path.exists(_path):
    _other = CKPT_LATEST if WHICH == 'best' else CKPT_BEST
    assert os.path.exists(_other), (
        f'No checkpoint in {MODULE5_DIR}. Looked for {os.path.basename(CKPT_BEST)} and '
        f'{os.path.basename(CKPT_LATEST)}. There is nothing to evaluate.')
    print(f'  ⚠️  {os.path.basename(_path)} is absent; using '
          f'{os.path.basename(_other)} instead.')
    _path = _other

# Drive buffers writes, and this repository has already lost one large file that way, so
# say how big it is rather than letting a short read surface as a cryptic unpickling error.
print(f'  Checkpoint    : {os.path.basename(_path)} '
      f'({os.path.getsize(_path) / 1e6:.1f} MB)')
try:
    ck = torch.load(_path, map_location=DEVICE, weights_only=False)
except Exception as _e:
    raise SystemExit(
        f'\\n  ✗ {_path} failed to load: {type(_e).__name__}: {_e}\\n'
        f'    This cell does NOT rename or replace it -- it is read-only. If Drive\\n'
        f'    truncated the write, the size above will be short of the model size.')

arch = ck.get('arch_cfg') or {}
cfg  = ck.get('config') or {}
assert arch, ('This checkpoint carries no arch_cfg, so the architecture it was trained '
              'with cannot be recovered. It predates horizon conditioning and cannot be '
              'evaluated by this cell.')

# Every constant below comes from the checkpoint. Nothing here second-guesses it.
WINDOW_SIZE   = int(arch['WINDOW_SIZE'])
LATENT_DIM    = int(arch['LATENT_DIM'])
HIDDEN_DIM    = int(arch['HIDDEN_DIM'])
N_HEADS       = int(arch['N_HEADS'])
N_TF_LAYERS   = int(arch['N_TF_LAYERS'])
N_ANCHOR      = int(arch['N_ANCHOR'])
T_DIFF        = int(arch['T_DIFF'])
MAX_PRED_STEP = int(arch['MAX_PRED_STEP'])
TARGET_MODE   = arch.get('TARGET_MODE', 'delta')
CTX_MODE      = arch.get('CTX_MODE', 'abs3')
DATA_SCALE    = float(cfg['DATA_SCALE'])
CTX_SCALE     = float(cfg.get('CTX_SCALE', 1.0))
DROPOUT       = 0.1                # eval-mode only; affects no parameter shape
# Nothing is trained here. BATCH_SIZE is read only because QT3, QT4 and QT6 use it to
# size their shared diagnostic subset, and taking Module 5's own value keeps those tests
# scoring the same number of windows they did during the run.
BATCH_SIZE    = int(cfg.get('BATCH_SIZE', 32))

assert CTX_MODE == 'rel6', (
    f"This checkpoint was trained with CTX_MODE={CTX_MODE!r}, but the context encoder in "
    f"this file takes 6 relative channels ('rel6'). Its weights have a 3-channel first "
    f"layer and different semantics throughout, so loading them here would report "
    f"confident nonsense rather than fail. Evaluate it with the Module 5 revision that "
    f"trained it.")

print(f'  Trained to    : epoch {ck.get("epoch", "?")}  |  best subset CD '
      f'{ck.get("best_cd", float("nan")):.6f}')
print(f'  Architecture  : W={WINDOW_SIZE}  latent={LATENT_DIM}  hidden={HIDDEN_DIM}  '
      f'heads={N_HEADS}  layers={N_TF_LAYERS}  anchors={N_ANCHOR}')
print(f'  Target mode   : {TARGET_MODE}' + ('   (network output is the correction to '
      'constant velocity)' if TARGET_MODE == 'cv_residual' else ''))
print(f'  DATA_SCALE    : {DATA_SCALE:.3f}   CTX_SCALE: {CTX_SCALE:.3f}'
      '   (both inverted before every reported metric)')

# ── Validation data, full set ────────────────────────────────────────
val_inp   = np.load(f'{COMBINED_DIR}/val_inputs.npy').astype(np.float32)
val_tgt   = np.load(f'{COMBINED_DIR}/val_targets.npy').astype(np.float32)
val_dnorm = np.load(f'{COMBINED_DIR}/val_disp_norm.npy').astype(np.float32)
val_ps    = np.load(f'{COMBINED_DIR}/val_pred_step.npy').astype(np.int64)

# Provenance, the same check Modules 5-8 make. *_inputs/_targets are written by the
# notebook's original Module 4 cell too, at a different window, while *_disp_norm and
# *_pred_step come only from module4_improved -- so the two can describe different
# datasets, with different window counts.
assert len(val_inp) == len(val_ps), (
    f'val_inputs.npy holds {len(val_inp):,} windows but val_pred_step.npy holds '
    f'{len(val_ps):,}. The arrays in {COMBINED_DIR} come from two different Module 4 '
    f'runs -- re-run Module 4 (module4_improved), then re-run this cell.')

_W_DATA = val_inp.shape[1]
assert _W_DATA >= WINDOW_SIZE, (
    f'The checkpoint needs {WINDOW_SIZE} context frames but Module 4 emitted '
    f'{_W_DATA}. Re-run Module 4 with a WINDOW_SIZE of at least {WINDOW_SIZE}.')
if _W_DATA > WINDOW_SIZE:
    val_inp = val_inp[:, -WINDOW_SIZE:]
    print(f'  Context       : last {WINDOW_SIZE} of the {_W_DATA} frames Module 4 built')

assert val_inp.shape[2] == N_ANCHOR, (
    f'Module 4 emitted {val_inp.shape[2]} anchors but the checkpoint expects {N_ANCHOR}.')

M_va = len(val_inp)
val_inp_t   = torch.from_numpy(val_inp).to(DEVICE)
val_tgt_t   = torch.from_numpy(val_tgt).to(DEVICE)
val_dnorm_t = torch.from_numpy(val_dnorm).to(DEVICE)
val_ps_t    = torch.from_numpy(val_ps).to(DEVICE)
print(f'  Val windows   : {M_va:,}  (full set, no subsampling)')
print(f'  Sampling      : {FINAL_STEPS} DDIM steps x {FINAL_SAMPLES} averaged samples')

# ── Diffusion schedule ──────────────────────────────────────────────
'''

MIDDLE = '''
betas = cosine_beta_schedule(T_DIFF).to(DEVICE)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_ac = torch.sqrt(alphas_cumprod)
sqrt_1m_ac = torch.sqrt(1.0 - alphas_cumprod)

# ── Model, from the checkpoint's EMA weights ─────────────────────────
# The EMA copy is what every reported Module 5 number was measured with; the raw weights
# are the noisier last-step snapshot.
ema_model = ConditionalDenoiser(WINDOW_SIZE, LATENT_DIM, HIDDEN_DIM, N_HEADS,
                                N_TF_LAYERS, DROPOUT, MAX_PRED_STEP).to(DEVICE)
_sd = ck.get('ema') or ck.get('model')
assert _sd is not None, "Checkpoint has neither an 'ema' nor a 'model' state_dict."
if ck.get('ema') is None:
    print('  ⚠️  No EMA weights in this checkpoint; scoring the raw weights instead.')
ema_model.load_state_dict(_sd)       # strict: a silent partial load would be worse
ema_model.eval()
_n_params = sum(p.numel() for p in ema_model.parameters())
print(f'  Model         : {_n_params:,} parameters loaded, eval mode')

# History, for QT1. Stored in the checkpoint, so it survives any disconnect.
train_losses = ck.get('train_losses') or []
val_cds      = ck.get('val_cds') or []
val_epochs   = ck.get('val_epochs') or []
best_cd      = float(ck.get('best_cd', float('inf')))

# Full set always: this cell exists to produce the reportable number.
_qt_idx = None
val_subset_idx = representative_subset_indices(M_va, min(M_va, 512))

print('\\n' + '=' * 70)
print('  QUALITY TESTS')
print('=' * 70)
print(f'  Evaluation scope: FULL val set ({M_va:,} windows)')

'''

EPILOGUE = '''
# ── Record, so the paper can read it without a rerun ────────────────
_out = {
    'checkpoint': os.path.basename(_path),
    'epoch': ck.get('epoch'),
    'which': WHICH,
    'n_val_windows': int(M_va),
    'scope': 'full validation set',
    'final_steps': FINAL_STEPS,
    'final_samples': FINAL_SAMPLES,
    'target_mode': TARGET_MODE,
    'window_size': WINDOW_SIZE,
    'data_scale': DATA_SCALE,
    'passes': {'qt1': bool(qt1_pass), 'qt2': bool(qt2_pass), 'qt3': bool(qt3_pass),
               'qt4': bool(qt4_pass), 'qt5': bool(qt5_pass), 'qt6': bool(qt6_pass)},
    'all_pass': bool(all_pass),
}
if val_cds:
    _out['cd'] = {'ours': float(final_cd), 'naive': float(final_cd_naive),
                  'const_velocity': float(cd_cv), 'single_sample': float(cd_single),
                  'improvement_vs_naive_pct': float(improvement),
                  'improvement_vs_cv_pct': float(
                      (cd_cv - final_cd) / max(cd_cv, 1e-9) * 100)}

_dest = f'{MODULE5_DIR}/module5_quality_tests.json'
with open(_dest, 'w') as _fh:
    json.dump(_out, _fh, indent=2)
print(f'\\n  Written: {_dest}')
print('  (no checkpoint was modified: this cell contains no torch.save and no os.rename)')
'''


if __name__ == '__main__':
    main()
