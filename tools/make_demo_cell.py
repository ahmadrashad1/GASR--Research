#!/usr/bin/env python3
"""Generate module9_demo_standalone.py: the demo export without Module 6 in memory.

Why this exists
---------------
module9_demo_export.py reuses Module 6's in-runtime objects on purpose, and that is the
right default: one prediction path, the one every metric was measured with. But those
objects are memory, not Drive, so a runtime disconnect means re-running Module 6 --
roughly an hour of held-out evaluation, rollouts, a hypothesis study and two rendered
MP4s, none of which the demo export reads.

What IS on Drive is everything needed to rebuild them:

    module5/diffusion_ckpt_best.pth        the weights, and the arch/config that built them
    module4_combined/module4_metadata.json per-video smooth_window, disp_norm, split_frame
    <workspace>/exports/gaussian_trajectory.npy   the raw per-frame Gaussian positions
    <workspace>/checkpoints/4dgs_v4.pth           opacity, for the anchor gate

So this cell replays Module 6's reconstruction (minutes, not an hour), loads the
checkpoint, and runs the export. The replay is deterministic: FPS is seeded with
default_rng(M4_SEED + video_id) and the smoothing window is READ from Module 4's
metadata rather than re-selected, which is what makes 'rebuilt' equal to 'the same'.

Why generated rather than hand-written
--------------------------------------
A hand-written version would be a fifth copy of ConditionalDenoiser and a second
implementation of ddim_sample, the baselines and the export itself. This project has
paid for that once already: module5_eval_sweep.py was missed when the context encoder
moved from 3 absolute channels to 6 relative ones, and sat in the repo unable to load
the current checkpoint at all.

Every definition here is therefore lifted VERBATIM by source span -- the model, sampler
and reconstruction out of module6_prediction.py, the export out of
module9_demo_export.py. Nothing is retyped, so nothing can drift at generation time.
Change either module and re-run this script.

Usage:  python3 tools/make_demo_cell.py
"""
import ast
import builtins
import io
import os
import sys

SRC6 = 'module6_prediction.py'
SRC9 = 'module9_demo_export.py'
OUT = 'module9_demo_standalone.py'

# From Module 6. The transitive closure of load_bundle, rebuild_video and
# predict_direct, computed by free-name analysis rather than guessed -- plus `batched`,
# which only the calibration block below needs.
DEFS6 = ['GeomMLP', 'sinusoidal_embedding', 'ConditionalDenoiser', 'cosine_beta_schedule',
         'pred_x0_from_v', 'pred_eps_from_v', 'ddim_sample', 'load_bundle',
         'farthest_point_sampling', 'select_anchors', 'rebuild_video', 'to_physical',
         'base_pred', 'predict_direct', 'batched']

# From Module 9. The export, unchanged, so the two cells cannot produce different
# payloads.
DEFS9 = ['needed_frame_indices', 'frame_file_map', 'motion_by_horizon',
         'camera_from', 'project_norm', 'js_payload',
         '_subsample', '_timing_for', '_frame_hw', '_export_frames', 'export_demo']

# Statement blocks, located by content so a shifted line number cannot silently
# truncate one. (first line, last line) -- both matched exactly, stripped of trailing
# whitespace, except LAST entries marked `in` which are matched as substrings.
BLOCK_VIDEOS = ("print('\\nReplaying Module 4 per-video trajectories (anchors, smoothing, "
                "normalisation)...')",
                "assert videos, 'No video trajectories could be rebuilt -- check DRIVE_BASE.'")
BLOCK_CALIB = ('alpha = {k: 1.0 for k in PRED_STEPS}',
               "    print('  (disabled — CALIBRATE = False)')")
BLOCK_BUNDLE = ('BUNDLE = load_bundle(_ckpt_path)',
                "      '   (network output is the displacement itself)'))")
BLOCK_DEMOCFG = ("DEMO_DIR      = f'{DRIVE_BASE}/module9_demo'",
                 'os.makedirs(DEMO_DIR, exist_ok=True)')
# DEMO_ALPHA is split off from the config above because it reads `alpha`, which the
# calibration block produces, while the calibration block reads DEMO_HORIZONS, which the
# config above produces. Config, then fit, then alpha -- in that order, or one of the two
# is a NameError.
BLOCK_DEMOALPHA = ('# Module 6 fits one magnitude scalar per horizon on TRAINING origins '
                   'and reports its',
                   '              for k in DEMO_HORIZONS}')
BLOCK_RUN = ("_vid = max(videos, key=lambda v: v['n_frames'] - int(v['split_frame']))",
             "print('  demo/clinical/index.html, then open index.html in a browser.')")


def spans(src, names, label):
    """Verbatim source for each named top-level def, comment block above it included."""
    lines = src.splitlines()
    tree = ast.parse(src)
    found = {n.name: n for n in tree.body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names}
    missing = [d for d in names if d not in found]
    if missing:
        sys.exit(f'{label} no longer defines, at module level: {missing}\n'
                 f'It has been restructured; update this script.')
    out = []
    for d in names:
        node = found[d]
        start = node.lineno - 1
        while start > 0 and (lines[start - 1].lstrip().startswith('#')
                             or lines[start - 1].startswith('@')):
            start -= 1
        out.append('\n'.join(lines[start:node.end_lineno]))
    return out, found


def block(src, first_last, label):
    """Verbatim source between two exactly-matched lines."""
    first, last = first_last
    lines = src.splitlines()
    try:
        a = next(i for i, l in enumerate(lines) if l.rstrip() == first)
        b = next(i for i, l in enumerate(lines) if i >= a and l.rstrip() == last)
    except StopIteration:
        sys.exit(f'Could not locate the {label} block. Looked for a line equal to\n'
                 f'  {first!r}\nfollowed by one equal to\n  {last!r}')
    return '\n'.join(lines[a:b + 1])


def check_calls(out):
    """Every call to a locally-defined function must match its signature.

    compile() does not check arity, so a template calling a lifted definition with the
    wrong number of arguments produces a file that imports cleanly and then raises
    TypeError in Colab. That is how ConditionalDenoiser(7 args) shipped in the quality-
    test cell: Module 5's class takes only self and reads its config from module scope,
    while the copies in Modules 6-8 take it as parameters.
    """
    tree = ast.parse(out)
    sigs = {}
    for n in tree.body:
        if isinstance(n, ast.FunctionDef):
            a = n.args
            sigs[n.name] = (len(a.args) - len(a.defaults), len(a.args), bool(a.vararg), a)
        elif isinstance(n, ast.ClassDef):
            init = next((x for x in n.body
                         if isinstance(x, ast.FunctionDef) and x.name == '__init__'), None)
            if init:
                a = init.args
                sigs[n.name] = (len(a.args) - 1 - len(a.defaults), len(a.args) - 1,
                                bool(a.vararg), a)
    bad = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        if node.func.id not in sigs:
            continue
        lo, hi, star, a = sigs[node.func.id]
        npos = len(node.args)
        kw = {k.arg for k in node.keywords if k.arg}
        unknown = kw - {x.arg for x in a.args}
        if (npos > hi and not star) or (npos + len(kw) < lo) or unknown:
            bad.append(f'  line {node.lineno}: {node.func.id}() given {npos} positional '
                       f'and {sorted(kw)}, but it accepts {lo}..{hi} positional'
                       + (f'; unknown keyword(s) {sorted(unknown)}' if unknown else ''))
    return bad


def _binds(node):
    """Every name `node` binds anywhere inside it."""
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for al in n.names:
                out.add((al.asname or al.name).split('.')[0])
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
    return out


def _reads(node):
    """(name, lineno) for every load inside `node`."""
    return [(n.id, n.lineno) for n in ast.walk(node)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)]


def _comp_targets(node):
    """Names bound by a comprehension's own `for` clauses. These live in the
    comprehension's scope, not the module's, so `sum(p.numel() for p in ...)` must not
    read as `p` being used before assignment."""
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.comprehension):
            for y in ast.walk(n.target):
                if isinstance(y, ast.Name):
                    out.add(y.id)
    return out


MUTATING = {'torch.save', 'os.rename', 'os.remove', 'os.rmdir', 'os.unlink',
            'shutil.rmtree', 'shutil.move', 'np.save', 'np.savez'}


def check_read_only(out):
    """The header promises no torch.save and no os.rename. Enforce it.

    This cell exists to be run after a lost runtime, when the checkpoint on Drive is the
    only copy of 100 epochs of training. A regenerated cell that quietly acquired a
    write -- because a lifted definition in Module 6 gained one -- could cost exactly
    that. Checked as AST calls, not as text, so the promise in the comment cannot be
    what satisfies the check.
    """
    tree = ast.parse(out)
    return [f'  line {n.lineno}: {ast.unparse(n.func)}()' for n in ast.walk(tree)
            if isinstance(n, ast.Call) and ast.unparse(n.func) in MUTATING]


def check_order(out):
    """No module-level statement may READ a module-level name before it is bound.

    tools/preflight.py binds every module-level name in a first pass before looking for
    reads, so it reports an unbound name but never an out-of-ORDER one. Assembling this
    cell is exactly where order goes wrong: the demo config defines DEMO_HORIZONS, which
    the calibration block reads, which produces `alpha`, which DEMO_ALPHA reads. Get that
    sequence wrong and Colab raises NameError hundreds of lines into a generated file.

    Only top-level statements are walked, and only names this file binds somewhere are
    considered -- anything genuinely external is preflight's job, not this one.

    An assignment is checked value-first, because its right-hand side runs before its
    target binds; that is what catches `X = [k for k in DEMO_HORIZONS]` placed above the
    line defining DEMO_HORIZONS. Every other statement is checked against what it binds
    as well as what precedes it, since a loop body legitimately reads its own loop
    variable and `if 'X' not in globals(): X = ...` legitimately reads what it is about
    to bind.
    """
    tree = ast.parse(out)
    everything = set()
    for stmt in tree.body:
        everything |= _binds(stmt)
    bound, bad, seen = set(), [], set()
    for stmt in tree.body:
        mine = _binds(stmt)
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # A def body does not run at definition time, so its reads say nothing about
            # module order. Its decorators, argument defaults and base classes DO run.
            parts = list(stmt.decorator_list)
            if isinstance(stmt, ast.ClassDef):
                parts += list(stmt.bases) + [k.value for k in stmt.keywords]
            else:
                parts += [d for d in stmt.args.defaults if d is not None]
                parts += [d for d in stmt.args.kw_defaults if d is not None]
            scope = bound
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            scope, parts = bound, ([stmt.value] if stmt.value is not None else [])
            if isinstance(stmt, ast.AugAssign):
                scope = bound | mine          # x += 1 reads x legitimately
        else:
            scope, parts = bound | mine, [stmt]
        scope = scope | _comp_targets(stmt)
        for part in parts:
            for name, ln in _reads(part):
                if (name in everything and name not in scope
                        and not hasattr(builtins, name) and (name, ln) not in seen):
                    seen.add((name, ln))
                    bad.append(f'  line {ln}: {name} is read here but bound later')
        bound |= mine
    return sorted(bad, key=lambda t: int(t.split()[1].rstrip(':')))


def main():
    for p in (SRC6, SRC9):
        if not os.path.exists(p):
            sys.exit(f'{p} not found -- run this from the repository root.')
    src6 = io.open(SRC6, encoding='utf-8').read()
    src9 = io.open(SRC9, encoding='utf-8').read()

    b6, found6 = spans(src6, DEFS6, SRC6)
    b9, _ = spans(src9, DEFS9, SRC9)

    # Module 6's ConditionalDenoiser takes its config as keyword arguments and
    # load_bundle supplies them from the checkpoint's arch_cfg. Both are lifted together,
    # so check_calls() below verifies that pairing -- but if the class ever moved to
    # reading module scope (as Module 5's does), load_bundle would still pass seven
    # keywords and check_calls would catch it. Asserted here anyway, so the failure names
    # the cause rather than a line number in a generated file.
    init = next((n for n in found6['ConditionalDenoiser'].body
                 if isinstance(n, ast.FunctionDef) and n.name == '__init__'), None)
    n_args = len(init.args.args) if init else -1
    if n_args <= 1:
        sys.exit(f"ConditionalDenoiser.__init__ in {SRC6} now takes {n_args} positional "
                 f"argument(s). It used to take its config as parameters, which is what "
                 f"the lifted load_bundle passes. Reconcile the two before regenerating.")

    cut = DEFS6.index('rebuild_video') + 1
    out = (PREAMBLE
           + '\n\n'.join(b6[:cut]) + '\n\n'
           + block(src6, BLOCK_BUNDLE, 'bundle') + '\n\n'
           + block(src6, BLOCK_VIDEOS, 'videos') + '\n\n'
           + '\n\n'.join(b6[cut:]) + '\n\n'
           + DEMO_INTRO + block(src9, BLOCK_DEMOCFG, 'demo config') + '\n\n'
           + CALIB_INTRO + block(src6, BLOCK_CALIB, 'calibration') + '\n\n'
           + block(src9, BLOCK_DEMOALPHA, 'demo alpha') + '\n\n'
           + '\n\n'.join(b9) + '\n\n'
           + RUN_INTRO + block(src9, BLOCK_RUN, 'run') + '\n')

    # Never write a file that will not parse. The templates in this script are
    # triple-quoted strings whose contents pass through a second round of literal
    # parsing, so a '\n' that should have been '\\n' produces a raw newline inside a
    # quoted string and the generated cell dies on a SyntaxError in Colab, far from the
    # cause.
    try:
        compile(out, OUT, 'exec')
    except SyntaxError as e:
        bad = out.splitlines()[(e.lineno or 1) - 1] if e.lineno else ''
        sys.exit(f'Refusing to write {OUT}: generated source does not parse.\n'
                 f'  {e.msg} at generated line {e.lineno}\n'
                 f'  {bad.strip()[:100]}\n'
                 f"  Check the escaping in this script's template strings.")

    mismatches = check_calls(out)
    if mismatches:
        sys.exit(f'Refusing to write {OUT}: calls do not match local signatures.\n'
                 + '\n'.join(mismatches))

    writes = check_read_only(out)
    if writes:
        sys.exit(f'Refusing to write {OUT}: it claims to write no checkpoint, but it '
                 f'calls\n' + '\n'.join(writes)
                 + '\n  Either the lifted code gained a write, or the claim in PREAMBLE '
                   'is now false.')

    disorder = check_order(out)
    if disorder:
        sys.exit(f'Refusing to write {OUT}: module-level statements are out of order.\n'
                 + '\n'.join(disorder)
                 + '\n  Reorder the blocks in main().')

    io.open(OUT, 'w', encoding='utf-8').write(out)
    n_lifted = sum(b.count('\n') + 1 for b in b6 + b9)
    print(f'wrote {OUT}')
    print(f'  {len(DEFS6)} definitions lifted verbatim from {SRC6}')
    print(f'  {len(DEFS9)} definitions lifted verbatim from {SRC9}')
    print(f'  4 statement blocks lifted by content anchor')
    print(f'  {n_lifted} lifted lines, {out.count(chr(10)) + 1} total')


PREAMBLE = '''# ════════════════════════════════════════════════════════════════════
# MODULE 9 — DEMO EXPORT, STANDALONE  (needs nothing but Drive)
#
# GENERATED FILE -- do not edit. Produced by tools/make_demo_cell.py, which lifts the
# model, the sampler, the trajectory reconstruction and the export itself verbatim out
# of module6_prediction.py and module9_demo_export.py. Edit those and re-run the script.
#
# WHY THIS EXISTS
# The normal Module 9 cell reuses Module 6's objects from memory, which is the right
# default: one prediction path, the one every metric was measured with. But memory is
# lost on a runtime disconnect, and re-running Module 6 costs roughly an hour of
# held-out evaluation, rollouts, a hypothesis study and two rendered MP4s -- none of
# which the demo export reads.
#
# This cell rebuilds only what the export needs, from Drive:
#   diffusion_ckpt_best.pth   -> the weights, and the arch/config that built them
#   module4_metadata.json     -> per-video smooth_window, disp_norm, split_frame
#   gaussian_trajectory.npy   -> the raw per-frame Gaussian positions
#   4dgs_v4.pth               -> opacity, for the anchor gate
#
# 'Rebuilt' is the same trajectory, not a similar one: anchor selection is seeded with
# default_rng(M4_SEED + video_id) and the smoothing window is READ from Module 4's
# metadata rather than re-selected. Module 6's QT0 is what verifies that claim against
# Module 4's own saved windows; it is not repeated here, so if the replay has never been
# verified in this project, run Module 6 at least once before trusting this cell.
#
# WHAT IS LOST BY NOT RUNNING MODULE 6
#   - frame timing. Module 6 recovers which raw video frame each trajectory frame came
#     from, well past the point this cell stops. Absent it the payload carries
#     timing_ok: false and the viewer states the horizon in frames only -- which is
#     exact, and the behaviour the design asks for anyway.
#   - nothing else. The magnitude calibration IS fitted below, on training origins, the
#     same scalar Module 6 reports its headline numbers with.
#
# It does not train and it writes no checkpoint: there is no torch.save and no os.rename
# anywhere in this file.
# ════════════════════════════════════════════════════════════════════
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, os, math, json, time
from scipy.signal import savgol_filter

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
USE_BEST_CKPT = True     # the best-validating EMA snapshot, which is what the paper
                          # quotes. Set False to export the final-epoch weights instead.

# Sampling budget. Matches Module 6 exactly, so the numbers this cell prints are
# comparable with Module 6's own -- a different DDIM_STEPS or DDIM_SAMPLES would give a
# different prediction from identical weights.
DDIM_STEPS    = 20
DDIM_SAMPLES  = 4
EVAL_ROWS     = 64

# Calibration. Module 6's defaults, repeated here rather than imported, because this
# cell has to stand on its own -- and flagged in the payload either way.
CALIBRATE         = True
APPLY_CALIBRATION = True
CALIB_PER_VIDEO   = 24
CALIB_MAX         = 3.0

SEED = 2026
torch.manual_seed(SEED)
np.random.seed(SEED)

assert os.path.exists(f'{COMBINED_DIR}/module4_metadata.json'), (
    f'module4_metadata.json not found in {COMBINED_DIR} -- run Module 4 first.')
_ckpt_path = CKPT_BEST if (USE_BEST_CKPT and os.path.exists(CKPT_BEST)) else CKPT_LATEST
assert os.path.exists(_ckpt_path), (
    f'No Module 5 checkpoint found ({CKPT_BEST} / {CKPT_LATEST}) -- train Module 5 first.')

meta4 = json.load(open(f'{COMBINED_DIR}/module4_metadata.json'))
N_ANCHOR    = meta4['n_anchor']
WINDOW_SIZE = meta4['window_size']
PRED_STEPS  = meta4['pred_steps']
M4_SEED     = meta4['seed']
POLYORDER   = meta4['smooth_polyorder']
video_order = meta4['video_order']

print('=' * 64)
print('  MODULE 9 — Demo export (standalone: Drive only, no Module 6 needed)')
print('=' * 64)
print(f'  Device          : {DEVICE}')
print(f'  Checkpoint      : {os.path.basename(_ckpt_path)}')
print(f'  Videos          : {len(video_order)}')
print(f'  Anchors / frame : {N_ANCHOR}   |  context window W = {WINDOW_SIZE}')
print(f'  Trained horizons: {PRED_STEPS}')


'''

CALIB_INTRO = '''# ════════════════════════════════════════════════════════════════════
# CALIBRATION — one scalar per horizon, fitted on TRAINING origins
#
# Lifted verbatim from Module 6, which is what makes the alpha below the same alpha the
# paper's headline numbers use. predict_direct defaults to alpha=1.0, so without this
# the demonstrator would show a different predictor than the write-up describes.
#
# Narrowed to the horizons the demo actually exports. Each horizon's fit is independent
# -- num and den are reset at the top of every k -- so the alpha for a retained horizon
# is bit-identical to the one a full-PRED_STEPS loop would produce. All that changes is
# that horizons nothing will be exported at are not paid for.
# ════════════════════════════════════════════════════════════════════
_unsupported = [k for k in DEMO_HORIZONS if k not in PRED_STEPS]
assert not _unsupported, (
    f'DEMO_HORIZONS asks for k={_unsupported}, which the model was never trained on '
    f'(trained: {PRED_STEPS}). Direct prediction is only valid at trained horizons.')
PRED_STEPS_TRAINED = list(PRED_STEPS)
PRED_STEPS = [k for k in PRED_STEPS_TRAINED if k in DEMO_HORIZONS]
print('\\n' + '=' * 64)
print(f'  CALIBRATION — fitted on training origins, for k={PRED_STEPS}')
print('=' * 64)

'''

DEMO_INTRO = '''# ════════════════════════════════════════════════════════════════════
# THE EXPORT — lifted verbatim from module9_demo_export.py
#
# Identical code, so the two cells cannot write different payloads. The DEMO_HORIZONS
# below is the one the calibration block above was narrowed to; it is defined there in
# the normal cell too, which is why this block has to come after the config and before
# the fit in the generated order.
# ════════════════════════════════════════════════════════════════════
'''

RUN_INTRO = '''# ════════════════════════════════════════════════════════════════════
# RUN
# ════════════════════════════════════════════════════════════════════
'''

if __name__ == '__main__':
    main()
