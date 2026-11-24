#!/usr/bin/env python3
"""Generate module8_study_c.py: the Module 8 cell with Study C and nothing else.

    python3 tools/make_study_c_cell.py

WHY GENERATE RATHER THAN WRITE ONE
Studies A and B are finished. The full Module 8 cell still carries their code, plus
Study D's and the diffusion model only A and D load -- about 350 lines that cannot run
and cannot be read past. Deleting them by hand would create a second copy of the data
loader, the backbone and the save logic, which would then drift from module8_hpo.py
silently. The Module 9 standalone cell drifted exactly that way once already.

So this lifts Study C out of module8_hpo.py mechanically. It drops the three disabled
study blocks, then keeps only the top-level definitions still reachable from what
remains -- computed, not listed, so a new helper in module8_hpo.py is carried over
automatically and a helper that stops being used is dropped.

WHAT IT DELIBERATELY KEEPS
The resume-from-Drive block and the figure. Study C alone must still reload Studies A
and B from module8_hpo.json and redraw their panels; a figure with two empty panels
reads as "those studies failed" rather than "those studies are done".
"""
import ast
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, 'module8_hpo.py')
OUT = os.path.join(REPO, 'module8_study_c.py')

DROP_BLOCKS = ('RUN_A', 'RUN_B', 'RUN_D')

HEADER = '''# ════════════════════════════════════════════════════════════════════
# MODULE 8 — STUDY C ONLY: optimisation sensitivity
#
# GENERATED FILE -- do not edit. Produced by tools/make_study_c_cell.py from
# module8_hpo.py. Edit that, then re-run the generator.
#
# Studies A and B are finished (14.3 h between them) and their results live in
# module8/module8_hpo.json on Drive. This cell RELOADS them, adds Study C, and
# writes all three back -- so the output JSON and the figure still carry A and B
# without re-measuring either. Study D is not in this cell at all.
#
# NEEDS   module4_combined/ on Drive, and module8/module8_hpo.json if A and B are
#         to be preserved. It does NOT need the Module 5 checkpoint: only Study A
#         loads that.
# COSTS   16 backbones at W=4, 12 epochs each -- about 2.9 h on a T4. The cell
#         prints its own budget, from Study B's measured seconds, before training.
# ════════════════════════════════════════════════════════════════════
'''


def bound_names(node):
    """Every name a statement binds at its own level."""
    out = set()
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        out.add(node.name)
    elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for t in targets:
            for n in ast.walk(t):
                if isinstance(n, ast.Name):
                    out.add(n.id)
    elif isinstance(node, ast.Import):
        for a in node.names:
            out.add((a.asname or a.name).split('.')[0])
    elif isinstance(node, ast.ImportFrom):
        for a in node.names:
            out.add(a.asname or a.name)
    return out


def loaded_names(node):
    return {n.id for n in ast.walk(node)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def main():
    with open(SRC, encoding='utf-8') as fh:
        src = fh.read()
    lines = src.splitlines(keepends=True)
    tree = ast.parse(src)

    # 1. Split the top level into the disabled study blocks and everything else.
    keep, dropped = [], []
    for node in tree.body:
        is_study = (isinstance(node, ast.If) and isinstance(node.test, ast.Name)
                    and node.test.id in DROP_BLOCKS)
        (dropped if is_study else keep).append(node)
    if len(dropped) != len(DROP_BLOCKS):
        sys.exit(f'expected one top-level block per {DROP_BLOCKS}, found {len(dropped)}. '
                 f'module8_hpo.py has been restructured; update this generator.')

    # 2. Definitions are kept only if something still standing needs them.
    #    Iterate to a fixed point: a kept helper may pull in another.
    defs = {n.name: n for n in keep
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    needed, frontier = set(), set()
    for node in keep:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            frontier |= loaded_names(node)
    while frontier:
        name = frontier.pop()
        if name in needed or name not in defs:
            continue
        needed.add(name)
        frontier |= loaded_names(defs[name])

    final = [n for n in keep
             if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
             or n.name in needed]
    cut = sorted(set(defs) - needed)

    # 3. Emit, preserving each kept statement's own source text and the comments
    #    attached above it.
    out, prev_end = [HEADER], None
    for node in final:
        start = node.lineno - 1
        # Walk back over the comment block and decorators directly above.
        probe = start
        while probe > 0:
            above = lines[probe - 1].strip()
            if above.startswith('#') or above.startswith('@'):
                probe -= 1
            else:
                break
        if prev_end is not None and probe > prev_end:
            # A gap means something was removed between the two; keep one blank line.
            out.append('\n')
        out.append(''.join(lines[probe:node.end_lineno]))
        prev_end = node.end_lineno

    text = ''.join(out)

    # 4. The flags are now facts about this file, not switches.
    text = re.sub(r'^RUN_A,\s*RUN_B,\s*RUN_C\s*=.*$',
                  'RUN_C = True              # this cell is Study C and nothing else',
                  text, count=1, flags=re.M)
    text = re.sub(r'^RUN_D\s*=.*$', '', text, count=1, flags=re.M)
    text = re.sub(r'\n{4,}', '\n\n\n', text)

    # 5. Refuse to emit something that cannot run.
    try:
        ast.parse(text)
    except SyntaxError as exc:
        sys.exit(f'generated cell does not parse: {exc}')

    with open(OUT, 'w', encoding='utf-8') as fh:
        fh.write(text)

    before, after = len(lines), len(text.splitlines())
    print(f'  wrote {OUT}')
    print(f'  {before} lines -> {after}  (dropped {before - after})')
    print(f'  removed study blocks : {", ".join(DROP_BLOCKS)}')
    print(f'  removed definitions  : {", ".join(cut) if cut else "(none)"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
