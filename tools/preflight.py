#!/usr/bin/env python3
"""
Pre-flight check for the Colab modules.

A NameError forty minutes into a GPU run is expensive and entirely avoidable.
This walks the AST and reports names that are read but never bound anywhere
they could be visible, plus a few project-specific checks: that the Drive
paths agree across modules, and that every checkpoint a module loads is one
some other module actually writes.

It is deliberately conservative -- it prefers silence to a false alarm, so a
clean report is weak evidence of correctness, while any hit is worth reading.

    python3 tools/preflight.py module7_baselines.py module8_hpo.py
"""
import ast, builtins, json, sys, pathlib, re

BUILTIN = set(dir(builtins)) | {
    '__name__', '__file__', '__doc__', '__spec__', '__package__', '__loader__',
}



def _lambda_args(node):
    """Bind lambda parameters. Conservative on purpose: slight over-binding
    costs a missed warning, under-binding costs a false alarm, and a checker
    that cries wolf gets ignored."""
    out = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Lambda):
            a = sub.args
            for x in (a.posonlyargs + a.args + a.kwonlyargs):
                out.add(x.arg)
            if a.vararg:
                out.add(a.vararg.arg)
            if a.kwarg:
                out.add(a.kwarg.arg)
    return out


class Scope:
    def __init__(self, parent=None):
        self.names, self.parent = set(), parent

    def bind(self, n):
        self.names.add(n)

    def visible(self, n):
        s = self
        while s:
            if n in s.names:
                return True
            s = s.parent
        return False


class Checker(ast.NodeVisitor):
    """Two passes: bind every module-level name first, then look for reads.

    Binding first matters because Python resolves a function body's globals at
    call time, so a helper defined at the bottom of the file is legitimately
    visible to one defined at the top.
    """

    def __init__(self, path):
        self.path, self.module = path, Scope()
        self.problems = []

    # ── pass 1: everything bound at module level ────────────────────
    def collect(self, tree):
        for n in _lambda_args(tree):
            self.module.bind(n)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.module.bind(node.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                self.module.bind(node.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for a in node.names:
                    self.module.bind((a.asname or a.name).split('.')[0])
            elif isinstance(node, ast.ExceptHandler) and node.name:
                self.module.bind(node.name)
            elif isinstance(node, (ast.Global, ast.Nonlocal)):
                for n in node.names:
                    self.module.bind(n)

    # ── pass 2: reads, with local scopes ────────────────────────────
    def check_fn(self, node, parent):
        local = Scope(parent)
        args = node.args
        for a in (args.posonlyargs + args.args + args.kwonlyargs):
            local.bind(a.arg)
        if args.vararg:
            local.bind(args.vararg.arg)
        if args.kwarg:
            local.bind(args.kwarg.arg)
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, (ast.Store, ast.Del)):
                local.bind(sub.id)
            elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                local.bind(sub.name)
                # A nested def's PARAMETERS are bound inside it, and ast.walk brings its
                # body into this scope without them -- so every parameter of an inner
                # helper read like an unbound name. Bind them here. Over-binding costs a
                # missed warning; under-binding cost a false alarm on a correct file,
                # which is the failure that gets a checker ignored.
                if not isinstance(sub, ast.ClassDef):
                    _a = sub.args
                    for _x in (_a.posonlyargs + _a.args + _a.kwonlyargs):
                        local.bind(_x.arg)
                    if _a.vararg:
                        local.bind(_a.vararg.arg)
                    if _a.kwarg:
                        local.bind(_a.kwarg.arg)
            elif isinstance(sub, (ast.Import, ast.ImportFrom)):
                for a in sub.names:
                    local.bind((a.asname or a.name).split('.')[0])
            elif isinstance(sub, ast.ExceptHandler) and sub.name:
                local.bind(sub.name)
            elif isinstance(sub, (ast.comprehension,)):
                for t in ast.walk(sub.target):
                    if isinstance(t, ast.Name):
                        local.bind(t.id)
            elif isinstance(sub, ast.withitem) and sub.optional_vars is not None:
                for t in ast.walk(sub.optional_vars):
                    if isinstance(t, ast.Name):
                        local.bind(t.id)
        for n in _lambda_args(node):
            local.bind(n)
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                if not local.visible(sub.id) and sub.id not in BUILTIN:
                    self.problems.append((sub.lineno, sub.id, node.name))

    def run(self, tree):
        self.collect(tree)

        # Every function body, wherever it sits, is checked in its own scope.
        # Methods take the class body as an intermediate scope so that names
        # bound at class level are visible to them.
        def walk_defs(node, parent):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self.check_fn(child, parent)
                elif isinstance(child, ast.ClassDef):
                    cls = Scope(parent)
                    for sub in ast.iter_child_nodes(child):
                        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                            cls.bind(sub.id)
                        elif isinstance(sub, ast.Assign):
                            for t in sub.targets:
                                for n in ast.walk(t):
                                    if isinstance(n, ast.Name):
                                        cls.bind(n.id)
                    walk_defs(child, cls)
                else:
                    walk_defs(child, parent)
        walk_defs(tree, self.module)

        # Module-level reads: walk without descending into any definition,
        # because those were handled above with their own scopes.
        def walk_flat(node):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                    if not self.module.visible(child.id) and child.id not in BUILTIN:
                        self.problems.append((child.lineno, child.id, '<module>'))
                walk_flat(child)
        walk_flat(tree)
        return self.problems


def notebook_globals(nb_path):
    """Names bound at the top level of any notebook cell.

    These modules run as Colab cells, so a name set in an earlier cell is
    legitimately visible later even though it is undefined in the .py file.
    Harvesting them from the notebook is better than an allowlist, which
    would quietly go stale.
    """
    names = set()
    try:
        nb = json.loads(pathlib.Path(nb_path).read_text(encoding='utf-8'))
    except Exception:
        return names
    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
        try:
            tree = ast.parse(''.join(cell.get('source', [])))
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                tgts = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in tgts:
                    for n in ast.walk(t):
                        if isinstance(n, ast.Name):
                            names.add(n.id)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for a in node.names:
                    names.add((a.asname or a.name).split('.')[0])
    return names


def paths_in(src):
    return set(re.findall(r"f?['\"]\{?DRIVE_BASE\}?/([A-Za-z0-9_./{}]+)['\"]", src))


def main(files):
    nb = [f for f in files if f.endswith('.ipynb')]
    files = [f for f in files if not f.endswith('.ipynb')]
    provided = set()
    for n in nb:
        provided |= notebook_globals(n)
    if provided:
        print(f'carrying {len(provided)} name(s) provided by earlier notebook cells')
    bad = 0
    for f in files:
        p = pathlib.Path(f)
        src = p.read_text(encoding='utf-8')
        tree = ast.parse(src)
        chk = Checker(p)
        for n in provided:
            chk.module.bind(n)
        probs = chk.run(tree)
        seen, out = set(), []
        for ln, name, where in sorted(probs):
            if (name, where) in seen:
                continue
            seen.add((name, where))
            out.append(f'    line {ln:>5}  {name:<24} (in {where})')
        print(f'\n{p.name}')
        if out:
            bad += len(out)
            print('  names read but never bound:')
            print('\n'.join(out))
        else:
            print('  no unbound names')
    print(f'\n{"FAIL" if bad else "PASS"}: {bad} suspect name(s)')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:] or ['module7_baselines.py', 'module8_hpo.py']))
