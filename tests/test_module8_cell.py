"""The Module 8 cell, and the properties that let one study run without the others.

Studies A and B cost 14.3 hours between them and are finished. Running Study C on its
own is only safe because of three things, and none of them is obvious from reading the
cell top to bottom:

  * the notebook cell is the same text as module8_hpo.py, so editing the file is
    editing what actually runs;
  * previous results are reloaded from Drive and written back after EVERY study, so a
    crash in a later study cannot discard an earlier one -- which is exactly what
    happened on 19 November, when a Study C OOM threw away 14.3 hours;
  * the figure's panels are chosen by what is in `results`, not by which studies ran
    this session, so switching a study off does not blank its panel.

Each is one line of code and each is silently reversible, so each gets a test.
"""
import ast
import builtins
import json
import os
import re
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, 'module8_hpo.py')
NB = os.path.join(REPO, 'FYP_MultiVideo_Pipeline_(11).ipynb')


def source():
    with open(SRC, encoding='utf-8') as fh:
        return fh.read()


def cells():
    with open(NB, encoding='utf-8') as fh:
        return [''.join(c['source']) for c in json.load(fh)['cells']]


class TestTheCellIsTheFile(unittest.TestCase):
    def test_exactly_one_cell_holds_module8(self):
        want = source().strip()
        hits = [i for i, c in enumerate(cells()) if c.strip() == want]
        self.assertEqual(len(hits), 1,
                         'the Module 8 cell and module8_hpo.py have diverged, so editing '
                         f'the file is not editing what runs in Colab (matched {hits})')

    def test_the_cell_is_self_contained(self):
        """It must run in a fresh runtime with nothing above it.

        A name resolved from an earlier cell works in the notebook the author is using
        and raises NameError on a clean Colab, hours in, after the GPU has been paid
        for. The cell recovers DRIVE_BASE itself for the same reason.
        """
        tree = ast.parse(source())
        bound = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                bound.add(node.id)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bound.add(node.name)
            elif isinstance(node, ast.Import):
                for a in node.names:
                    bound.add((a.asname or a.name).split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                for a in node.names:
                    bound.add(a.asname or a.name)
            elif isinstance(node, ast.arg):
                bound.add(node.arg)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                bound.add(node.name)
            elif isinstance(node, ast.Global):
                bound.update(node.names)
            elif isinstance(node, (ast.For, ast.comprehension)):
                for n in ast.walk(node.target):
                    if isinstance(n, ast.Name):
                        bound.add(n.id)
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                for item in node.items:
                    if item.optional_vars:
                        for n in ast.walk(item.optional_vars):
                            if isinstance(n, ast.Name):
                                bound.add(n.id)
        known = bound | set(dir(builtins)) | {'__name__', '__file__'}
        used = {n.id for n in ast.walk(tree)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        self.assertEqual(sorted(used - known), [],
                         'the cell reads names it never defines, so it depends on an '
                         'earlier cell having been run')


class TestOneStudyCanRunAlone(unittest.TestCase):
    def test_previous_results_are_reloaded(self):
        src = source()
        self.assertRegex(src, r'RESULTS_PATH\s*=',
                         'no results path: nothing to resume from')
        self.assertRegex(
            src, r'if os\.path\.exists\(RESULTS_PATH\)',
            'the cell does not reload previous results, so running Study C alone would '
            'write a file containing only Study C and drop A and B')

    def test_results_are_saved_after_every_study(self):
        """A later study's crash must not cost an earlier study's hours."""
        src = source()
        stages = set(re.findall(r"save_results\(\s*'([^']+)'", src))
        for study in ('Study A', 'Study B', 'Study C'):
            self.assertIn(study, stages,
                          f'nothing is saved after {study}; a crash in the next study '
                          f'would discard it')

    def test_the_run_flags_exist_and_are_independent(self):
        src = source()
        self.assertRegex(src, re.compile(r'^RUN_A,\s*RUN_B,\s*RUN_C\s*=', re.M))
        for flag in ('RUN_A', 'RUN_B', 'RUN_C', 'RUN_D'):
            self.assertRegex(src, rf'\bif {flag}\b',
                             f'{flag} is set but never gates anything')

    def test_a_disabled_study_still_appears_in_the_figure(self):
        """Panels come from `results`, not from which studies ran this session.

        Keyed on the RUN flags instead, re-running only Study C would publish a figure
        with A and B blank, which reads as "those studies failed" rather than "those
        studies are already done".
        """
        src = source()
        for name, key in (('has_A', 'A_inference'), ('has_B', 'B_context'),
                          ('has_C', 'C_optimisation')):
            self.assertRegex(
                src, rf"{name}\s*=\s*'{key}'\s+in\s+results",
                f'{name} is not derived from the loaded results, so switching a study '
                f'off would blank its panel')

    def test_study_c_states_that_its_numbers_are_not_comparable(self):
        """Study C runs at a narrower window and a shorter schedule than the deployed
        model. Its MPEs rank factors against each other; read as absolute numbers they
        would undercut Module 7's table. The JSON has to say so, because whoever reads
        it later will not have this cell in front of them."""
        src = source()
        self.assertRegex(src, r"'note':\s*\(?f?'Run at W=",
                         'C_optimisation carries no note explaining that its absolute '
                         'MPE is not comparable with Study A or Module 7')


if __name__ == '__main__':
    unittest.main(verbosity=2)
