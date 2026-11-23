"""The paper's macro generator, exercised against the Drive state the close-out audit
actually found.

The case that matters: module6_metrics.json is PRESENT but incomplete. Module 6 writes
its metrics file before the rollout section runs, so a run that stopped early -- or was
interrupted in the renderer -- leaves 'rollout_growth_exponent' absent while every other
key is there. A fallback resolved per FILE treats that file as authoritative and emits a
literal 'nan' into the PDF, with a perfectly good transcribed value sitting unused in
measured.json. Nothing downstream complains: 'nan' is valid LaTeX.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, 'paper', 'make_tables.py')

# Exactly the keys the close-out audit reported in module6_metrics.json on Drive.
AUDIT_KEYS = ['calibration', 'checkpoint', 'checkpoint_epoch', 'data_scale',
              'ddim_samples', 'ddim_steps', 'direct', 'export', 'hyp_eta',
              'hypotheses', 'n_hypotheses', 'per_origin']


def run(drive, outdir):
    r = subprocess.run([sys.executable, SCRIPT, '--drive', drive, '--outdir', outdir],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, f'make_tables failed:\n{r.stdout}\n{r.stderr}'
    with open(os.path.join(outdir, 'generated_macros.tex')) as fh:
        return r.stdout, fh.read()


class TestIncompleteModule6(unittest.TestCase):
    """A present-but-partial metrics file must not poison the macros."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.drive = os.path.join(self.tmp.name, 'drive')
        os.makedirs(os.path.join(self.drive, 'module6'))
        payload = {k: {} for k in AUDIT_KEYS}
        payload.update({'checkpoint_epoch': 100, 'ddim_samples': 4, 'ddim_steps': 20,
                        'hyp_eta': 1.0, 'n_hypotheses': 8,
                        'hypotheses': {'spread_error_correlation': 0.41}})
        # deliberately NO 'rollout_growth_exponent'
        with open(os.path.join(self.drive, 'module6', 'module6_metrics.json'), 'w') as fh:
            json.dump(payload, fh)
        self.addCleanup(self.tmp.cleanup)

    def test_no_nan_anywhere_in_the_macros(self):
        _, tex = run(self.drive, self.tmp.name)
        self.assertNotIn('nan', tex.lower(),
                         'a non-finite value reached the macros; it would render in the PDF')

    def test_live_value_is_preferred_when_present(self):
        out, tex = run(self.drive, self.tmp.name)
        self.assertIn(r'\newcommand{\calibCorr}{0.41}', tex)
        self.assertIn('calibCorr: module6_metrics.json', out)

    def test_absent_field_falls_back_rather_than_emitting_nan(self):
        out, tex = run(self.drive, self.tmp.name)
        self.assertIn('growthExp: measured.json (transcribed)', out)
        self.assertRegex(tex, r'\\newcommand\{\\growthExp\}\{\d')

    def test_provenance_of_every_resolved_macro_is_reported(self):
        """Mixing live and transcribed sources is only acceptable if it is auditable."""
        out, _ = run(self.drive, self.tmp.name)
        for name in ('calibCorr', 'growthExp'):
            self.assertRegex(out, rf'  {name}: \S')


class TestNoModule6(unittest.TestCase):
    """With no live file at all, the transcribed values still carry both macros."""

    def test_falls_back_to_measured_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, tex = run(os.path.join(tmp, 'absent'), tmp)
            self.assertNotIn('nan', tex.lower())
            self.assertIn('calibCorr: measured.json (transcribed)', out)
            self.assertIn('growthExp: measured.json (transcribed)', out)


class TestStaleComparisonIsBypassed(unittest.TestCase):
    """measured.json still carries the epoch-60 W=4 comparison, in which Ours is
    byte-identical to Static because that checkpoint predicted no motion. The live
    Module 7 branch must win outright wherever module7_comparison.json exists, or that
    retracted result would be printed as the paper's headline table."""

    def test_measured_comparison_is_only_an_elif(self):
        with open(SCRIPT) as fh:
            src = fh.read()
        i_live = src.index("if D['m7']:\n    res, order")
        i_stale = src.index("elif MEAS.get('comparison')")
        self.assertLess(i_live, i_stale,
                        'the transcribed comparison must be the fallback, never the first branch')


class TestEveryProseMacroIsDefined(unittest.TestCase):
    """build.sh promises a readable PDF whatever subset of modules has been run.

    A macro emitted only inside `if D['m7']:` quietly breaks that promise: with the
    results downloaded everything builds, and the failure surfaces only on a machine
    without them, as "Undefined control sequence" naming a macro rather than a missing
    file. make_tables.py therefore reads main.tex and guarantees a definition for every
    macro the prose invokes.
    """

    def _used_in_main(self):
        with open(os.path.join(ROOT, 'paper', 'main.tex')) as fh:
            return set(re.findall(r'\\([a-zA-Z]+)\{\}', fh.read()))

    def test_no_result_files_at_all_still_defines_every_macro(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, tex = run(os.path.join(tmp, 'nothing'), tmp)
            defined = set(re.findall(r'\\newcommand\{\\([a-zA-Z]+)\}', tex))
            missing = self._used_in_main() - defined
            self.assertFalse(missing, f'main.tex would not compile; undefined: {sorted(missing)}')

    def test_full_results_also_define_every_macro(self):
        drive = os.environ.get('GASR_RESULTS')
        if not drive or not os.path.isdir(drive):
            self.skipTest('no downloaded results; set GASR_RESULTS to the closeout tree')
        with tempfile.TemporaryDirectory() as tmp:
            _, tex = run(drive, tmp)
            defined = set(re.findall(r'\\newcommand\{\\([a-zA-Z]+)\}', tex))
            self.assertFalse(self._used_in_main() - defined)

    def test_no_macro_is_defined_twice(self):
        """Two \\newcommand lines for one name abort the LaTeX build."""
        with tempfile.TemporaryDirectory() as tmp:
            _, tex = run(os.path.join(tmp, 'nothing'), tmp)
            names = re.findall(r'\\newcommand\{\\([a-zA-Z]+)\}', tex)
            dupes = {n for n in names if names.count(n) > 1}
            self.assertFalse(dupes, f'duplicate macro definitions: {sorted(dupes)}')


class TestDuplicateGuardFires(unittest.TestCase):
    def test_a_second_definition_is_rejected(self):
        with open(os.path.join(ROOT, 'paper', 'make_tables.py')) as fh:
            src = fh.read()
        injected = src.replace("if MEAS.get('direction'):",
                               "macro('ctxWindow', 'COLLISION')\nif MEAS.get('direction'):", 1)
        self.assertNotEqual(injected, src, 'injection anchor missing')
        with tempfile.TemporaryDirectory() as tmp:
            script = os.path.join(tmp, 'dup.py')
            with open(script, 'w') as fh:
                fh.write(injected)
            r = subprocess.run([sys.executable, script, '--drive', tmp, '--outdir', tmp],
                               capture_output=True, text=True, cwd=os.path.join(ROOT, 'paper'))
            self.assertNotEqual(r.returncode, 0, 'a duplicate macro was accepted')
            self.assertIn('DUPLICATE MACRO', r.stdout + r.stderr)


if __name__ == '__main__':
    unittest.main()
