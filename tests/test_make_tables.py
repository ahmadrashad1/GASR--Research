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


if __name__ == '__main__':
    unittest.main()
