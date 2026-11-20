"""Task 4 of the clinical-demo plan: the pure helpers Module 9's export rests on.

Module 9 is a notebook cell. It imports torch and reads DRIVE_BASE at import time, so
it cannot be imported here at all. The four helpers under test are therefore required
to sit at module top level with no torch in their bodies, and are lifted out by AST so
this suite runs on a laptop with neither torch nor a Drive mount.
"""
import ast
import json
import os
import unittest

import numpy as np

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'module9_demo_export.py')
NAMES = ['needed_frame_indices', 'frame_file_map', 'motion_by_horizon', 'js_payload']


def load_helpers():
    with open(SRC) as fh:
        tree = ast.parse(fh.read())
    want = {n.name: n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in NAMES}
    missing = set(NAMES) - set(want)
    assert not missing, f'helpers not at module level: {sorted(missing)}'
    ns = {'np': np, 'json': json}
    exec(compile(ast.Module(body=[want[n] for n in NAMES], type_ignores=[]),
                 SRC, 'exec'), ns)
    return ns


class TestHelpers(unittest.TestCase):
    def setUp(self):
        self.h = load_helpers()

    def test_needed_indices_are_sorted_and_unique(self):
        # Module 9's own convention, read off its loop: ctx is tn[t-W:t], so the last
        # OBSERVED frame is t-1, and gt{k} is tn[t+k-1]. Origins 10 and 12 at horizons
        # 1 and 3 therefore need 9,10,12 and 11,12,14 -> 5 unique indices.
        got = self.h['needed_frame_indices']([10, 12], [1, 3])
        self.assertEqual(got, [9, 10, 11, 12, 14])

    def test_needed_indices_accept_an_unsorted_origin_list(self):
        self.assertEqual(self.h['needed_frame_indices']([12, 10], [1]), [9, 10, 11, 12])

    def test_frame_file_map_is_zero_padded(self):
        self.assertEqual(self.h['frame_file_map']([7, 1234])[7], 'frames/00007.jpg')
        self.assertEqual(self.h['frame_file_map']([7, 1234])[1234], 'frames/01234.jpg')

    def test_motion_by_horizon_measures_distance_travelled(self):
        frames = [{'ctx': np.zeros((2, 3)),
                   'gt1': np.array([[0., 0., 1.], [0., 0., 3.]]),
                   'gt3': np.array([[0., 0., 2.], [0., 0., 4.]])}]
        m = self.h['motion_by_horizon'](frames, [1, 3])
        self.assertAlmostEqual(m['1'], 2.0)
        self.assertAlmostEqual(m['3'], 3.0)

    def test_motion_by_horizon_reports_a_static_window_as_zero_not_null(self):
        """The app's skill score divides by this. Zero is the truth for a static
        window and the app is specified to return null from it; a missing key or a
        NaN here would instead crash or poison the comparison."""
        frames = [{'ctx': np.zeros((2, 3)), 'gt1': np.zeros((2, 3))}]
        m = self.h['motion_by_horizon'](frames, [1])
        self.assertEqual(m['1'], 0.0)

    def test_motion_by_horizon_skips_a_horizon_it_has_no_ground_truth_for(self):
        frames = [{'ctx': np.zeros((2, 3)), 'gt1': np.ones((2, 3))}]
        self.assertEqual(list(self.h['motion_by_horizon'](frames, [1, 3])), ['1'])

    def test_js_payload_is_an_assignment_the_browser_can_load(self):
        s = self.h['js_payload']({'video': 'v1', 'n': 2})
        self.assertTrue(s.startswith('window.DEMO_DATA = '))
        self.assertTrue(s.rstrip().endswith(';'))
        self.assertEqual(
            json.loads(s[len('window.DEMO_DATA = '):].rstrip().rstrip(';')),
            {'video': 'v1', 'n': 2})


class TestExportContract(unittest.TestCase):
    def setUp(self):
        with open(SRC) as fh:
            self.src = fh.read()

    def test_no_authored_verdict_survives_in_the_comments(self):
        """The old header claimed constant velocity 'beats us at k=1 and loses by
        k=3' -- a finding from the pre-fix, epoch-60 W=4 checkpoint. The design
        forbids the app, and this exporter, from asserting a result not yet measured."""
        for phrase in ('beats us', 'loses by'):
            self.assertNotIn(phrase, self.src)

    def test_sigma_is_exported_per_horizon(self):
        self.assertIn("f'sigma{k}'", self.src)

    def test_the_demo_passes_the_same_calibration_the_headline_uses(self):
        """predict_direct defaults to alpha=1.0. Module 6's reported numbers use the
        fitted alpha, so an export that omits it shows a different predictor than the
        paper describes."""
        self.assertIn('alpha=', self.src)

    def test_the_cell_names_its_missing_dependency_instead_of_raising_nameerror(self):
        """Run on its own in a fresh runtime, this cell used to die on a bare
        `NameError: name 'videos' is not defined` 139 lines in."""
        self.assertIn('MODULE 9 NEEDS MODULE 6', self.src)


if __name__ == '__main__':
    unittest.main()


class TestNotebookSync(unittest.TestCase):
    """The notebook is what actually runs in Colab; the .py file is what is reviewed.

    When they drift, every reader of this repo is reading something other than what
    produced the results. The plan names cell 47; the Module 9 cell is at index 48
    (47 is empty), so the index is located by content rather than trusted.
    """

    NB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'FYP_MultiVideo_Pipeline_(11).ipynb')

    def _cells(self):
        with open(self.NB, encoding='utf-8') as fh:
            return [''.join(c['source']) for c in json.load(fh)['cells']]

    def test_exactly_one_cell_holds_the_module9_export(self):
        with open(SRC, encoding='utf-8') as fh:
            want = fh.read().strip()
        hits = [i for i, c in enumerate(self._cells()) if c.strip() == want]
        self.assertEqual(len(hits), 1,
                         f'expected the Module 9 export in exactly one cell, found {hits}')

    def test_the_module9_cell_is_at_index_48(self):
        with open(SRC, encoding='utf-8') as fh:
            want = fh.read().strip()
        self.assertEqual(self._cells()[48].strip(), want,
                         'cell 48 has drifted from module9_demo_export.py')
