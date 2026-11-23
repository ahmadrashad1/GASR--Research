"""Task 4 of the clinical-demo plan: the pure helpers Module 9's export rests on.

Module 9 is a notebook cell. It imports torch and reads DRIVE_BASE at import time, so
it cannot be imported here at all. The four helpers under test are therefore required
to sit at module top level with no torch in their bodies, and are lifted out by AST so
this suite runs on a laptop with neither torch nor a Drive mount.
"""
import ast
import json
import math
import os
import unittest

import numpy as np

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'module9_demo_export.py')
NAMES = ['needed_frame_indices', 'frame_file_map', 'motion_by_horizon', 'js_payload',
         'camera_from', 'project_norm']


def load_helpers():
    with open(SRC) as fh:
        tree = ast.parse(fh.read())
    want = {n.name: n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in NAMES}
    missing = set(NAMES) - set(want)
    assert not missing, f'helpers not at module level: {sorted(missing)}'
    ns = {'np': np, 'json': json, 'math': math}
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

    def test_the_standalone_cell_is_synced_too(self):
        """Cell 50 holds the generated standalone export. It drifted once already,
        silently, because only cell 48 was checked -- and the generated file changes
        whenever its sources do, which is exactly when nobody thinks to resync it."""
        gen = os.path.join(os.path.dirname(SRC), 'module9_demo_standalone.py')
        with open(gen, encoding='utf-8') as fh:
            want = fh.read().strip()
        hits = [i for i, c in enumerate(self._cells()) if c.strip() == want]
        self.assertEqual(hits, [50], f'standalone cell out of sync; matched {hits}')

    def test_the_module9_cell_is_at_index_48(self):
        with open(SRC, encoding='utf-8') as fh:
            want = fh.read().strip()
        self.assertEqual(self._cells()[48].strip(), want,
                         'cell 48 has drifted from module9_demo_export.py')


class TestOverlayProjection(unittest.TestCase):
    """Drawing the prediction onto the endoscopic frame.

    Module 6 states the trajectories are already in camera space ("World (= camera)
    points"), so no extrinsics are involved -- only the pinhole intrinsics. The export
    does the projection, because the browser must not carry camera maths or assumptions.
    """

    def setUp(self):
        self.h = load_helpers()

    def test_a_point_on_the_optical_axis_lands_at_the_principal_point(self):
        cam = {'fx': 100.0, 'fy': 100.0, 'cx': 320.0, 'cy': 180.0, 'W': 640, 'H': 360}
        uv = self.h['project_norm'](np.array([[0.0, 0.0, 2.0]]), cam)
        self.assertAlmostEqual(float(uv[0][0]), 0.5, places=6)
        self.assertAlmostEqual(float(uv[0][1]), 0.5, places=6)

    def test_projection_is_resolution_independent(self):
        """Normalised [0,1] coordinates, so the browser can draw at any displayed size
        and the exported JPEG's own width never enters the payload."""
        pt = np.array([[1.0, 0.0, 1.0]])
        small = {'fx': 100.0, 'fy': 100.0, 'cx': 320.0, 'cy': 180.0, 'W': 640, 'H': 360}
        large = {'fx': 200.0, 'fy': 200.0, 'cx': 640.0, 'cy': 360.0, 'W': 1280, 'H': 720}
        a = float(self.h['project_norm'](pt, small)[0][0])
        b = float(self.h['project_norm'](pt, large)[0][0])
        self.assertAlmostEqual(a, b, places=9)
        self.assertAlmostEqual(a, 420.0 / 640.0, places=9)

    def test_a_point_behind_the_camera_does_not_blow_up(self):
        cam = {'fx': 100.0, 'fy': 100.0, 'cx': 320.0, 'cy': 180.0, 'W': 640, 'H': 360}
        uv = self.h['project_norm'](np.array([[1.0, 1.0, -5.0], [1.0, 1.0, 0.0]]), cam)
        self.assertTrue(np.all(np.isfinite(uv)), 'a non-finite pixel would poison the canvas')

    def test_real_intrinsics_are_used_and_marked_calibrated(self):
        meta = {'fx': 500.0, 'fy': 501.0, 'cx': 320.0, 'cy': 180.0, 'W': 640, 'H': 360}
        cam = self.h['camera_from'](meta, (360, 640))
        self.assertEqual(cam['calibrated'], True)
        self.assertAlmostEqual(cam['fx'], 500.0)
        self.assertAlmostEqual(cam['fy'], 501.0)

    def test_absent_intrinsics_fall_back_and_say_so(self):
        """Module 1's documented 75-degree fallback. The overlay is then plausible but
        NOT calibrated, and the payload has to carry that distinction or the app will
        present a guess with the same confidence as a measurement."""
        cam = self.h['camera_from'](None, (360, 640))
        self.assertEqual(cam['calibrated'], False)
        self.assertAlmostEqual(cam['fx'], (640 / 2) / math.tan(math.radians(75.0) / 2), places=6)
        self.assertEqual(cam['cx'], 320.0)
        self.assertEqual(cam['cy'], 180.0)

    def test_the_payload_carries_pixel_coordinates_and_the_camera_block(self):
        with open(SRC) as fh:
            src = fh.read()
        self.assertIn("'camera': cam", src,
                      'the payload must state which intrinsics the overlay used')
        self.assertIn("'px_ctx': enc(", src, 'the observed frame needs image coordinates')
        self.assertIn("f'px_{tag}{k}': enc(", src,
                      'gt, pred and cv each need image coordinates per horizon')
        self.assertIn("for tag in ('gt', 'pred', 'cv')", src)


class TestCalibrationIsNotAssumed(unittest.TestCase):
    """Module 1 ALWAYS writes camera_meta.json, and always with fx derived from a
    nominal 75-degree laparoscope FOV -- a textbook figure for a 10mm scope, not a
    measurement of this one. The presence of the file is therefore not evidence of
    calibration, and treating it as such suppresses the one warning that stops an
    assumption reaching a projector with the confidence of a measurement.
    """

    def setUp(self):
        self.h = load_helpers()

    @staticmethod
    def _module1_meta(W=640, H=360):
        fx = (W / 2) / math.tan(math.radians(75.0) / 2)
        return {'H': H, 'W': W, 'fx': fx, 'fy': fx, 'cx': W / 2.0, 'cy': H / 2.0,
                'fov_deg': 75.0, 'n_frames': 300}

    def test_module_1s_assumed_intrinsics_are_not_called_calibrated(self):
        cam = self.h['camera_from'](self._module1_meta(), (360, 640))
        self.assertEqual(cam['calibrated'], False)
        self.assertEqual(cam['source'], 'assumed-fov')

    def test_the_assumption_is_caught_even_without_the_fov_field(self):
        meta = self._module1_meta()
        del meta['fov_deg']
        cam = self.h['camera_from'](meta, (360, 640))
        self.assertEqual(cam['calibrated'], False)

    def test_genuinely_measured_intrinsics_are_still_honoured(self):
        meta = {'H': 360, 'W': 640, 'fx': 503.2, 'fy': 499.7, 'cx': 322.4, 'cy': 178.9}
        cam = self.h['camera_from'](meta, (360, 640))
        self.assertEqual(cam['calibrated'], True)
        self.assertEqual(cam['source'], 'measured')

    def test_an_absent_file_is_also_uncalibrated(self):
        cam = self.h['camera_from'](None, (360, 640))
        self.assertEqual(cam['calibrated'], False)
        self.assertEqual(cam['source'], 'assumed-fov')
