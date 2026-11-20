"""Run module9_demo_export.py end to end against a stubbed Module 6.

The QT cell shipped two runtime errors that neither compile() nor the name-checker
caught, because nothing had ever executed it. This does: a fake Module 6 is placed in
globals, the cell is exec'd against a temporary Drive tree, and the payload it writes is
inspected. Tiny tensors and a stub predictor, so it runs in under a second on CPU.
"""
import json
import os
import shutil
import tempfile
import unittest

import numpy as np

try:
    import torch
except ImportError:                                     # pragma: no cover
    torch = None

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'module9_demo_export.py')

W, N_A, T = 4, 16, 120
SPLIT = 90


def stub_globals(drive_base):
    """Exactly the eight names module9's dependency guard demands, and nothing else:
    if the cell starts reading some other Module 6 object, this test fails first."""
    dev = 'cpu'
    g = np.random.default_rng(0)
    traj = torch.from_numpy(g.standard_normal((T, N_A, 3)).cumsum(0).astype(np.float32) * 0.01)

    def base_pred(ctx, hstep):
        last, prev = ctx[:, -1], ctx[:, -2]
        return last + (last - prev) * hstep.to(last.dtype)[:, None, None]

    def to_physical(d, disp_norm):
        return (d / 10.0) / disp_norm

    def predict_direct(ctx, k, disp_norm=None, eta=0.0, n_samples=4, alpha=1.0, **kw):
        hstep = torch.full((ctx.shape[0],), k, dtype=torch.long)
        # eta > 0 is the stochastic path the sigma estimate relies on; a stub that
        # returned the same value for every row would hide a broken spread.
        noise = torch.randn(ctx.shape[0], N_A, 3) * (0.5 if eta > 0 else 0.0)
        return base_pred(ctx, hstep) + alpha * to_physical(noise + 0.1, disp_norm)

    return {
        # scl/ctr are what Module 6's rebuild_video records (module6_prediction.py:505)
        # and are what returns a normalised trajectory to camera space for the overlay
        # projection. A fixture without them is not a stand-in for `videos`.
        'videos': [{'video_id': 3, 'workspace': 'workspace_demo', 'traj_n': traj,
                    'disp_norm': 1.25, 'n_frames': T, 'split_frame': SPLIT,
                    'ctr': np.zeros(3, dtype=np.float32), 'scl': 2.0}],
        'predict_direct': predict_direct, 'base_pred': base_pred,
        'to_physical': to_physical,
        'BUNDLE': {'epoch': 100, 'target_mode': 'cv_residual', 'data_scale': 10.0},
        'WINDOW_SIZE': W, 'DRIVE_BASE': drive_base, 'DEVICE': dev,
    }


def run_cell(extra=None, with_frames=True):
    base = tempfile.mkdtemp()
    try:
        if with_frames:
            import cv2
            fdir = f'{base}/workspace_demo/frames'
            os.makedirs(fdir)
            for i in range(T):
                cv2.imwrite(f'{fdir}/{i:05d}.png',
                            np.full((360, 640, 3), i % 256, dtype=np.uint8))
        ns = stub_globals(base)
        ns.update(extra or {})
        ns['__name__'] = '__main__'
        with open(SRC) as fh:
            exec(compile(fh.read(), SRC, 'exec'), ns)
        with open(f'{base}/module9_demo/demo_data.json') as fh:
            payload = json.load(fh)
        with open(f'{base}/module9_demo/demo_data.js') as fh:
            js = fh.read()
        return payload, js, base, ns
    except BaseException:
        shutil.rmtree(base, ignore_errors=True)
        raise


@unittest.skipIf(torch is None, 'torch not installed')
class TestModule9RunsEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload, cls.js, cls.base, cls.ns = run_cell()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.base, ignore_errors=True)

    def test_every_field_the_design_requires_is_present(self):
        for key in ('video', 'n_points', 'n_frames', 'horizons', 'window', 'centre',
                    'radius', 'span', 'target_mode', 'checkpoint_epoch', 'mean_error',
                    'motion', 'timing', 'frame_files', 'frames', 'alpha', 'calibrated',
                    'degenerate_horizons'):
            self.assertIn(key, self.payload)

    def test_span_is_twice_the_radius(self):
        self.assertAlmostEqual(self.payload['span'], 2.0 * self.payload['radius'])

    def test_each_horizon_has_its_own_geometry_and_its_own_spread(self):
        n = self.payload['n_points']
        for f in self.payload['frames']:
            self.assertEqual(len(f['ctx']), 3 * n)
            for k in self.payload['horizons']:
                for tag in ('gt', 'pred', 'cv'):
                    self.assertEqual(len(f[f'{tag}{k}']), 3 * n)
                self.assertEqual(len(f[f'sigma{k}']), n)
        self.assertNotIn('sigma', self.payload['frames'][0])

    def test_the_spread_is_not_identical_across_horizons(self):
        """k=1's spread drawn around a k=3 prediction is the mistake this export was
        changed to stop making, and it would show up as two identical arrays."""
        f = self.payload['frames'][0]
        self.assertNotEqual(f['sigma1'], f['sigma3'])

    def test_the_spread_is_a_real_positive_radius(self):
        s = self.payload['frames'][0]['sigma1']
        self.assertGreater(min(s), 0.0)

    def test_frame_indices_follow_the_stated_convention(self):
        for f in self.payload['frames']:
            self.assertEqual(f['img'], f['t'] - 1)
            for k in self.payload['horizons']:
                self.assertEqual(f[f'img{k}'], f['t'] + k - 1)

    def test_every_referenced_frame_was_actually_written(self):
        files = self.payload['frame_files']
        for f in self.payload['frames']:
            for i in [f['img']] + [f[f'img{k}'] for k in self.payload['horizons']]:
                self.assertIn(str(i), files)
                self.assertTrue(os.path.exists(f'{self.base}/module9_demo/{files[str(i)]}'))

    def test_exported_frames_are_downscaled_jpegs(self):
        import cv2
        p = f'{self.base}/module9_demo/' + next(iter(self.payload['frame_files'].values()))
        self.assertTrue(p.endswith('.jpg'))
        self.assertEqual(cv2.imread(p).shape[1], 320)

    def test_every_origin_is_in_the_held_out_tail(self):
        for f in self.payload['frames']:
            self.assertGreaterEqual(f['t'], SPLIT)

    def test_the_js_file_is_loadable_and_identical_to_the_json(self):
        self.assertTrue(self.js.startswith('window.DEMO_DATA = '))
        parsed = json.loads(self.js[len('window.DEMO_DATA = '):].rstrip().rstrip(';\n'))
        self.assertEqual(parsed, self.payload)

    def test_timing_degrades_to_frames_only_when_module_6_never_recovered_it(self):
        self.assertEqual(self.payload['timing'], {'timing_ok': False})

    def test_an_uncalibrated_run_is_not_reported_as_degenerate(self):
        """alpha = 1.0 means no calibration, not a worthless model. Only a fitted
        alpha of exactly 0 collapses the prediction onto the baseline."""
        self.assertEqual(self.payload['degenerate_horizons'], [])

    def test_uncalibrated_is_reported_as_such(self):
        self.assertFalse(self.payload['calibrated'])
        self.assertEqual(self.payload['alpha'], {'1': 1.0, '3': 1.0})


@unittest.skipIf(torch is None, 'torch not installed')
class TestModule9Variants(unittest.TestCase):
    def test_calibration_is_picked_up_from_module_6_when_it_fitted_one(self):
        payload, _, base, _ = run_cell({'CALIBRATE': True, 'APPLY_CALIBRATION': True,
                                        'alpha': {1: 0.62, 3: 0.48}})
        try:
            self.assertTrue(payload['calibrated'])
            self.assertEqual(payload['alpha'], {'1': 0.62, '3': 0.48})
        finally:
            shutil.rmtree(base, ignore_errors=True)

    def test_timing_is_carried_through_when_module_6_recovered_it(self):
        raw = list(range(0, 2 * T, 2))
        raw[95] += 8                                    # one dropped-frame gap
        payload, _, base, _ = run_cell(
            {'timing_by_ws': {'workspace_demo': {'ok': True, 'fps': 30.0,
                                                 'raw_index': raw}}})
        try:
            t = payload['timing']
            self.assertTrue(t['timing_ok'])
            self.assertEqual(t['fps'], 30.0)
            self.assertGreater(t['gap_max'], t['gap_min'])
        finally:
            shutil.rmtree(base, ignore_errors=True)

    def test_a_missing_frames_directory_is_not_fatal(self):
        payload, _, base, _ = run_cell(with_frames=False)
        try:
            self.assertEqual(payload['frame_files'], {})
            self.assertEqual(payload['n_frames'], len(payload['frames']))
        finally:
            shutil.rmtree(base, ignore_errors=True)

    def test_the_guard_names_module_6_rather_than_raising_nameerror(self):
        ns = {'__name__': '__main__'}
        with open(SRC) as fh:
            src = fh.read()
        with self.assertRaises(SystemExit) as cm:
            exec(compile(src, SRC, 'exec'), ns)
        msg = str(cm.exception)
        self.assertIn('MODULE 9 NEEDS MODULE 6', msg)
        self.assertIn('videos', msg)


if __name__ == '__main__':
    unittest.main()
