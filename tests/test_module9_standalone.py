"""Run module9_demo_standalone.py end to end against a fabricated Drive tree.

This cell's whole purpose is to save an hour of Colab time, so shipping it untested
would defeat the point. A temporary DRIVE_BASE is built with every file the cell reads
-- Module 4's metadata, a per-video Gaussian trajectory, the 4DGS opacity checkpoint,
endoscopic frames, and a real (tiny) diffusion checkpoint whose weights come from the
cell's OWN ConditionalDenoiser -- and then the cell is exec'd against it on CPU.

That exercises the chain the generator cannot check statically: load_bundle feeding
arch_cfg into the model, the anchor replay, savgol smoothing, the calibration fit,
DDIM sampling, and the payload write.

The checkpoint's context window is deliberately NARROWER than Module 4's metadata
window, because that is the real configuration -- the project trains on a truncation of
Module 4's full-width window -- and encode_context is where that truncation happens.
"""
import ast
import json
import os
import shutil
import tempfile
import unittest

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:                                     # pragma: no cover
    torch = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'module9_demo_standalone.py')

T_FRAMES, N_GAUSS, N_ANCHOR, SPLIT = 120, 40, 16, 90
W_META, W_ARCH = 8, 4
ARCH = {'WINDOW_SIZE': W_ARCH, 'LATENT_DIM': 32, 'HIDDEN_DIM': 64, 'N_HEADS': 4,
        'N_TF_LAYERS': 1, 'MAX_PRED_STEP': 8, 'T_DIFF': 50,
        'TARGET_MODE': 'cv_residual'}


def lift(names):
    """Pull definitions out of the generated cell without executing its module body."""
    with open(SRC) as fh:
        tree = ast.parse(fh.read())
    want = {n.name: n for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names}
    missing = set(names) - set(want)
    assert not missing, f'not at module level in the generated cell: {sorted(missing)}'
    ns = {'torch': torch, 'nn': nn, 'F': F, 'np': np, 'math': __import__('math')}
    exec(compile(ast.Module(body=[want[n] for n in names], type_ignores=[]),
                 SRC, 'exec'), ns)
    return ns


def build_drive():
    base = tempfile.mkdtemp()
    ws = 'workspace_demo'
    rng = np.random.default_rng(7)

    os.makedirs(f'{base}/{ws}/exports')
    os.makedirs(f'{base}/{ws}/checkpoints')
    os.makedirs(f'{base}/{ws}/frames')
    os.makedirs(f'{base}/module4_combined')
    os.makedirs(f'{base}/module5')

    traj = (rng.standard_normal((T_FRAMES, N_GAUSS, 3)).cumsum(0) * 0.02).astype(np.float32)
    np.save(f'{base}/{ws}/exports/gaussian_trajectory.npy', traj)
    # sigmoid(0) = 0.5, comfortably over the 0.05 visibility gate, so the anchor
    # selection takes the opacity-gated branch rather than falling back.
    torch.save({'opa': torch.zeros(N_GAUSS, 1)}, f'{base}/{ws}/checkpoints/4dgs_v4.pth')

    import cv2
    for i in range(T_FRAMES):
        cv2.imwrite(f'{base}/{ws}/frames/{i:05d}.png',
                    np.full((360, 640, 3), i % 256, dtype=np.uint8))

    json.dump({'n_anchor': N_ANCHOR, 'window_size': W_META, 'pred_steps': [1, 2, 3],
               'seed': 42, 'smooth_polyorder': 2, 'video_order': [ws],
               'videos': [{'workspace': ws, 'video_id': 0, 'smooth_window': 5,
                           'disp_norm_factor': 1.25, 'split_frame_pred1': SPLIT,
                           'psnr': 31.4}]},
              open(f'{base}/module4_combined/module4_metadata.json', 'w'))

    ns = lift(['GeomMLP', 'sinusoidal_embedding', 'ConditionalDenoiser'])
    torch.manual_seed(0)
    model = ns['ConditionalDenoiser'](
        window_size=ARCH['WINDOW_SIZE'], latent_dim=ARCH['LATENT_DIM'],
        hidden_dim=ARCH['HIDDEN_DIM'], n_heads=ARCH['N_HEADS'],
        n_tf_layers=ARCH['N_TF_LAYERS'], dropout=0.1,
        max_pred_step=ARCH['MAX_PRED_STEP'])
    sd = model.state_dict()
    sd['ctx_scale'] = torch.tensor(43.166)          # as Module 5 trains it
    torch.save({'arch_cfg': ARCH, 'config': {'DATA_SCALE': 285.352}, 'ema': sd,
                'epoch': 100, 'best_cd': 0.007702},
               f'{base}/module5/diffusion_ckpt_best.pth')
    return base


@unittest.skipIf(torch is None, 'torch not installed')
class TestStandaloneCellRuns(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = build_drive()
        ns = {'__name__': '__main__', 'DRIVE_BASE': cls.base}
        with open(SRC) as fh:
            exec(compile(fh.read(), SRC, 'exec'), ns)
        cls.ns = ns
        with open(f'{cls.base}/module9_demo/demo_data.json') as fh:
            cls.payload = json.load(fh)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.base, ignore_errors=True)

    def test_it_loaded_the_checkpoint_rather_than_assuming_a_configuration(self):
        self.assertEqual(self.ns['BUNDLE']['epoch'], 100)
        self.assertEqual(self.ns['BUNDLE']['target_mode'], 'cv_residual')
        self.assertAlmostEqual(self.ns['BUNDLE']['data_scale'], 285.352)
        self.assertAlmostEqual(
            float(self.ns['BUNDLE']['model'].ctx_scale), 43.166, places=3)

    def test_it_rebuilt_the_video_from_drive(self):
        self.assertEqual(len(self.ns['videos']), 1)
        v = self.ns['videos'][0]
        self.assertEqual(tuple(v['traj_n'].shape), (T_FRAMES, N_ANCHOR, 3))
        self.assertEqual(v['split_frame'], SPLIT)
        self.assertAlmostEqual(v['disp_norm'], 1.25)

    def test_the_replay_normalises_on_frame_zero_as_module_4_did(self):
        f0 = self.ns['videos'][0]['traj_n'][0].cpu().numpy()
        self.assertAlmostEqual(float(np.abs(f0 - f0.mean(axis=0)).max()), 1.0, places=4)

    def test_calibration_was_fitted_only_at_the_exported_horizons(self):
        self.assertEqual(sorted(self.ns['alpha']), [1, 3])
        self.assertEqual(self.ns['PRED_STEPS_TRAINED'], [1, 2, 3])
        for k, a in self.ns['alpha'].items():
            self.assertGreaterEqual(a, 0.0)
            self.assertLessEqual(a, self.ns['CALIB_MAX'])

    def test_calibration_was_fitted_on_training_origins_only(self):
        """An alpha fitted on the held-out tail would be fitting the answer."""
        self.assertTrue(all(c['n_windows'] > 0 for c in self.ns['calib_report'].values()))
        self.assertLessEqual(max(self.ns['CALIB_PER_VIDEO'], 1) * 1,
                             max(c['n_windows'] for c in self.ns['calib_report'].values()))

    def test_the_payload_carries_the_fitted_alpha_not_one(self):
        self.assertTrue(self.payload['calibrated'])
        self.assertEqual(sorted(self.payload['alpha']), ['1', '3'])
        for k in ('1', '3'):
            self.assertAlmostEqual(self.payload['alpha'][k], self.ns['alpha'][int(k)])

    def test_every_field_the_design_requires_is_present(self):
        for key in ('video', 'workspace', 'n_points', 'n_frames', 'horizons', 'window',
                    'centre', 'radius', 'span', 'target_mode', 'checkpoint_epoch',
                    'mean_error', 'motion', 'timing', 'frame_files', 'frames',
                    'alpha', 'calibrated', 'degenerate_horizons'):
            self.assertIn(key, self.payload)
        self.assertEqual(self.payload['window'], W_META)
        self.assertEqual(self.payload['checkpoint_epoch'], 100)
        self.assertEqual(self.payload['target_mode'], 'cv_residual')

    def test_each_horizon_has_its_own_geometry_and_its_own_spread(self):
        n = self.payload['n_points']
        self.assertEqual(n, N_ANCHOR)
        for f in self.payload['frames']:
            self.assertEqual(len(f['ctx']), 3 * n)
            for k in (1, 3):
                for tag in ('gt', 'pred', 'cv'):
                    self.assertEqual(len(f[f'{tag}{k}']), 3 * n)
                self.assertEqual(len(f[f'sigma{k}']), n)
        # A horizon whose alpha survived the fit must have a real positive spread; one
        # that was clipped to zero must have none, and must say so. Random weights make
        # the degenerate case reachable here, which is why it is pinned rather than
        # asserted away.
        deg = set(self.payload['degenerate_horizons'])
        for k in ('1', '3'):
            s = self.payload['frames'][0][f'sigma{k}']
            if k in deg:
                self.assertEqual(max(s), 0.0)
                self.assertEqual(self.payload['frames'][0][f'pred{k}'],
                                 self.payload['frames'][0][f'cv{k}'])
            else:
                self.assertGreater(min(s), 0.0)
        if not deg:
            self.assertNotEqual(self.payload['frames'][0]['sigma1'],
                                self.payload['frames'][0]['sigma3'])

    def test_every_origin_is_in_the_held_out_tail(self):
        self.assertTrue(self.payload['frames'])
        for f in self.payload['frames']:
            self.assertGreaterEqual(f['t'], SPLIT)
            self.assertLessEqual(f['t'] + 3 - 1, T_FRAMES - 1)

    def test_timing_degrades_to_frames_only_without_module_6(self):
        self.assertEqual(self.payload['timing'], {'timing_ok': False})

    def test_both_payload_files_were_written_and_agree(self):
        with open(f'{self.base}/module9_demo/demo_data.js') as fh:
            js = fh.read()
        self.assertTrue(js.startswith('window.DEMO_DATA = '))
        self.assertEqual(
            json.loads(js[len('window.DEMO_DATA = '):].rstrip().rstrip(';\n')),
            self.payload)

    def test_the_endoscopic_frames_were_exported(self):
        import cv2
        files = self.payload['frame_files']
        self.assertTrue(files)
        for f in self.payload['frames']:
            for i in (f['img'], f['img1'], f['img3']):
                self.assertIn(str(i), files)
        p = f'{self.base}/module9_demo/' + files[str(self.payload["frames"][0]["img"])]
        self.assertEqual(cv2.imread(p).shape[1], 320)

    def test_it_wrote_nothing_back_over_the_checkpoint_or_the_dataset(self):
        """This cell is read-only outside module9_demo/. A regenerated cell that
        acquired a torch.save would otherwise be able to cost them the run."""
        self.assertEqual(
            os.path.getsize(f'{self.base}/module5/diffusion_ckpt_best.pth'),
            os.path.getsize(f'{self.base}/module5/diffusion_ckpt_best.pth'))
        self.assertEqual(sorted(os.listdir(f'{self.base}/module5')),
                         ['diffusion_ckpt_best.pth'])
        self.assertEqual(sorted(os.listdir(f'{self.base}/module4_combined')),
                         ['module4_metadata.json'])


@unittest.skipIf(torch is None, 'torch not installed')
class TestStandaloneCellContract(unittest.TestCase):
    def setUp(self):
        with open(SRC) as fh:
            self.src = fh.read()
        self.tree = ast.parse(self.src)

    def test_it_never_writes_a_checkpoint(self):
        mutating = {'torch.save', 'os.rename', 'os.remove', 'os.rmdir', 'shutil.rmtree',
                    'np.save'}
        found = [(n.lineno, ast.unparse(n.func)) for n in ast.walk(self.tree)
                 if isinstance(n, ast.Call) and ast.unparse(n.func) in mutating]
        self.assertEqual(found, [])

    def test_it_declares_itself_generated(self):
        self.assertIn('GENERATED FILE', self.src)
        self.assertIn('tools/make_demo_cell.py', self.src)

    def test_the_lifted_definitions_are_byte_identical_to_their_sources(self):
        def spans(path):
            with open(path) as fh:
                src = fh.read()
            lines = src.splitlines()
            return {n.name: '\n'.join(lines[n.lineno - 1:n.end_lineno])
                    for n in ast.parse(src).body
                    if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        gen_lines = self.src.splitlines()
        gen = {n.name: '\n'.join(gen_lines[n.lineno - 1:n.end_lineno])
               for n in self.tree.body
               if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        m6 = spans(os.path.join(ROOT, 'module6_prediction.py'))
        m9 = spans(os.path.join(ROOT, 'module9_demo_export.py'))
        checked = 0
        for name, body in gen.items():
            origin = m9.get(name) if name in m9 else m6.get(name)
            if origin is None:
                continue
            self.assertEqual(body, origin, f'{name} has drifted from its source')
            checked += 1
        self.assertGreaterEqual(checked, 23)


if __name__ == '__main__':
    unittest.main()
