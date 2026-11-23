"""Guards on the local inference server.

The server's whole claim is that it produces the SAME prediction as the Colab path at
a thirteenth of the cost. These tests exist to stop that claim quietly becoming false.

The equivalence tests run the unmodified sampler, which re-encodes the context on every
diffusion step and takes about two minutes per call on a laptop CPU. They are therefore
gated behind GASR_SLOW=1 and are not part of the ordinary run -- but they are the only
tests here that matter, so run them before the viva and after any change to engine.py:

    GASR_SLOW=1 python3 -m unittest tests.test_serve_engine -v
"""
import json
import os
import sys
import unittest

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVE = os.path.join(REPO, 'serve')
sys.path.insert(0, SERVE)

CKPT = os.path.join(REPO, 'diffusion_ckpt_best.pth')
HAVE_CKPT = os.path.exists(CKPT)
SLOW = os.environ.get('GASR_SLOW') == '1'


def _fake_bundle(path, T=300, N=512, split=240):
    rng = np.random.default_rng(7)
    base = rng.normal(0, 0.4, (N, 3)).astype(np.float32)
    t = np.arange(T, dtype=np.float32)[:, None, None]
    traj = (base[None] + 0.004 * t * np.array([1.0, 0.3, -0.2], dtype=np.float32)
            + 0.0015 * np.sin(t / 7.0) * base[None]).astype(np.float32)
    cam = {'fx': 417.0321193091858, 'fy': 417.0321193091858, 'cx': 320.0, 'cy': 180.0,
           'W': 640, 'H': 360, 'calibrated': False, 'source': 'assumed-fov'}
    np.savez_compressed(
        path, traj_n=traj, disp_norm=np.float32(0.0042),
        scl=np.asarray([.9, .9, .9], dtype=np.float32),
        ctr=np.asarray([0., 0., 8.], dtype=np.float32),
        split_frame=np.int64(split), window=np.int64(32),
        horizons=np.asarray([1, 3], dtype=np.int64), video=np.asarray('fake'),
        workspace=np.asarray('ws'), camera=np.asarray(json.dumps(cam)),
        calibrated=np.bool_(True), alpha_1=np.float32(0.9), alpha_3=np.float32(1.02))
    return path


class TestNoSecondCopyOfTheModel(unittest.TestCase):
    """serve/ must read the architecture out of the training script, never restate it.

    A second definition of the network is the one bug in this design that would not
    announce itself: the weights would load, the server would start, and every number
    would be subtly not the paper's.
    """

    def test_serve_defines_no_model_classes(self):
        for name in os.listdir(SERVE):
            if not name.endswith('.py'):
                continue
            src = open(os.path.join(SERVE, name)).read()
            for banned in ('class ConditionalDenoiser', 'class GeomMLP'):
                self.assertNotIn(banned, src,
                                 f'{name} defines {banned}; it must be lifted from '
                                 f'module5_diffusion.py instead')

    def test_extraction_fails_loudly_on_rename(self):
        import gasr_arch
        with self.assertRaises(RuntimeError):
            gasr_arch.extract_source(names=('NoSuchThingInModule5',))

    def test_server_needs_no_third_party_web_framework(self):
        src = open(os.path.join(SERVE, 'server.py')).read()
        for banned in ('fastapi', 'uvicorn', 'flask', 'starlette'):
            self.assertNotIn(banned, src.lower().replace('no fastapi', '')
                                        .replace('no uvicorn', ''),
                             f'server.py reaches for {banned}; the demonstrator must '
                             f'run with nothing installed')

    def test_binds_loopback_only(self):
        src = open(os.path.join(SERVE, 'server.py')).read()
        self.assertIn("'127.0.0.1'", src)
        self.assertNotIn("'0.0.0.0'", src)


@unittest.skipUnless(HAVE_CKPT, 'diffusion_ckpt_best.pth not present')
class TestEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from engine import Engine
        import tempfile
        cls.tmp = tempfile.mkdtemp()
        cls.eng = Engine(bundle_path=_fake_bundle(os.path.join(cls.tmp, 'b.npz')))

    def test_parameter_count_is_the_papers(self):
        self.assertEqual(self.eng.n_params, 16_318_467)

    def test_epoch_is_the_reported_checkpoint(self):
        self.assertEqual(int(self.eng.epoch), 100)

    def test_window_slicing_matches_module9(self):
        """Module 9: ctx = tn[t-W:t], last observed = tn[t-1], gt_k = tn[t+k-1].

        Off by one here and the live and exported rows for the same labelled frame are
        different frames.
        """
        traj = self.eng.bundle['traj_n']
        t = self.eng.origin_bounds()[0] + 5
        w = self.eng.window_at(t).numpy()[0]
        self.assertEqual(w.shape, (32, 512, 3))
        np.testing.assert_allclose(w[-1], traj[t - 1])
        np.testing.assert_allclose(w[0], traj[t - 32])
        np.testing.assert_allclose(self.eng.truth_at(t, 1).numpy()[0], traj[t])
        np.testing.assert_allclose(self.eng.truth_at(t, 3).numpy()[0], traj[t + 2])

    def test_refuses_training_origins(self):
        lo, hi = self.eng.origin_bounds()
        self.assertEqual(lo, 240, 'lower bound must be max(W, split_frame)')
        for bad in (lo - 1, 0, 31, hi, hi + 50):
            with self.assertRaises(IndexError):
                self.eng.window_at(bad)

    def test_the_two_refusals_say_different_things(self):
        # Below the split and past the end are refused for unrelated reasons. One is
        # "those frames trained the model"; the other is "the video ends before the
        # horizon, so there is nothing to compare against". A panel will ask which,
        # and a single message for both would be wrong half the time.
        lo, hi = self.eng.origin_bounds()
        with self.assertRaises(IndexError) as under:
            self.eng.window_at(lo - 1)
        with self.assertRaises(IndexError) as over:
            self.eng.window_at(hi)
        self.assertIn('trained the model', str(under.exception))
        self.assertNotIn('trained the model', str(over.exception))
        self.assertIn('no recorded outcome', str(over.exception))

    def test_prediction_is_cv_plus_calibrated_correction(self):
        """pred must be exactly module6's `base_pred + alpha * to_physical(x0)`.

        Reconstructed here from the pieces rather than trusted, so a change to the
        arithmetic cannot pass unnoticed.
        """
        import torch
        ctx = self.eng.window_at(250)
        g = torch.Generator().manual_seed(11)
        pred = self.eng.predict(ctx, 3, [self.eng.disp_norm], alpha=1.02,
                                n_samples=1, steps=4, generator=g)
        g2 = torch.Generator().manual_seed(11)
        x0 = self.eng.ddim(ctx, torch.tensor([3]), steps=4, n_samples=1, generator=g2)
        # Rebuilt with the same tensor operations in the same order, so this can assert
        # exact equality. Dividing by a Python float instead reassociates in float32 and
        # differs in the last bit -- which would force a tolerance here and hide the
        # sign and factor errors this test exists to catch.
        dn = torch.as_tensor([self.eng.disp_norm], dtype=torch.float32).reshape(-1)
        # Grouped as predict() groups it: alpha * ((x0 / scale) / dn). Multiplying
        # before the second divide reassociates and differs in the last float32 bit.
        expect = (self.eng.base_pred(ctx, 3)
                  + 1.02 * ((x0 / self.eng.data_scale) / dn[:, None, None]))
        np.testing.assert_allclose(pred.numpy(), expect.numpy(), rtol=0, atol=0)

    @unittest.skipUnless(SLOW, 'set GASR_SLOW=1 (runs the 13x slower sampler)')
    def test_hoisting_the_context_encoder_is_exact(self):
        # The load-bearing claim, at n_samples=1 where it is exactly testable:
        # encode_context does not depend on the timestep or the noisy target, so
        # computing it once instead of twenty times is an identity.
        import torch
        ctx = self.eng.window_at(250)
        h = torch.tensor([3])
        a = self.eng.ddim(ctx, h, steps=20, n_samples=1,
                          generator=torch.Generator().manual_seed(2026), cached=True)
        b = self.eng.ddim(ctx, h, steps=20, n_samples=1,
                          generator=torch.Generator().manual_seed(2026), cached=False)
        self.assertEqual((a - b).abs().max().item(), 0.0,
                         'hoisting the context encoder changed the result')

    @unittest.skipUnless(SLOW, 'set GASR_SLOW=1 (runs the 13x slower sampler)')
    def test_expanding_one_encoded_context_is_arithmetic_not_a_shortcut(self):
        # At n_samples>1 the fast path encodes ONE context row and expands it to all
        # four, because repeat_interleave made them identical. That is the same
        # quantity -- but a matmul over batch 1 and over batch 4 sum in a different
        # order, so the two float32 roundings differ in the last bits.
        #
        # This test separates the two possibilities. Encoding at the FULL interleaved
        # batch must be bit-identical to the unmodified sampler: if it is, the
        # expansion is correct and the only difference is batch-dependent rounding. If
        # that assertion ever fails, the expansion is a genuine bug.
        import torch
        ctx = self.eng.window_at(250)
        h = torch.tensor([3])

        def run(**kw):
            return self.eng.ddim(ctx, h, steps=20, n_samples=4,
                                 generator=torch.Generator().manual_seed(2026), **kw)

        plain = run(cached=False)
        exact = run(cached=True, encode_full_batch=True)
        self.assertEqual((exact - plain).abs().max().item(), 0.0,
                         'encoding at the full batch must reproduce the original '
                         'exactly; if it does not, the expansion is wrong')

        fast = run(cached=True)
        gap = (fast - plain).abs().max().item()
        # Bounded, and stated in the units the page actually draws rather than left as
        # a bare float. One frame of tissue motion is ~5e-03 in normalised units; this
        # has to be orders of magnitude below that to be irrelevant, which is the only
        # claim being made for it.
        denorm = gap / (self.eng.data_scale * self.eng.disp_norm)
        self.assertLess(gap, 1e-5, f'batching rounding grew to {gap:.2e}')
        self.assertLess(denorm, 1e-5,
                        f'after denormalisation the disagreement is {denorm:.2e}, '
                        f'which is no longer negligible against ~5e-03 of motion')

    @unittest.skipUnless(SLOW, 'set GASR_SLOW=1 (runs the 13x slower sampler)')
    def test_cached_sampler_is_bit_identical_with_eta(self):
        """Same, on the stochastic path the uncertainty halo uses (eta=1)."""
        import torch
        ctx = self.eng.window_at(250)
        h = torch.tensor([1])
        a = self.eng.ddim(ctx, h, steps=20, n_samples=1, eta=1.0,
                          generator=torch.Generator().manual_seed(5), cached=True)
        b = self.eng.ddim(ctx, h, steps=20, n_samples=1, eta=1.0,
                          generator=torch.Generator().manual_seed(5), cached=False)
        self.assertEqual((a - b).abs().max().item(), 0.0)


@unittest.skipUnless(HAVE_CKPT, 'diffusion_ckpt_best.pth not present')
class TestServerContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tempfile
        import server
        from engine import Engine
        cls.S = server
        cls.tmp = tempfile.mkdtemp()
        cls.live = server.Live(Engine(bundle_path=_fake_bundle(
            os.path.join(cls.tmp, 'b.npz'))))

    def test_health_reports_what_the_page_needs(self):
        h = self.live.health()
        for key in ('live', 'checkpoint_epoch', 'params', 'window', 'n_anchor',
                    'steps', 'samples', 'origin_min', 'origin_max', 'horizons',
                    'n_points', 'split_frame', 'calibration'):
            self.assertIn(key, h)
        self.assertEqual(h['steps'], 20)
        self.assertEqual(h['samples'], 4)

    def test_health_is_json_serialisable(self):
        json.loads(json.dumps(self.live.health()))

    def test_projection_goes_through_world_space(self):
        """px_* must be project_norm(to_world(x)), not project_norm(x).

        Projecting normalised coordinates produces a confident, entirely wrong overlay.
        """
        src = open(os.path.join(SERVE, 'server.py')).read()
        self.assertIn('project_norm(self.e.to_world(a), cam)', src)

    def test_display_point_count_comes_from_the_exporter(self):
        # Not a literal here. If the live arrays and the exported arrays disagree on
        # how many points they hold, the page unpacks triples off the end of one of
        # them -- wrong numbers rather than an error.
        import re
        with open(os.path.join(REPO, 'module9_demo_export.py')) as fh:
            want = int(re.search(r'^DEMO_POINTS\s*=\s*(\d+)', fh.read(),
                                 re.M).group(1))
        self.assertEqual(self.S.DEMO_POINTS, want)
        self.assertEqual(self.live.health()['n_points'], want)

    def test_page_dir_is_the_built_demonstrator(self):
        self.assertTrue(os.path.isfile(os.path.join(self.S.PAGE_DIR, 'index.html')))


if __name__ == '__main__':
    unittest.main()
