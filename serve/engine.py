"""Live inference for the clinical demonstrator, on a CPU laptop.

Why this file exists at all: the demonstrator's exported payload is fixed at the
origins Module 9 happened to choose. Live inference lets the panel pick any origin
in the held-out tail and watch the model answer. The cost of that is 1 h 15 m of
Colab GPU time replaced by ~10 s of ThinkPad CPU time per prediction, which is only
possible because of the one observation below.

THE OBSERVATION
---------------
`ConditionalDenoiser.forward` recomputes `encode_context(ctx)` on every call.
`ddim_sample` calls it once per diffusion step. But across the steps of a single
prediction, `ctx` and `hstep` never change -- only `x_noisy` and `t` do. So the
context encoder, which is a pointwise MLP plus a 3-layer transformer over
N_ANCHOR x WINDOW_SIZE = 16,384 tokens, runs 20 times to produce 20 identical
tensors.

Measured on the target machine (Core Ultra 5 225U, 12 threads, N=512, W=32):

    full forward      2299 ms
    encode_context    2206 ms   (95.9 % of it)
    20-step DDIM, re-encoding every step   126.72 s
    20-step DDIM, context encoded once       9.33 s

and the two outputs differ by exactly 0.0 -- not "within tolerance", bit-for-bit
identical, because hoisting a pure function out of a loop over inputs it does not
depend on is an algebraic identity, not an approximation. `test_engine.py` asserts
that equality against the real checkpoint so it cannot rot.

WHAT THIS IS NOT
----------------
It is not a reimplementation of the model. `gasr_arch` lifts the network out of
module5_diffusion.py verbatim. It is not a second prediction path either: the
arithmetic below mirrors module6's `predict_direct` (cv baseline + calibrated
model correction), and the sampler mirrors module5's `ddim_sample` including eta.
"""
import os
import time

import numpy as np
import torch

from gasr_arch import build_namespace

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CKPT = os.path.join(REPO_ROOT, 'diffusion_ckpt_best.pth')
DEFAULT_BUNDLE = os.path.join(REPO_ROOT, 'serve', 'live_bundle.npz')

# Module 6's settings, which every number in the paper and in the exported payload was
# measured with. Changing either makes the live demo and the paper disagree, so they
# are named here once and not exposed as knobs on the request.
DDIM_STEPS = 20
DDIM_SAMPLES = 4


class Engine:
    def __init__(self, ckpt_path=DEFAULT_CKPT, bundle_path=DEFAULT_BUNDLE, threads=None):
        if threads:
            torch.set_num_threads(int(threads))
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f'{ckpt_path} is missing. It is 261 MB and deliberately not committed; '
                f'download it from Drive before starting the server.')
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        self.arch_cfg = ck['arch_cfg']
        self.config = ck['config']
        self.epoch = ck.get('epoch')

        ns = build_namespace(self.arch_cfg, self.config)
        self._ns = ns
        self.model = ns['ConditionalDenoiser']()
        # strict=True: a partial load would run and predict nonsense.
        self.model.load_state_dict(ck['ema'], strict=True)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.n_params = sum(p.numel() for p in self.model.parameters())

        self.T = self.arch_cfg['T_DIFF']
        self.window = self.arch_cfg['WINDOW_SIZE']
        self.n_anchor = self.arch_cfg['N_ANCHOR']
        self.target_mode = self.arch_cfg['TARGET_MODE']
        self.data_scale = float(self.config['DATA_SCALE'])

        betas = ns['cosine_beta_schedule'](self.T)
        self.ac = torch.cumprod(1.0 - betas, dim=0)
        self.sqrt_ac = torch.sqrt(self.ac)
        self.sqrt_1m_ac = torch.sqrt(1.0 - self.ac)

        self.bundle = None
        if bundle_path and os.path.exists(bundle_path):
            self.load_bundle(bundle_path)

    # ── the live data bundle ──────────────────────────────────────────────────
    def load_bundle(self, path):
        """Module 4's tracked trajectory for one video, exported by
        serve/export_live_bundle.py. Small (a few MB) because it is T x N_ANCHOR x 3
        floats and nothing else; the frames are already beside the page."""
        z = np.load(path, allow_pickle=True)
        traj = np.asarray(z['traj_n'], dtype=np.float32)
        if traj.ndim != 3 or traj.shape[1] != self.n_anchor or traj.shape[2] != 3:
            raise ValueError(
                f'live_bundle traj_n is {traj.shape}; this checkpoint needs '
                f'(T, {self.n_anchor}, 3). Feeding it a different anchor count would '
                f'run, and would not be the model the paper measured.')
        meta = {k: z[k].item() if z[k].shape == () else z[k] for k in z.files
                if k != 'traj_n'}
        self.bundle = {'traj_n': traj, **meta}
        return self.bundle

    @property
    def ready(self):
        return self.bundle is not None

    @property
    def disp_norm(self):
        """Module 4's per-video displacement normaliser -- one scalar, not per frame."""
        return float(np.asarray(self.bundle['disp_norm']).reshape(-1)[0])

    @property
    def horizons(self):
        return [int(x) for x in np.asarray(self.bundle.get('horizons', [1, 3])).reshape(-1)]

    def alpha_for(self, k):
        """The magnitude calibration Module 6 fitted on TRAINING origins.

        It travels in the bundle rather than being refitted here: refitting at serve
        time on whatever origins the panel happens to visit would fit the calibration
        to the held-out data it is being evaluated on.
        """
        a = self.bundle.get(f'alpha_{int(k)}')
        return 1.0 if a is None else float(np.asarray(a).reshape(-1)[0])

    # ── the sampler ───────────────────────────────────────────────────────────
    def _x0_from_v(self, x, v, t):
        return self.sqrt_ac[t][:, None, None] * x - self.sqrt_1m_ac[t][:, None, None] * v

    def _eps_from_v(self, x, v, t):
        return self.sqrt_1m_ac[t][:, None, None] * x + self.sqrt_ac[t][:, None, None] * v

    def _forward_with_context(self, ctx_latent, h_emb, x_noisy, t):
        """`ConditionalDenoiser.forward` from the point after the context encoder.

        Mirrors it line for line; the only change is that `ctx_latent` arrives as an
        argument instead of being recomputed. Note `ctx_upd` is still recomputed every
        step -- it depends on the target branch -- so nothing step-dependent is cached.
        """
        m = self.model
        tgt = m.target_geo(x_noisy)
        tgt = tgt + m.time_mlp(
            self._ns['sinusoidal_embedding'](t, tgt.shape[-1]))[:, None, :]
        tgt = tgt + h_emb
        attn_t, _ = m.cross_t2c(tgt, ctx_latent, ctx_latent)
        tgt_upd = m.norm_t(tgt + attn_t)
        attn_c, _ = m.cross_c2t(ctx_latent, tgt, tgt)
        ctx_upd = m.norm_c(ctx_latent + attn_c)
        return m.output_mlp(torch.cat([tgt_upd, ctx_upd], dim=-1))

    @torch.no_grad()
    def encode(self, ctx, hstep):
        """The invariant half of the forward pass: 95.9 % of its cost, computed once."""
        m = self.model
        h_emb = m.horizon_mlp(m.horizon_emb(hstep))[:, None, :]
        return m.encode_context(ctx) + h_emb, h_emb

    @torch.no_grad()
    def ddim(self, ctx, hstep, steps=DDIM_STEPS, n_samples=DDIM_SAMPLES, eta=0.0,
             generator=None, cached=True, encode_full_batch=False):
        """module5's `ddim_sample`, with the context encoder hoisted out of the loop.

        `cached=False` runs the original, re-encoding every step. It exists so the test
        suite can assert the two agree exactly; it is 13.6x slower and never serves a
        request.
        """
        B, N_A = ctx.shape[0], ctx.shape[2]
        if n_samples > 1:
            ctx_s = ctx.repeat_interleave(n_samples, dim=0)
            hstep_s = hstep.repeat_interleave(n_samples, dim=0)
        else:
            ctx_s, hstep_s = ctx, hstep
        BK = ctx_s.shape[0]

        ts = torch.linspace(self.T - 1, 0, steps).long()
        x = torch.randn(BK, N_A, 3, generator=generator)

        if cached:
            # repeat_interleave duplicates the context row, so all BK rows have the
            # same latent. Encoding ONE row and expanding computes the same quantity
            # for a quarter of the work.
            #
            # It is the same quantity but not the same float32: a matmul over batch 1
            # and over batch 4 use different kernels and sum in a different order, so
            # the two roundings of it differ in the last bits (measured: 7.2e-07 on
            # the model's output, which is 0.01 % of one frame of tissue motion after
            # denormalisation). `encode_full_batch=True` encodes at BK instead and is
            # then bit-identical to the unmodified sampler -- the test suite uses it to
            # prove the expansion is arithmetic, not a mistake.
            if encode_full_batch and n_samples > 1:
                ctx_latent, h_emb = self.encode(ctx_s, hstep_s)
            else:
                ctx_latent, h_emb = self.encode(ctx, hstep)
                if n_samples > 1:
                    ctx_latent = ctx_latent.repeat_interleave(n_samples, dim=0)
                    h_emb = h_emb.repeat_interleave(n_samples, dim=0)

        for i, t in enumerate(ts):
            tb = t.expand(BK)
            if cached:
                v = self._forward_with_context(ctx_latent, h_emb, x, tb)
            else:
                v = self.model(ctx_s, x, tb, hstep_s)
            x0 = self._x0_from_v(x, v, tb)
            eps = self._eps_from_v(x, v, tb)
            if i == len(ts) - 1:
                x = x0
                break
            t_next = ts[i + 1]
            ac_t, ac_next = self.ac[t], self.ac[t_next]
            sigma = eta * torch.sqrt((1 - ac_next) / (1 - ac_t)) \
                        * torch.sqrt(1 - ac_t / ac_next)
            x = torch.sqrt(ac_next) * x0 \
                + torch.sqrt(torch.clamp(1 - ac_next - sigma ** 2, min=0.0)) * eps
            if eta > 0:
                x = x + sigma * torch.randn(BK, N_A, 3, generator=generator)

        if n_samples > 1:
            x = x.view(B, n_samples, N_A, 3).mean(dim=1)
        return x

    # ── the prediction, as module6 defines it ─────────────────────────────────
    def const_velocity(self, ctx, k):
        last, prev = ctx[:, -1], ctx[:, -2]
        return last + (last - prev) * float(k)

    def base_pred(self, ctx, k):
        if self.target_mode == 'cv_residual':
            return self.const_velocity(ctx, k)
        return ctx[:, -1]

    @torch.no_grad()
    def predict(self, ctx, k, disp_norm, alpha=1.0, steps=DDIM_STEPS,
                n_samples=DDIM_SAMPLES, eta=0.0, generator=None, cached=True):
        """module6's `predict_direct`: the cv baseline plus the calibrated correction.

        ctx: (B, W, N_A, 3) normalised positions. Returns (B, N_A, 3) absolute geometry.
        """
        hstep = torch.full((ctx.shape[0],), int(k), dtype=torch.long)
        x0 = self.ddim(ctx, hstep, steps=steps, n_samples=n_samples, eta=eta,
                       generator=generator, cached=cached)
        dn = torch.as_tensor(disp_norm, dtype=torch.float32).reshape(-1)
        physical = (x0 / self.data_scale) / dn[:, None, None]
        return self.base_pred(ctx, k) + float(alpha) * physical

    # ── requests ──────────────────────────────────────────────────────────────
    # Module 9 indexes an origin `t` as: context = tn[t-W : t], last observed frame =
    # tn[t-1], truth at horizon k = tn[t+k-1], still image = t-1. The payload's
    # frames[].t carry exactly those t values, so the live path uses the same
    # convention -- off by one here would put the live and exported rows for the same
    # labelled frame on different frames, which is the sort of discrepancy a panel
    # finds by scrubbing and nobody can explain on the spot.
    def origin_bounds(self):
        """The origins a live request may ask for: held-out only.

        The lower bound is Module 9's own: max(W, split_frame). Module 4 split each
        video 80/20 temporally, and an origin before split_frame contributed gradient
        steps. Letting the panel scrub into training frames would show the model at its
        most flattering for the one reason that proves nothing, so the server refuses
        rather than serving it with a caveat nobody reads.
        """
        traj = self.bundle['traj_n']
        lo = max(self.window, int(self.bundle.get('split_frame', self.window)))
        hi = traj.shape[0] - max(int(k) for k in self.horizons)
        return lo, hi

    def window_at(self, origin):
        """The W context frames of origin `t`, as Module 9 slices them."""
        if not self.ready:
            raise RuntimeError('no live bundle loaded')
        traj = self.bundle['traj_n']
        lo, hi = self.origin_bounds()
        if origin < lo:
            # Two different refusals, because they mean different things and a panel
            # will ask which. Below the split the frames trained the model; above the
            # top there is no recorded outcome left to compare a prediction against.
            raise IndexError(
                f'origin {origin} is before the held-out tail, which starts at {lo}. '
                f'Earlier frames trained the model, so predicting one would flatter '
                f'it for a reason that proves nothing.')
        if origin >= hi:
            kmax = max(self.horizons)
            raise IndexError(
                f'origin {origin} is past the last origin, {hi - 1}. Beyond that the '
                f'video ends before the {kmax}-frame horizon, so there is no recorded '
                f'outcome to compare against.')
        return torch.from_numpy(traj[origin - self.window:origin]).unsqueeze(0)

    def truth_at(self, origin, k):
        """What actually happened k frames after origin `t`: Module 9's tn[t+k-1]."""
        traj = self.bundle['traj_n']
        j = origin + k - 1
        if j >= traj.shape[0]:
            return None
        return torch.from_numpy(traj[j]).unsqueeze(0)

    def to_world(self, arr):
        """Normalised -> camera space, Module 9's `to_world`: a * scl + ctr.

        The overlay's projection expects camera-space points. Projecting normalised
        ones instead would land every dot in the wrong place while still looking like
        a plausible overlay, so scl/ctr travel in the bundle and are applied here.
        """
        b = self.bundle
        return np.asarray(arr) * np.asarray(b['scl']) + np.asarray(b['ctr'])

    @torch.no_grad()
    def predict_origin(self, origin, k, hypotheses=0):
        """Everything the page draws for one origin and one horizon, computed live."""
        t0 = time.perf_counter()
        ctx = self.window_at(origin)
        dn = self.disp_norm
        alpha = self.alpha_for(k)

        pred = self.predict(ctx, k, [dn], alpha=alpha)
        cv = self.const_velocity(ctx, k)
        gt = self.truth_at(origin, k)

        out = {
            'origin': int(origin), 'k': int(k), 'alpha': alpha,
            'ctx': ctx[0, -1], 'pred': pred[0], 'cv': cv[0],
            'gt': None if gt is None else gt[0],
            'steps': DDIM_STEPS, 'samples': DDIM_SAMPLES,
        }
        if hypotheses > 0:
            hyp = self.predict(ctx.repeat(hypotheses, 1, 1, 1), k, [dn] * hypotheses,
                               alpha=alpha, eta=1.0, n_samples=1)
            out['sigma'] = hyp.std(dim=0).norm(dim=-1)
        out['seconds'] = time.perf_counter() - t0
        return out
