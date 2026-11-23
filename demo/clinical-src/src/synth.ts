import type { Payload, Frame } from './core';

// Placeholder geometry, so the app runs and can be rehearsed before Module 9 has been
// exported, and so it can be reviewed without the real data present.
//
// GEOMETRY ONLY. This file never produces imagery, and never image coordinates: a
// drawn or generated "endoscopic" picture would be synthetic clinical imagery on a
// projector, which the spec forbids even when labelled. Scenes that ask for a frame
// get the "imagery comes from the export" panel instead.
//
// mean_error is null, deliberately and permanently. The app derives every comparative
// sentence from mean_error, so a null here means placeholder mode cannot state a
// verdict even by accident: there is nothing for it to read.
export function synth(): Payload {
  const N = 256, F = 80, H = [1, 3];
  const frames: Frame[] = [];

  const base: [number, number, number][] = [];
  for (let i = 0; i < N; i++) {
    const u = (i % 16) / 15 - 0.5, v = Math.floor(i / 16) / 15 - 0.5;
    base.push([u * 1.6, v * 1.6, 0.25 * Math.sin(u * 5) * Math.cos(v * 4)]);
  }

  const pos = (t: number) => base.map(([x, y, z]) => {
    const br = 0.055 * Math.sin(t * 0.16);
    const lo = 0.035 * Math.sin(t * 0.27 + x * 3.1 + y * 2.3);
    return [x + lo * 0.35, y + br * 0.5, z + br + lo] as [number, number, number];
  });

  const flat = (pts: [number, number, number][]) => {
    const out = new Array<number>(pts.length * 3);
    for (let i = 0; i < pts.length; i++) {
      out[i * 3] = pts[i][0]; out[i * 3 + 1] = pts[i][1]; out[i * 3 + 2] = pts[i][2];
    }
    return out;
  };

  for (let n = 0; n < F; n++) {
    const t = n + 8;
    const cur = pos(t - 1), prev = pos(t - 2);
    const f: Frame = { t, ctx: flat(cur), img: t - 1 };

    for (const k of H) {
      const gt = pos(t + k - 1);
      f['gt' + k] = flat(gt);
      f['cv' + k] = flat(cur.map((p, i) =>
        p.map((c, d) => c + (c - prev[i][d]) * k) as [number, number, number]));
      // Error roughly flat in k while constant velocity's grows with it -- the shape of
      // the real finding, so rehearsing here teaches nothing the measured data denies.
      f['pred' + k] = flat(gt.map((p, i) =>
        p.map((c, d) => c + Math.sin(i * 2.7 + d + n * 0.1) * 0.0045) as [number, number, number]));
      f['sigma' + k] = base.map((_, i) => 0.004 + 0.004 * Math.abs(Math.sin(i * 1.3 + n * 0.08)));
      f['img' + k] = t + k - 1;
    }
    frames.push(f);
  }

  return {
    video: 'synthetic', workspace: 'synthetic',
    n_points: N, n_frames: F, horizons: H, window: 8,
    centre: [0, 0, 0], radius: 1.0, span: 2.0,
    synthetic: true, target_mode: '—', checkpoint_epoch: '—',
    mean_error: null, motion: null,
    calibrated: false, alpha: null, degenerate_horizons: [],
    frame_files: {}, timing: { timing_ok: false },
    frames,
  };
}
