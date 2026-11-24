#!/usr/bin/env python3
"""Render the predicted future as video, with no overlay on it.

    python3 tools/make_prediction_video.py --horizon 3
    python3 tools/make_prediction_video.py --horizon 3 --gain 4 --compare

WHAT THIS IS
The demonstrator draws the prediction as dots because dots are what the pipeline
produces: 512 tracked anchor points, and no appearance model of any kind. There are no
Gaussians here and no texture, so there is nothing to "render" a future frame from.

What can honestly be built instead is a WARP. Each tracked point has a predicted
position k frames ahead; scattering those displacements over the image gives a dense
motion field, and pushing the last observed frame through it gives an image of where
the tissue is predicted to be. No dots, and every pixel is real endoscopic imagery.

WHAT IT IS NOT
It is not a predicted image. It moves pixels that already exist and can invent
nothing: tissue about to be revealed from behind a fold, an instrument entering the
view, a specular highlight that is about to appear -- none of that is predicted,
because none of it is in the geometry. Over the horizons this model supports the
tissue moves a few pixels, so the warp is small and the illusion holds; it would fall
apart over a second of video, and that is a property of the method, not a bug here.

The caption is on by default for that reason. A clip that looks like endoscopy but is
synthetic should say so on its face, especially one shown to clinicians. --no-caption
removes it.

HORIZON LIMIT
The checkpoint's horizon embedding is nn.Embedding(MAX_PRED_STEP + 1, D) with
MAX_PRED_STEP = 8, so k must be 1..8. k=10 has no row in that table and cannot be
asked for without retraining Module 5. Reading from the exported payload, k is
further limited to the horizons Module 9 exported (1 and 3).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit('needs opencv (cv2) for the warp and the video')
from scipy.interpolate import griddata

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE = os.path.join(REPO, 'demo', 'clinical')
MAX_PRED_STEP = 8          # the checkpoint's embedding table; see the docstring


def load_payload(path):
    with open(path, encoding='utf-8') as fh:
        text = fh.read()
    return json.loads(text[text.index('{'):text.rindex('}') + 1])


def dense_flow(src_px, dst_px, shape):
    """A dense displacement field from scattered point correspondences.

    Interpolated at the DESTINATION positions, because the warp is applied backwards:
    for each output pixel the sampler needs to know where that pixel came from. With
    a few pixels of motion the distinction barely shows, but it is free to get right.

    Linear inside the convex hull of the tracked points, nearest outside it. The
    alternative -- zero outside the hull -- would hold the image border rigid while
    the middle moved, and tear the picture along the hull.
    """
    h, w = shape
    disp = dst_px - src_px
    gy, gx = np.mgrid[0:h, 0:w]
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)

    flow = griddata(dst_px, disp, grid, method='linear')
    holes = np.isnan(flow[:, 0])
    if holes.any():
        flow[holes] = griddata(dst_px, disp, grid[holes], method='nearest')
    return flow.reshape(h, w, 2).astype(np.float32)


def warp(img, src_px, dst_px, gain=1.0):
    h, w = img.shape[:2]
    flow = dense_flow(src_px, dst_px, (h, w)) * float(gain)
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = gx - flow[..., 0]
    map_y = gy - flow[..., 1]
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LANCZOS4,
                     borderMode=cv2.BORDER_REPLICATE)


def caption(img, text, sub=None):
    out = img.copy()
    h, w = out.shape[:2]
    scale = w / 640.0
    cv2.rectangle(out, (0, h - int(34 * scale)), (w, h), (0, 0, 0), -1)
    cv2.putText(out, text, (int(10 * scale), h - int(20 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42 * scale, (255, 255, 255), 1, cv2.LINE_AA)
    if sub:
        cv2.putText(out, sub, (int(10 * scale), h - int(7 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34 * scale, (150, 150, 158), 1,
                    cv2.LINE_AA)
    return out


def evaluate(d, frames_dir, k):
    """Is the warped frame closer to what actually happened than doing nothing?

    The honest question about this video, and the one a pretty clip does not answer.
    Three warps are compared against the null of leaving the last frame alone:

      the model's prediction   -- what the video shows
      constant velocity        -- the baseline the paper reports against
      the TRUE future motion   -- the control. It is the ceiling this method could
                                  reach with a perfect predictor, so if it does not
                                  beat the null either, the limit is appearance
                                  change and no amount of prediction would help.
    """
    series = {'none': [], 'pred': [], 'cv': [], 'gt': []}
    for f in d['frames']:
        src = os.path.join(frames_dir, f'{f["img"]:05d}.jpg')
        tgt = os.path.join(frames_dir, f'{f[f"img{k}"]:05d}.jpg')
        if not (os.path.exists(src) and os.path.exists(tgt)):
            continue
        img, truth = cv2.imread(src), cv2.imread(tgt)
        if img is None or truth is None:
            continue
        h, w = img.shape[:2]
        px = np.asarray(f['px_ctx'], float).reshape(-1, 2) * [w, h]

        def mae(a):
            return float(np.abs(a.astype(float) - truth.astype(float)).mean())

        series['none'].append(mae(img))
        for field, name in ((f'px_pred{k}', 'pred'), (f'px_cv{k}', 'cv'),
                            (f'px_gt{k}', 'gt')):
            if f.get(field) is None:
                continue
            pk = np.asarray(f[field], float).reshape(-1, 2) * [w, h]
            series[name].append(mae(warp(img, px, pk)))

    if not series['none']:
        sys.exit(f'no stills found in {frames_dir} to evaluate against')
    null = float(np.mean(series['none']))
    print(f'\n  Mean absolute error against the recorded frame {k} ahead '
          f'({len(series["none"])} origins)\n')
    print(f'    {"warped by":<26}{"MAE":>8}{"vs doing nothing":>20}')
    print(f'    {"-" * 54}')
    print(f'    {"nothing (last frame)":<26}{null:>8.3f}{"":>20}')
    for name, label in (('pred', 'the model prediction'),
                        ('cv', 'constant velocity'),
                        ('gt', 'the TRUE future motion')):
        if series[name]:
            v = float(np.mean(series[name]))
            print(f'    {label:<26}{v:>8.3f}{v - null:>+20.3f}')
    print()
    if series['pred'] and float(np.mean(series['pred'])) > null:
        print('  The predicted warp is WORSE than leaving the frame alone. Resampling')
        print('  costs more than the predicted motion recovers at this horizon, so the')
        print('  video looks plausible and is measurably not an improvement. Do not')
        print('  present it as the model predicting the future.')
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--horizon', type=int, default=3,
                    help='frames ahead. Limited to what Module 9 exported.')
    ap.add_argument('--gain', type=float, default=1.0,
                    help='exaggerate the predicted motion. 1.0 is the real prediction; '
                         'anything else is a visualisation and is captioned as one.')
    ap.add_argument('--fps', type=int, default=8)
    ap.add_argument('--scale', type=int, default=3, help='upscale factor for the output')
    ap.add_argument('--out', default=os.path.join(REPO, 'demo', 'predicted_future.mp4'))
    ap.add_argument('--data', default=os.path.join(PAGE, 'demo_data.js'))
    ap.add_argument('--frames', default=os.path.join(PAGE, 'frames'))
    ap.add_argument('--compare', action='store_true',
                    help='also write a side-by-side against the recorded future frame')
    ap.add_argument('--no-caption', dest='caption', action='store_false')
    ap.add_argument('--evaluate', action='store_true',
                    help='measure the warp against the recorded future instead of '
                         'rendering. Run this before believing the video.')
    args = ap.parse_args(argv)

    k = args.horizon
    if not 1 <= k <= MAX_PRED_STEP:
        sys.exit(f'horizon {k} is outside the trained range 1..{MAX_PRED_STEP}. The '
                 f"checkpoint's horizon embedding has no row for it; predicting that "
                 f'far needs Module 5 retrained with a larger MAX_PRED_STEP.')
    if not os.path.exists(args.data):
        sys.exit(f'no exported payload at {args.data}')
    d = load_payload(args.data)
    if k not in d.get('horizons', []):
        sys.exit(f'the export carries horizons {d.get("horizons")}, not {k}. Re-export '
                 f'Module 9 with DEMO_HORIZONS including {k}, or pick one of those.')

    if args.evaluate:
        return evaluate(d, args.frames, k)

    tmp = tempfile.mkdtemp(prefix='predvid_')
    n, skipped = 0, 0

    for f in d['frames']:
        src_name = f'{f["img"]:05d}.jpg'
        src_path = os.path.join(args.frames, src_name)
        if not os.path.exists(src_path):
            skipped += 1
            continue
        img = cv2.imread(src_path)
        if img is None:
            skipped += 1
            continue
        h, w = img.shape[:2]
        px = np.asarray(f['px_ctx'], float).reshape(-1, 2) * [w, h]
        pk = np.asarray(f[f'px_pred{k}'], float).reshape(-1, 2) * [w, h]

        out = warp(img, px, pk, gain=args.gain)
        if args.scale != 1:
            out = cv2.resize(out, (w * args.scale, h * args.scale),
                             interpolation=cv2.INTER_LANCZOS4)
        if args.caption:
            gtxt = '' if args.gain == 1.0 else f'  ·  motion x{args.gain:g} (exaggerated)'
            out = caption(
                out,
                f'PREDICTED  +{k} frames{gtxt}',
                f'frame {f["img"]} warped by predicted geometry · not recorded video')
        cv2.imwrite(os.path.join(tmp, f'{n:05d}.png'), out)
        n += 1

    if not n:
        sys.exit(f'no frames written: {args.frames} has none of the stills the payload '
                 f'names (they are gitignored -- copy them beside the page first)')

    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(args.fps),
           '-i', os.path.join(tmp, '%05d.png'), '-c:v', 'libx264',
           '-pix_fmt', 'yuv420p', '-crf', '18', args.out]
    subprocess.run(cmd, check=True)
    size = os.path.getsize(args.out) / 1024
    print(f'  {args.out}  ({n} frames, {size:.0f} KB, {args.fps} fps)')
    if skipped:
        print(f'  {skipped} origin(s) skipped: still not found beside the page')

    if args.compare:
        comp = re.sub(r'\.mp4$', '_vs_truth.mp4', args.out)
        n2 = 0
        for f in d['frames']:
            src = os.path.join(args.frames, f'{f["img"]:05d}.jpg')
            tgt = os.path.join(args.frames, f'{f[f"img{k}"]:05d}.jpg')
            if not (os.path.exists(src) and os.path.exists(tgt)):
                continue
            img, truth = cv2.imread(src), cv2.imread(tgt)
            h, w = img.shape[:2]
            px = np.asarray(f['px_ctx'], float).reshape(-1, 2) * [w, h]
            pk = np.asarray(f[f'px_pred{k}'], float).reshape(-1, 2) * [w, h]
            pred = warp(img, px, pk, gain=args.gain)
            if args.caption:
                pred = caption(pred, f'PREDICTED  +{k}', 'warped, not recorded')
                truth = caption(truth, f'RECORDED  frame {f[f"img{k}"]}', 'what happened')
            pair = np.hstack([pred, truth])
            if args.scale != 1:
                pair = cv2.resize(pair, (pair.shape[1] * args.scale,
                                         pair.shape[0] * args.scale),
                                  interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(os.path.join(tmp, f'c{n2:05d}.png'), pair)
            n2 += 1
        if n2:
            subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate',
                            str(args.fps), '-i', os.path.join(tmp, 'c%05d.png'),
                            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18',
                            comp], check=True)
            print(f'  {comp}  ({n2} frames, side by side with the recorded future)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
