import { useEffect, useRef } from 'react';
import type { Payload, Scene } from './types';
import { COLOUR, SERIES_LABEL } from './types';
import { STR } from './scenes';
import { displayGain, measuredCamera, unflat, type Point } from './core';

interface Props {
  data: Payload;
  scene: Scene;
  k: number;
  frame: number;
  placeholder: boolean;
  yaw: number;
  pitch: number;
  zoom: number;
  onOrbit: (dYaw: number, dPitch: number) => void;
  onZoom: (factor: number) => void;
}

// Median tissue motion over this horizon in source pixels, and the round factor that
// lifts it to something visible across a room. Memoised per horizon because it is a
// property of the sequence, not of the frame on screen.
const gainCache = new Map<string, number>();
export function overlayGain(d: Payload, k: number): number {
  const key = `${d.video}:${k}`;
  const hit = gainCache.get(key);
  if (hit != null) return hit;
  let g = 1;
  const f = d.frames?.[0];
  const a = f?.px_ctx as number[] | undefined;
  const b = f?.[`px_gt${k}`] as number[] | undefined;
  if (a && b && d.camera) {
    const W = d.camera.W || 640, H = d.camera.H || 360;
    const dists: number[] = [];
    for (let i = 0; i < d.n_points; i++) {
      const dx = (b[i * 2] - a[i * 2]) * W, dy = (b[i * 2 + 1] - a[i * 2 + 1]) * H;
      dists.push(Math.hypot(dx, dy));
    }
    dists.sort((x, y) => x - y);
    g = displayGain(dists[Math.floor(dists.length / 2)], 45);
  }
  gainCache.set(key, g);
  return g;
}

export function Stage({ data, scene, k, frame, placeholder, yaw, pitch, zoom,
                        onOrbit, onZoom }: Props) {
  const geomRef = useRef<HTMLCanvasElement>(null);
  const ovlRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const drag = useRef<[number, number] | null>(null);
  // Stills that already failed to load. Without this the panel is torn down and the
  // broken-image box re-exposed on every scene change and every playback tick.
  const failed = useRef(new Set<string>());

  const f = data.frames[frame];
  const wantsFrame = scene.layers.includes('frame') || scene.layers.includes('overlay');
  const hasGeometry = scene.layers.some(l => l !== 'frame' && l !== 'overlay');

  const files = data.frame_files ?? {};
  const imgIdx = f?.img != null ? String(f.img) : null;
  const candidate = imgIdx != null ? files[imgIdx] : undefined;
  const src = candidate && !failed.current.has(candidate) ? candidate : null;

  // ── geometry canvas ───────────────────────────────────────────────────────
  useEffect(() => {
    const cv = geomRef.current;
    if (!cv || !hasGeometry) return;
    const box = cv.parentElement!.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    cv.width = Math.max(1, box.width * dpr);
    cv.height = Math.max(1, box.height * dpr);
    const g = cv.getContext('2d');
    if (!g) return;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, box.width, box.height);
    if (!f) return;

    const W = box.width, H = box.height;
    const c = data.centre, s = 1 / (data.radius || 1);
    const project = (p: Point): [number, number, number] => {
      let x = (p[0] - c[0]) * s, y = (p[1] - c[1]) * s, z = (p[2] - c[2]) * s;
      const cy = Math.cos(yaw), sy = Math.sin(yaw);
      const x1 = x * cy - z * sy, z1 = x * sy + z * cy;
      const cp = Math.cos(pitch), sp = Math.sin(pitch);
      const y1 = y * cp - z1 * sp, z2 = y * sp + z1 * cp;
      const m = Math.min(W, H) * 0.4 * zoom;
      return [W / 2 + x1 * m, H / 2 - y1 * m, z2];
    };

    const grab = (key: string): Point[] | null => {
      const flat = f[key] as number[] | undefined;
      return flat && flat.length ? unflat(flat, data.n_points) : null;
    };

    const want = scene.layers;
    const series: Record<string, Point[] | null> = {};
    if (want.includes('ctx')) series.ctx = grab('ctx');
    if (want.includes('gt')) series.gt = grab(`gt${k}`);
    if (want.includes('pred')) series.pred = grab(`pred${k}`);
    if (want.includes('cv')) series.cv = grab(`cv${k}`);
    if (want.includes('still')) series.still = grab('ctx');
    const degenerate = (data.degenerate_horizons ?? []).map(String).includes(String(k));
    const sigma = want.includes('sigma') && !degenerate
      ? (f[`sigma${k}`] as number[] | undefined) : undefined;

    const ref = series.gt || series.pred || series.ctx || series.cv;
    if (!ref) return;
    const order = ref.map((p, i) => [project(p)[2], i] as const)
                     .sort((a, b) => a[0] - b[0]);
    const sMax = sigma ? Math.max(...sigma) : 0;
    const R = Math.max(2.2, Math.min(W, H) / 190);
    const dot = (p: [number, number, number], colour: string, r: number) => {
      g.beginPath(); g.arc(p[0], p[1], r, 0, 6.2832); g.fillStyle = colour; g.fill();
    };

    for (const [, i] of order) {
      if (sigma && series.pred) {
        const p = project(series.pred[i]);
        g.globalAlpha = 0.33;
        dot(p, COLOUR.sigma, R * 1.1 + R * 5.5 * (sigma[i] / (sMax || 1)));
        g.globalAlpha = 1;
      }
      if (series.gt && series.pred) {
        const a = project(series.gt[i]), b = project(series.pred[i]);
        g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]);
        g.strokeStyle = 'rgba(245,245,247,.26)'; g.lineWidth = 1.2; g.stroke();
      }
      if (series.ctx && !series.still) dot(project(series.ctx[i]), 'rgba(245,245,247,.30)', R * 0.75);
      if (series.still) dot(project(series.still[i]), COLOUR.still, R * 0.9);
      if (series.cv) dot(project(series.cv[i]), COLOUR.cv, R);
      if (series.gt) dot(project(series.gt[i]), COLOUR.gt, R);
      if (series.pred) dot(project(series.pred[i]), COLOUR.pred, R);
    }
  }, [data, scene, k, frame, yaw, pitch, zoom, f, hasGeometry]);

  // ── overlay canvas, over the still ────────────────────────────────────────
  const paintOverlay = () => {
    const cv = ovlRef.current;
    if (!cv) return;
    const box = cv.parentElement!.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    cv.width = Math.max(1, box.width * dpr);
    cv.height = Math.max(1, box.height * dpr);
    const o = cv.getContext('2d');
    if (!o) return;
    o.setTransform(dpr, 0, 0, dpr, 0, 0);
    o.clearRect(0, 0, box.width, box.height);

    const anchor = f?.px_ctx as number[] | undefined;
    // Invented geometry is never drawn on imagery, and an export made before the
    // projection existed simply has nothing to draw.
    if (placeholder || !f || !scene.layers.includes('overlay') || !anchor) return;

    // The still is object-fit:contain, so the drawn rect is the letterboxed one.
    const img = imgRef.current;
    const iw = img?.naturalWidth || data.camera?.W || box.width;
    const ih = img?.naturalHeight || data.camera?.H || box.height;
    const fit = Math.min(box.width / iw, box.height / ih);
    const w = iw * fit, h = ih * fit;
    const ox = (box.width - w) / 2, oy = (box.height - h) / 2;

    // Every series is amplified about the last OBSERVED position by the same factor,
    // so the comparison between them is untouched -- only its scale on screen.
    const gain = overlayGain(data, k);
    const put = (flat: number[] | undefined, colour: string, r: number) => {
      if (!flat || !flat.length) return;
      o.fillStyle = colour;
      for (let i = 0; i < data.n_points; i++) {
        const au = anchor[i * 2], av = anchor[i * 2 + 1];
        const u = au + (flat[i * 2] - au) * gain;
        const v = av + (flat[i * 2 + 1] - av) * gain;
        if (!Number.isFinite(u) || !Number.isFinite(v)) continue;
        o.beginPath(); o.arc(ox + u * w, oy + v * h, r, 0, 6.2832); o.fill();
      }
    };
    const want = scene.layers;
    if (want.includes('still')) put(anchor, COLOUR.still, 2.2);
    if (want.includes('cv')) put(f[`px_cv${k}`] as number[], COLOUR.cv, 2.6);
    if (want.includes('gt')) put(f[`px_gt${k}`] as number[], COLOUR.gt, 2.6);
    if (want.includes('pred')) put(f[`px_pred${k}`] as number[], COLOUR.pred, 3.2);
  };
  useEffect(paintOverlay);

  const legend: [string, string][] = [];
  if (scene.layers.includes('gt')) legend.push(['gt', SERIES_LABEL.gt]);
  if (scene.layers.includes('pred')) legend.push(['pred', SERIES_LABEL.pred]);
  if (scene.layers.includes('cv')) legend.push(['cv', SERIES_LABEL.cv]);
  if (scene.layers.includes('still')) legend.push(['still', SERIES_LABEL.still]);
  if (scene.layers.includes('sigma') &&
      !(data.degenerate_horizons ?? []).map(String).includes(String(k))) {
    legend.push(['sigma', STR.legendSigma]);
  }

  const cols = wantsFrame && hasGeometry ? 'grid-cols-2' : 'grid-cols-1';

  return (
    // min-h-0 on every level: a grid item's default min-height is auto, which refuses
    // to shrink below its content, so one unconstrained level anywhere down this chain
    // is enough to let the canvas push the layout past the window.
    <section className={`grid ${cols} min-h-0 min-w-0 grid-rows-[minmax(0,1fr)] gap-3 p-3`}>
      {wantsFrame && (
        <div className="aperture relative min-h-0 overflow-hidden rounded-[3px]">
          {src ? (
            <>
              <img
                ref={imgRef}
                src={src}
                alt="endoscopic frame from the exported sequence"
                className="block h-full w-full object-contain"
                onLoad={paintOverlay}
                onError={() => { if (candidate) failed.current.add(candidate); paintOverlay(); }}
              />
              <canvas ref={ovlRef} className="pointer-events-none absolute inset-0 h-full w-full" />
            </>
          ) : (
            <div className="grid h-full place-content-center p-10 text-center text-serosa">
              <div>
                <b className="mb-2 block text-lg text-ink">{STR.noImageTitle}</b>
                {hasGeometry ? STR.noImageBodyWithGeometry : STR.noImageBody}
              </div>
            </div>
          )}
        </div>
      )}

      {hasGeometry && (
        <div className="aperture relative min-h-0 min-w-0 overflow-hidden rounded-[3px]">
          <canvas
            ref={geomRef}
            className="block h-full w-full touch-none"
            onPointerDown={e => { drag.current = [e.clientX, e.clientY];
                                  e.currentTarget.setPointerCapture(e.pointerId); }}
            onPointerUp={() => { drag.current = null; }}
            onPointerMove={e => {
              if (!drag.current) return;
              onOrbit((e.clientX - drag.current[0]) * 0.008,
                      (e.clientY - drag.current[1]) * 0.008);
              drag.current = [e.clientX, e.clientY];
            }}
            onWheel={e => onZoom(e.deltaY > 0 ? 0.92 : 1.08)}
          />
          {legend.length > 0 && (
            <div className="absolute bottom-4 left-4 flex flex-wrap gap-4 rounded
                            bg-black/55 px-3 py-2 text-sm text-[#E8E2E0]
                            ring-1 ring-white/10">
              {legend.map(([key, label]) => (
                <span key={key} className="flex items-center gap-2">
                  <i className="inline-block h-3 w-3 rounded-sm"
                     style={{ background: COLOUR[key as keyof typeof COLOUR] }} />
                  {label}
                </span>
              ))}
            </div>
          )}
          <div className="absolute right-4 top-4 text-xs text-serosa">
            drag to orbit · scroll to zoom
          </div>
        </div>
      )}
    </section>
  );
}

export { measuredCamera };
