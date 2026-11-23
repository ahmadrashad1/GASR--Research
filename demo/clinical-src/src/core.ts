// Pure logic for the clinical demonstrator. No DOM, no React, no data.
//
// Everything comparative the app says out loud is computed here, from the mean_error
// block of whatever demo_data.js was loaded. Nothing in this file, and nothing in the
// UI that uses it, states who wins. The repo's only other MPE-style comparison
// (paper/measured.json -> comparison) is from the superseded epoch-60 W=4 checkpoint
// and was read off a chart by eye; it must not shape what this app claims.

export interface MethodErrors { ours?: number; cv?: number; still?: number }
export type MeanError = Record<string, MethodErrors> | null;

export interface Timing {
  timing_ok?: boolean;
  fps?: number;
  gap_min?: number;
  gap_max?: number;
}

export interface Camera {
  fx: number; fy: number; cx: number; cy: number;
  W: number; H: number;
  calibrated?: boolean;
  source?: string;
}

export interface Frame {
  t: number;
  img?: number;
  ctx: number[];
  px_ctx?: number[];
  [key: string]: number | number[] | undefined;
}

export interface Payload {
  video: string;
  workspace?: string;
  n_points: number;
  n_frames: number;
  horizons: number[];
  window: number;
  centre: [number, number, number];
  radius: number;
  span?: number;
  synthetic?: boolean;
  target_mode?: string;
  checkpoint_epoch?: number | string;
  mean_error: MeanError;
  motion?: Record<string, number> | null;
  calibrated?: boolean;
  alpha?: Record<string, number> | null;
  degenerate_horizons?: string[];
  frame_files?: Record<string, string>;
  timing?: Timing | null;
  camera?: Camera;
  frames: Frame[];
}

export const METHOD_LABELS = {
  ours: 'GASR (ours)',
  cv: 'Straight-line motion',
  still: 'No movement',
} as const;

export type MethodKey = keyof typeof METHOD_LABELS;

// A finite number, or null. Everything below routes through this rather than trusting
// the payload: a missing horizon, a null from a partial export, or a NaN out of a
// degenerate calibration must all come back as "unknown" rather than as a number the
// UI would render with total confidence.
function num(v: unknown): number | null {
  return (typeof v === 'number' && Number.isFinite(v)) ? v : null;
}

export type Point = [number, number, number];

export function unflat(flat: number[], n: number): Point[] {
  const out: Point[] = [];
  for (let i = 0; i < n; i++) out.push([flat[i * 3], flat[i * 3 + 1], flat[i * 3 + 2]]);
  return out;
}

function meanDistance(a: Point[] | null, b: Point[] | null): number | null {
  if (!a || !b || !a.length || a.length !== b.length) return null;
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const dx = a[i][0] - b[i][0], dy = a[i][1] - b[i][1], dz = a[i][2] - b[i][2];
    sum += Math.sqrt(dx * dx + dy * dy + dz * dz);
  }
  return sum / a.length;
}

export const meanPointError = meanDistance;

// How far the tissue actually went over the horizon: the distance from the last
// observed frame to what really happened. This is the denominator the app leads with.
export const motionTravelled = meanDistance;

// Error as a percentage of the distance the tissue actually travelled. A skill score:
// 100 % means the prediction was no better than not predicting, under 100 % beats
// standing still, over 100 % is worse than standing still.
//
// Returns null when nothing moved. A static window makes the denominator zero, and
// Infinity rendered into a sentence reads as a catastrophic error, not an undefined one.
export function skillPercent(err: unknown, motion: unknown): number | null {
  const e = num(err), m = num(motion);
  if (e === null || m === null || m <= 0) return null;
  return e / m * 100;
}

// Error as a percentage of the reconstructed tissue's own span. Secondary context only:
// every method in this project, including assuming the tissue never moves, scores under
// half a percent against this denominator, so it distinguishes nothing and flatters
// everything. Shown once, labelled, never led with.
export function spanPercent(err: unknown, span: unknown): number | null {
  const e = num(err), s = num(span);
  if (e === null || s === null || s <= 0) return null;
  return e / s * 100;
}

export interface Ranked { key: MethodKey; err: number }

export function rankAtHorizon(meanError: MeanError, k: number): Ranked[] | null {
  if (!meanError) return null;
  const row = meanError[String(k)];
  if (!row) return null;
  const out: Ranked[] = [];
  for (const key of Object.keys(METHOD_LABELS) as MethodKey[]) {
    const e = num(row[key]);
    if (e !== null) out.push({ key, err: e });
  }
  if (!out.length) return null;
  return out.sort((x, y) => x.err - y.err);
}

// Two errors this close are the same number, not a ranking. The case is not
// hypothetical: where the calibration fit returns alpha = 0 the exporter states that
// pred{k} is EXACTLY cv{k}, and its own run has printed
// "k=1  ours 0.47478   const-vel 0.47478". A stable sort over METHOD_LABELS puts ours
// first on equal keys, so without this the page announces the model as the closest
// prediction at a horizon where its prediction IS the straight-line baseline.
//
// The threshold is numerical indistinguishability only. A wider one would be a
// significance threshold, and nothing in this pipeline would justify a value for it.
const TIE_REL = 1e-6;

export interface Verdict {
  winner: MethodKey;
  winnerLabel: string;
  text: string;
  tied: boolean;
  tiedWith: MethodKey[];
  oursBeatsAll: boolean;
  oursBeatsStatic: boolean | null;
}

export function verdict(meanError: MeanError, k: number): Verdict | null {
  const rank = rankAtHorizon(meanError, k);
  if (!rank) return null;

  const winner = rank[0];
  const label = METHOD_LABELS[winner.key];
  const ours = rank.find(r => r.key === 'ours');
  const still = rank.find(r => r.key === 'still');

  const scale = Math.max(...rank.map(r => Math.abs(r.err))) || 1;
  const level = rank.filter(r => Math.abs(r.err - winner.err) <= TIE_REL * scale);
  const tied = level.length > 1;

  const oursBeatsAll = winner.key === 'ours' && !tied;
  const oursBeatsStatic = (ours && still) ? ours.err < still.err : null;

  let text: string;
  if (tied) {
    const names = level.map(r => METHOD_LABELS[r.key]);
    text = `${names.slice(0, -1).join(', ')} and ${names[names.length - 1]} are ` +
           `numerically indistinguishable ${horizonLabel(k, null)}: to the precision ` +
           `reported here they are the same prediction.`;
  } else if (oursBeatsAll) {
    text = `${label} is the closest prediction ${horizonLabel(k, null)}.`;
  } else {
    // Say it plainly. Standing in front of an examination panel claiming a win the
    // number on the same screen contradicts is a worse outcome than reading out a loss.
    text = `${label} is closest ${horizonLabel(k, null)}; ${METHOD_LABELS.ours} is ` +
           `behind it here. The model is the only method on screen that also reports ` +
           `how uncertain it is at each point.`;
  }

  return { winner: winner.key, winnerLabel: label, text, tied,
           tiedWith: level.map(r => r.key), oursBeatsAll, oursBeatsStatic };
}

// The horizon in frames, which is exact, plus elapsed time as a RANGE when and only
// when the export recovered usable timing.
//
// Kept frames are not evenly spaced: Module 1 takes every second raw frame and then
// drops blurred ones, so consecutive frames can be 2 or 10 raw frames apart. A single
// averaged millisecond figure over origins whose gaps differ fivefold is the same class
// of fabrication as stating the error in millimetres.
export function horizonLabel(k: number, timing: Timing | null | undefined): string {
  const frames = `${k} frame${k === 1 ? '' : 's'} ahead`;
  if (!timing || timing.timing_ok !== true) return frames;
  const fps = num(timing.fps), lo = num(timing.gap_min), hi = num(timing.gap_max);
  if (fps === null || lo === null || hi === null || fps <= 0 || lo <= 0 || hi <= 0) {
    return frames;
  }
  const a = (k * lo / fps).toFixed(1), b = (k * hi / fps).toFixed(1);
  return `${frames} — between about ${a} and ${b} s at this sequence's recovered frame rate`;
}

// How much to exaggerate motion so it is visible at projector distance.
//
// Over three frames the tissue moves about two pixels in a 640x360 frame, and the
// methods differ from each other by about one. Drawn at true scale, prediction, truth
// and baseline all land inside a single dot and the overlay demonstrates nothing.
// Module 6 reached the same conclusion and records arrow_gain 5 and 100 on its own
// figures. Exaggeration is honest only when declared, so this returns a round number
// the page prints, and it never shrinks motion already large enough to see.
export function displayGain(medianPx: unknown, targetPx?: number): number {
  const m = num(medianPx), t = num(targetPx) || 45;
  if (m === null || m <= 0) return 1;
  const raw = t / m;
  if (raw <= 1) return 1;
  const steps = [1, 2, 5, 10, 20, 50, 100, 200, 500];
  let best = steps[0];
  for (const g of steps) if (g <= raw) best = g;
  return best;
}

// Whether these intrinsics were measured or assumed. The flag alone is not enough:
// Module 1 always writes camera_meta.json with fx derived from a nominal 75-degree
// laparoscope FOV, and an export made before that was understood carries
// calibrated:true over exactly those numbers. Re-deriving the signature means an
// existing payload stops overclaiming without being regenerated.
export function measuredCamera(cam: Camera | undefined | null): boolean {
  if (!cam || cam.calibrated !== true) return false;
  if (cam.source && cam.source !== 'measured') return false;
  const W = Number(cam.W), fx = Number(cam.fx);
  if (!Number.isFinite(W) || !Number.isFinite(fx) || W <= 0) return false;
  const nominal = (W / 2) / Math.tan(75 * Math.PI / 360);
  return Math.abs(fx - nominal) > 1e-4 * Math.max(fx, 1);
}
