// Live inference, as progressive enhancement and never as a dependency.
//
// The demonstrator's first rule is that it opens by double-click and works. This file
// is allowed to make it better when a local server happens to be running; it is not
// allowed to make it worse when one is not. So every function here fails soft: a
// missing server, a refused connection, a timeout and a malformed reply all end with
// the page showing exactly the exported payload it would have shown anyway.
//
// WHY A PROTOCOL CHECK COMES FIRST
// Double-clicked, the page is on the `file://` origin, which is opaque. A request from
// it to http://localhost is cross-origin, and in current Chrome also trips the
// local-network access prompt -- a permission dialog in the middle of a viva. So the
// page never attempts a request from `file://` at all. Live mode exists only when the
// page was itself served over http, by serve/server.py, which makes the API
// same-origin: no CORS, no preflight, no prompt.

import type { Payload, Frame } from './core';

export interface LiveInfo {
  ok?: boolean;
  live: boolean;
  checkpoint_epoch: number | string;
  params: number;
  window: number;
  n_anchor: number;
  target_mode: string;
  steps: number;
  samples: number;
  threads: number;
  video?: string;
  n_frames_total?: number;
  split_frame?: number;
  origin_min?: number;
  origin_max?: number;
  n_points?: number;
  horizons?: number[];
  calibration?: Record<string, number>;
  // Sign-in. The server gates live inference on a doctor's roll number and password;
  // unauthenticated it reports `live: false` and nothing else about the bundle, so an
  // unsigned-in page behaves exactly like a page with no server behind it.
  auth_required?: boolean;
  signed_in?: boolean;
  doctor?: string | null;
}

export interface LiveRow {
  origin: number;
  k: number;
  alpha: number;
  seconds: number;
  steps: number;
  samples: number;
  ctx: number[];
  pred: number[];
  cv: number[];
  gt: number[] | null;
  px_ctx: number[] | null;
  px_pred: number[] | null;
  px_cv: number[] | null;
  px_gt: number[] | null;
  sigma?: number[];
  mean_error?: { ours: number; cv: number; still: number };
  img?: number;
  img_src?: string | null;
}

// Health has to answer fast or be treated as absent: this runs at boot and must never
// delay first paint. A prediction is 20 diffusion steps on a laptop CPU and is allowed
// to take minutes; the ceiling exists only so a wedged server cannot hang the button
// for ever.
const HEALTH_TIMEOUT_MS = 1500;
const PREDICT_TIMEOUT_MS = 300_000;

type FetchLike = typeof fetch;

/** Whether a live request may be attempted at all. False on file://, always. */
export function liveReachable(loc?: { protocol?: string }): boolean {
  const protocol = loc?.protocol
    ?? (typeof location === 'undefined' ? '' : location.protocol);
  return protocol === 'http:' || protocol === 'https:';
}

async function withTimeout<T>(ms: number, run: (signal: AbortSignal) => Promise<T>) {
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), ms);
  try {
    return await run(ac.signal);
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Ask whether a local inference server is there. Resolves to null for every negative
 * answer -- wrong protocol, no fetch, connection refused, timeout, bad JSON, a server
 * that is up but has no data bundle. The caller gets one thing to check.
 */
export async function probeLive(
  fetchImpl?: FetchLike,
  loc?: { protocol?: string },
): Promise<LiveInfo | null> {
  if (!liveReachable(loc)) return null;
  const f = fetchImpl ?? (typeof fetch === 'function' ? fetch : null);
  if (!f) return null;
  try {
    const info = await withTimeout(HEALTH_TIMEOUT_MS, async signal => {
      const res = await f('/api/health', { signal });
      if (!res.ok) return null;
      return (await res.json()) as LiveInfo;
    });
    if (!info || info.ok === false) return null;
    // A server that wants a sign-in is a real server and the page must say so, even
    // though it reports live: false until a doctor has signed in. This is the one
    // negative answer that is not "there is nothing here".
    if (info.auth_required === true && info.signed_in !== true) return info;
    // A server without a bundle can serve the page but cannot predict. Reporting it as
    // live would light the badge and then fail on the first click.
    return info.live === true ? info : null;
  } catch {
    return null;
  }
}

/** True when a server is present but no doctor has signed in yet. */
export function needsSignIn(info: LiveInfo | null): boolean {
  return !!info && info.auth_required === true && info.signed_in !== true;
}

/**
 * Sign a doctor in. Resolves to the server's message on failure and null on success,
 * so the caller has one thing to check and always has something to show.
 *
 * The session is a HttpOnly cookie set by the server; nothing here stores or can read
 * a token, and the password is never kept after this call.
 */
export async function signIn(
  roll: string,
  password: string,
  fetchImpl?: FetchLike,
  loc?: { protocol?: string },
): Promise<string | null> {
  if (!liveReachable(loc)) return 'live mode needs the page served by the local server';
  const f = fetchImpl ?? (typeof fetch === 'function' ? fetch : null);
  if (!f) return 'this browser cannot reach the local server';
  try {
    return await withTimeout(HEALTH_TIMEOUT_MS * 8, async signal => {
      const res = await f('/api/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ roll, password }),
        signal,
      });
      let payload: { error?: string } = {};
      try { payload = await res.json(); } catch { /* keep the status-based message */ }
      if (res.ok) return null;
      return payload.error || 'sign-in failed';
    });
  } catch {
    return 'the local server did not answer the sign-in';
  }
}

/** Sign out. Never throws: the worst case is the session outliving the click. */
export async function signOut(fetchImpl?: FetchLike,
                              loc?: { protocol?: string }): Promise<void> {
  if (!liveReachable(loc)) return;
  const f = fetchImpl ?? (typeof fetch === 'function' ? fetch : null);
  if (!f) return;
  try {
    await withTimeout(HEALTH_TIMEOUT_MS * 4,
                      signal => f('/api/logout', { method: 'POST', signal }));
  } catch { /* the session expires on its own, and on restart */ }
}

/**
 * Whether this server is serving the same thing this page loaded. Returns a reason to
 * refuse, or null to proceed.
 *
 * This is not defensive padding. The server reads its own bundle and the page reads
 * its own payload, and nothing links the two files but the hand that copied them. Two
 * ways that goes wrong:
 *
 *  - A bundle exported from a different video. Every live array would be geometry from
 *    another sequence, drawn over this one's stills, beside this one's error table. It
 *    would look entirely plausible and be meaningless.
 *  - A different point count. The arrays are flat triples unpacked against
 *    `payload.n_points`, so a mismatch is read off the end: wrong numbers, or a canvas
 *    full of NaN.
 *
 * Both are silent. Refusing live mode with a stated reason is the only safe response.
 */
export function liveMismatch(info: LiveInfo, data: Payload): string | null {
  if (info.video != null && data.video != null &&
      String(info.video) !== String(data.video)) {
    return `the server is serving video ${info.video} but this page loaded video ` +
           `${data.video}, so live predictions would be another sequence's geometry ` +
           `drawn over this one`;
  }
  if (info.n_points != null && data.n_points != null &&
      Number(info.n_points) !== Number(data.n_points)) {
    return `the server returns ${info.n_points} points and this page expects ` +
           `${data.n_points}, which cannot be drawn together`;
  }
  return null;
}

/** One live prediction. Throws on failure; the caller keeps the exported row. */
export async function predictLive(
  origin: number,
  k: number,
  opts: { hypotheses?: number; fetchImpl?: FetchLike } = {},
): Promise<LiveRow> {
  const f = opts.fetchImpl ?? (typeof fetch === 'function' ? fetch : null);
  if (!f) throw new Error('no fetch available');
  const row = await withTimeout(PREDICT_TIMEOUT_MS, async signal => {
    const res = await f('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ origin, k, hypotheses: opts.hypotheses ?? 0 }),
      signal,
    });
    const body = await res.json().catch(() => null);
    if (!res.ok || !body || body.error) {
      throw new Error(body?.error || `server returned ${res.status}`);
    }
    return body as LiveRow;
  });
  if (!Array.isArray(row.pred) || !row.pred.length) {
    throw new Error('server returned no prediction');
  }
  return row;
}

/**
 * The payload the page should render while a live row is on screen.
 *
 * Only this frame's arrays are replaced, and only at this horizon. `mean_error`,
 * `motion` and `span` are left exactly as exported, because those are averages over
 * the whole held-out sequence and the readout's comparative sentences are about them.
 * Substituting one origin's errors there would turn a sequence-level claim into a
 * single-frame one without changing a word of the sentence making it -- so the live
 * origin's own errors are reported separately, labelled as one origin.
 */
export function mergeLiveRow(data: Payload, frameIndex: number, k: number,
                             row: LiveRow): Payload {
  const frames = data.frames.slice();
  const prev = frames[frameIndex] ?? ({ t: row.origin, ctx: [] } as unknown as Frame);
  const next: Frame = { ...prev, t: row.origin, ctx: row.ctx };
  const img = row.img ?? row.origin - 1;
  next.img = img;
  if (row.px_ctx) next.px_ctx = row.px_ctx;
  next[`pred${k}`] = row.pred;
  next[`cv${k}`] = row.cv;
  if (row.gt) next[`gt${k}`] = row.gt;
  if (row.px_pred) next[`px_pred${k}`] = row.px_pred;
  if (row.px_cv) next[`px_cv${k}`] = row.px_cv;
  if (row.px_gt) next[`px_gt${k}`] = row.px_gt;
  if (row.sigma) next[`sigma${k}`] = row.sigma;
  frames[frameIndex] = next;

  const frame_files = { ...(data.frame_files ?? {}) };
  if (row.img_src) frame_files[String(img)] = row.img_src;

  // A live alpha of zero means the same thing it means in the export: the calibration
  // found the model's contribution worthless at this horizon, so the prediction IS
  // constant velocity and its spread is identically zero. The page already knows how
  // to say that; it just has to be told.
  const degenerate = new Set((data.degenerate_horizons ?? []).map(String));
  if (row.alpha === 0) degenerate.add(String(k));

  return { ...data, frames, frame_files,
           degenerate_horizons: Array.from(degenerate) };
}

/** Error over this one origin, as a percentage of how far the tissue actually went. */
export function liveSkill(row: LiveRow): Record<string, number | null> | null {
  const e = row.mean_error;
  if (!e) return null;
  const travelled = e.still;          // distance from last observed frame to the truth
  const pct = (x: number) => (travelled > 0 ? (x / travelled) * 100 : null);
  return { ours: pct(e.ours), cv: pct(e.cv), still: travelled > 0 ? 100 : null };
}
