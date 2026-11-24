import { test, beforeEach } from 'vitest';
import a from 'node:assert/strict';
import { liveMismatch, liveReachable, mergeLiveRow, needsSignIn, predictLive,
         probeLive, signIn, signOut, liveSkill,
         type LiveRow, type LiveInfo } from './live';
import { fetchCalls, reply, stubFetch } from './test-setup';
import { synth } from './synth';
import type { Payload } from './core';

beforeEach(() => stubFetch());

const INFO: LiveInfo = {
  live: true, checkpoint_epoch: 100, params: 16318467, window: 32, n_anchor: 512,
  target_mode: 'cv_residual', steps: 20, samples: 4, threads: 12, video: '3',
  origin_min: 240, origin_max: 296, n_points: 256, horizons: [1, 3], split_frame: 240,
};

function row(over: Partial<LiveRow> = {}): LiveRow {
  const n = 4;
  const three = Array.from({ length: n * 3 }, (_, i) => i * 0.001);
  const two = Array.from({ length: n * 2 }, (_, i) => 0.3 + i * 0.01);
  return {
    origin: 250, k: 3, alpha: 1.021, seconds: 9.4, steps: 20, samples: 4,
    ctx: three, pred: three, cv: three, gt: three,
    px_ctx: two, px_pred: two, px_cv: two, px_gt: two,
    mean_error: { ours: 0.003, cv: 0.004, still: 0.005 },
    ...over,
  };
}

// ── the guarantee that matters ───────────────────────────────────────────────
// Double-clicked, the page is on file:// and must not make a request at all: a
// cross-origin call from an opaque origin to localhost means a CORS failure at best
// and Chrome's local-network permission prompt at worst, mid-viva.

test('no request is attempted from the file:// origin', async () => {
  const info = await probeLive(undefined, { protocol: 'file:' });
  a.equal(info, null);
  a.deepEqual(fetchCalls, [], 'the page called fetch while on file://');
});

test('liveReachable is false on file:// and true over http', () => {
  a.equal(liveReachable({ protocol: 'file:' }), false);
  a.equal(liveReachable({ protocol: 'http:' }), true);
  a.equal(liveReachable({ protocol: 'https:' }), true);
});

test('probeLive survives every way a server can be absent', async () => {
  for (const handler of [
    undefined,                                   // connection refused
    () => ({ ok: false }),                       // something else on the port
    () => ({ live: false, params: 0 }),          // server up, no data bundle
    () => null,                                  // empty reply
  ]) {
    stubFetch(handler as never);
    a.equal(await probeLive(undefined, { protocol: 'http:' }), null);
  }
});

test('probeLive reports a server that is up and has a bundle', async () => {
  stubFetch(() => INFO);
  const info = await probeLive(undefined, { protocol: 'http:' });
  a.equal(info?.params, 16318467);
  a.equal(info?.steps, 20);
  a.deepEqual(fetchCalls, ['/api/health']);
});

test('predictLive posts the origin and horizon, and returns the row', async () => {
  let seen: unknown = null;
  stubFetch((_u, init) => { seen = JSON.parse(String(init?.body)); return row(); });
  const r = await predictLive(250, 3);
  a.deepEqual(seen, { origin: 250, k: 3, hypotheses: 0 });
  a.deepEqual(fetchCalls, ['/api/predict']);
  a.equal(r.origin, 250);
});

test('predictLive rejects a refusal rather than returning a blank row', async () => {
  stubFetch(() => ({ error: 'origin 10 is outside the held-out range 240..296' }));
  await a.rejects(() => predictLive(10, 1), /held-out range/);
});

test('predictLive rejects a reply with no prediction in it', async () => {
  stubFetch(() => ({ origin: 250, k: 3, pred: [] }));
  await a.rejects(() => predictLive(250, 3), /no prediction/);
});

// ── merging ──────────────────────────────────────────────────────────────────

function payload(): Payload {
  const d = synth() as Payload;
  d.synthetic = false;
  d.mean_error = { '1': { ours: 1, cv: 2, still: 3 }, '3': { ours: 4, cv: 5, still: 6 } };
  d.motion = { '1': 10, '3': 20 };
  d.span = 1.5;
  return d;
}

test('merging a live row leaves the sequence-level measurement alone', () => {
  const d = payload();
  const m = mergeLiveRow(d, 0, 3, row());
  // These three are averages over the whole held-out sequence and the readout's
  // sentences are about them. One origin must not be able to restate them.
  a.deepEqual(m.mean_error, d.mean_error);
  a.deepEqual(m.motion, d.motion);
  a.equal(m.span, d.span);
});

test('merging replaces only the named frame and only the named horizon', () => {
  const d = payload();
  const m = mergeLiveRow(d, 0, 3, row());
  a.deepEqual(m.frames[0].pred3, row().pred);
  a.deepEqual(m.frames[0].pred1, d.frames[0].pred1, 'the other horizon was overwritten');
  a.deepEqual(m.frames[1], d.frames[1], 'a frame other than the live one changed');
  a.equal(m.frames[0].t, 250);
  a.equal(m.frames.length, d.frames.length);
});

test('merging does not mutate the loaded payload', () => {
  const d = payload();
  const before = JSON.stringify(d.frames[0]);
  mergeLiveRow(d, 0, 3, row());
  a.equal(JSON.stringify(d.frames[0]), before);
});

test('a live alpha of zero is recorded as a degenerate horizon', () => {
  const m = mergeLiveRow(payload(), 0, 3, row({ alpha: 0 }));
  a.ok((m.degenerate_horizons ?? []).includes('3'),
       'alpha=0 means the prediction IS constant velocity and the spread is zero');
});

test('a live row without a still does not invent one', () => {
  const m = mergeLiveRow(payload(), 0, 3, row({ img_src: null, img: 249 }));
  a.equal((m.frame_files ?? {})['249'], undefined);
});

test('a live row with a still registers it for the overlay', () => {
  const m = mergeLiveRow(payload(), 0, 3,
                         row({ img: 249, img_src: 'frames/00249.jpg' }));
  a.equal((m.frame_files ?? {})['249'], 'frames/00249.jpg');
});

test('liveSkill scores against how far the tissue actually travelled', () => {
  const s = liveSkill(row({ mean_error: { ours: 0.0025, cv: 0.004, still: 0.005 } }))!;
  a.equal(s.ours, 50);
  a.equal(s.cv, 80);
  a.equal(s.still, 100);
});

test('liveSkill returns nothing when there is no truth to score against', () => {
  a.equal(liveSkill(row({ mean_error: undefined })), null);
  const s = liveSkill(row({ mean_error: { ours: 0.1, cv: 0.1, still: 0 } }))!;
  a.equal(s.ours, null, 'nothing moved, so a percentage of motion is undefined');
});

// ── refusing a server that does not match the page ───────────────────────────

test('a server serving another video is refused, with the reason', () => {
  const d = synth() as Payload; d.video = '3'; d.n_points = 256;
  const why = liveMismatch({ ...INFO, video: '7', n_points: 256 }, d);
  a.match(String(why), /video 7 but this page loaded video 3/);
});

test('a server with a different point count is refused', () => {
  const d = synth() as Payload; d.video = '3'; d.n_points = 256;
  const why = liveMismatch({ ...INFO, video: '3', n_points: 512 }, d);
  a.match(String(why), /512 points and this page expects 256/);
});

test('a matching server is accepted', () => {
  const d = synth() as Payload; d.video = '3'; d.n_points = 256;
  a.equal(liveMismatch({ ...INFO, video: '3', n_points: 256 }, d), null);
});

test('a server that reports neither is not refused on a guess', () => {
  const d = synth() as Payload; d.video = '3'; d.n_points = 256;
  a.equal(liveMismatch({ ...INFO, video: undefined, n_points: undefined }, d), null);
});

// ── doctor sign-in ───────────────────────────────────────────────────────────
// The server gates live inference. These check the client half: that the file://
// guarantee still holds for the two new endpoints, that a refusal is surfaced rather
// than swallowed, and that no password check happens here.

test('signing in is not attempted from the file:// origin', async () => {
  const why = await signIn('doc-1', 'secret', undefined, { protocol: 'file:' });
  a.equal(typeof why, 'string', 'file:// should refuse locally, not silently succeed');
  a.deepEqual(fetchCalls, [], 'the page called fetch while on file://');
});

test('signing out is not attempted from the file:// origin', async () => {
  await signOut(undefined, { protocol: 'file:' });
  a.deepEqual(fetchCalls, [], 'the page called fetch while on file://');
});

test('a server asking for a sign-in is reported, not treated as absent', async () => {
  // It answers live:false until someone signs in. Returning null here, as for a
  // server with no bundle, would leave the page with no way to offer the sign-in.
  stubFetch(() => ({ ok: true, live: false, auth_required: true, signed_in: false }));
  const info = await probeLive(undefined, { protocol: 'http:' });
  a.ok(info, 'a server that wants a sign-in was reported as no server at all');
  a.equal(needsSignIn(info), true);
});

test('needsSignIn is false once signed in, and for no server', async () => {
  a.equal(needsSignIn(null), false);
  a.equal(needsSignIn({ ...INFO, auth_required: true, signed_in: true }), false);
  a.equal(needsSignIn({ ...INFO, auth_required: false }), false);
});

test('a correct sign-in resolves to null and posts to /api/login', async () => {
  let seen: { url: string; body: unknown } | null = null;
  stubFetch((url, init) => {
    seen = { url, body: JSON.parse(String(init?.body ?? '{}')) };
    return { ok: true, roll: 'doc-1' };
  });
  const why = await signIn('doc-1', 'secret', undefined, { protocol: 'http:' });
  a.equal(why, null);
  a.equal(seen!.url, '/api/login');
  a.deepEqual(seen!.body, { roll: 'doc-1', password: 'secret' });
});

test("a refused sign-in surfaces the server's own reason", async () => {
  stubFetch(() => reply(401, { error: 'that roll number and password do not match' }));
  const why = await signIn('doc-1', 'wrong', undefined, { protocol: 'http:' });
  a.equal(why, 'that roll number and password do not match');
});

test('a lockout message reaches the doctor unchanged', async () => {
  stubFetch(() => reply(401, { error: 'too many failed attempts; try again in 47 s' }));
  const why = await signIn('doc-1', 'wrong', undefined, { protocol: 'http:' });
  a.match(String(why), /try again in 47 s/);
});

test('a dead server during sign-in gives a message, not an exception', async () => {
  stubFetch();                       // every call throws
  const why = await signIn('doc-1', 'secret', undefined, { protocol: 'http:' });
  a.equal(typeof why, 'string');
  a.match(String(why), /did not answer/);
});

test('signing out survives a server that has already gone', async () => {
  stubFetch();
  await signOut(undefined, { protocol: 'http:' });   // must not throw
  a.deepEqual(fetchCalls, ['/api/logout']);
});
