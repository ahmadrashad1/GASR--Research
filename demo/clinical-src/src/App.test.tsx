import { test, afterEach } from 'vitest';
import a from 'node:assert/strict';
import { cleanup, render, screen, fireEvent } from '@testing-library/react';
import { App } from './App';
import { synth } from './synth';
import { SCENES } from './scenes';
import type { Payload } from './core';

afterEach(cleanup);

// A payload shaped like the real export: measured, with a motion denominator, the
// calibration metadata and the projected image coordinates.
function real(over: Partial<Payload> = {}): Payload {
  const d = synth() as Payload;
  d.synthetic = false;
  d.video = 'workspace_video4';
  d.checkpoint_epoch = 100;
  d.mean_error = { '1': { ours: 0.00611, cv: 0.00590, still: 0.00944 },
                   '3': { ours: 0.02000, cv: 0.02272, still: 0.02255 } };
  d.motion = { '1': 0.01, '3': 0.024 };
  d.camera = { fx: 503.2, fy: 499.7, cx: 322.4, cy: 178.9, W: 640, H: 360,
               calibrated: true, source: 'measured' };
  d.frame_files = {};
  for (const f of d.frames) {
    const uv: number[] = [];
    for (let i = 0; i < d.n_points; i++) uv.push(0.2 + 0.6 * i / d.n_points, 0.3);
    f.px_ctx = uv;
    for (const k of [1, 3]) {
      f[`px_gt${k}`] = uv.slice(); f[`px_pred${k}`] = uv.slice(); f[`px_cv${k}`] = uv.slice();
    }
    if (f.img != null) d.frame_files[String(f.img)] = `frames/${f.img}.jpg`;
  }
  return Object.assign(d, over);
}

test('every scene renders in placeholder mode without throwing', () => {
  render(<App data={synth()} placeholder />);
  a.ok(screen.getByText('PLACEHOLDER DATA'));
  for (const s of SCENES) {
    a.ok(screen.getByText(s.title), `${s.id} did not render`);
    fireEvent.keyDown(window, { key: 'ArrowRight' });
  }
});

test('every scene renders against a real payload at both horizons', () => {
  render(<App data={real()} placeholder={false} />);
  a.ok(screen.getByText('MEASURED'));
  for (let i = 0; i < SCENES.length; i++) {
    for (const key of ['1', '3', 'r', ' ']) fireEvent.keyDown(window, { key });
    fireEvent.keyDown(window, { key: 'ArrowRight' });
  }
});

test('placeholder mode states no verdict on the scenes that compare', () => {
  render(<App data={synth()} placeholder />);
  for (let i = 0; i < 4; i++) fireEvent.keyDown(window, { key: 'ArrowRight' });
  a.match(document.body.textContent!, /carries no measurement/);
  a.doesNotMatch(document.body.textContent!, /closest prediction/i);
});

test('walking the whole deck in placeholder mode never states a verdict', () => {
  render(<App data={synth()} placeholder />);
  for (let i = 0; i < SCENES.length + 1; i++) {
    a.doesNotMatch(document.body.textContent!, /closest prediction|indistinguishable/i);
    fireEvent.keyDown(window, { key: 'ArrowRight' });
  }
});

test('a baseline that wins is named, not hidden', () => {
  // cv is lowest at k=1 in this payload, as it is on the full held-out set.
  render(<App data={real()} placeholder={false} />);
  for (let i = 0; i < 4; i++) fireEvent.keyDown(window, { key: 'ArrowRight' });
  fireEvent.keyDown(window, { key: '1' });
  a.match(document.body.textContent!, /Straight-line motion is closest/);
});

test('an exact tie is not announced as a win for ours', () => {
  const d = real({ mean_error: { '1': { ours: 0.4, cv: 0.4, still: 0.5 },
                                 '3': { ours: 0.4, cv: 0.4, still: 0.5 } } });
  render(<App data={d} placeholder={false} />);
  for (let i = 0; i < 4; i++) fireEvent.keyDown(window, { key: 'ArrowRight' });
  a.match(document.body.textContent!, /indistinguishable/i);
  a.doesNotMatch(document.body.textContent!, /closest prediction/i);
});

test('an assumed field of view is declared on the overlay scenes', () => {
  const W = 640;
  const d = real();
  d.camera = { fx: (W / 2) / Math.tan(75 * Math.PI / 360), fy: 417, cx: 320, cy: 180,
               W, H: 360, calibrated: true };
  render(<App data={d} placeholder={false} />);
  for (let i = 0; i < 3; i++) fireEvent.keyDown(window, { key: 'ArrowRight' });
  a.match(document.body.textContent!, /nominal 75° field of view/);
});

test('a degenerate horizon is stated and its spread is not legended', () => {
  const d = real({ degenerate_horizons: ['1'] });
  render(<App data={d} placeholder={false} />);
  for (let i = 0; i < 5; i++) fireEvent.keyDown(window, { key: 'ArrowRight' });
  fireEvent.keyDown(window, { key: '1' });
  a.match(document.body.textContent!, /gave the model no weight/);
  a.doesNotMatch(document.body.textContent!, /Spread across sampled futures/);
});

test('a missing horizon degrades to a grammatical sentence', () => {
  const d = real();
  for (const f of d.frames) { delete f.gt3; delete f.sigma3; }
  render(<App data={d} placeholder={false} />);
  for (let i = 0; i < 3; i++) fireEvent.keyDown(window, { key: 'ArrowRight' });
  a.match(document.body.textContent!, /has no recorded outcome/);
  a.doesNotMatch(document.body.textContent!, /no the /i);
});

test('scene prose follows the horizon actually selected', () => {
  render(<App data={real()} placeholder={false} />);
  for (let i = 0; i < 3; i++) fireEvent.keyDown(window, { key: 'ArrowRight' });
  a.match(document.body.textContent!, /will be 3 frames ahead/);
  fireEvent.keyDown(window, { key: '1' });
  a.match(document.body.textContent!, /will be 1 frame ahead/);
});

test('a payload with no horizons list still renders instead of going blank', () => {
  const d = real();
  delete (d as Partial<Payload>).horizons;
  render(<App data={d} placeholder={false} />);
  a.ok(screen.getByText(SCENES[0].title));
});

// ── live inference ───────────────────────────────────────────────────────────
// Live mode is progressive enhancement. These tests pin the half that matters: that
// its absence, and its failure, cost the page nothing.
import { beforeEach } from 'vitest';
import { waitFor } from '@testing-library/react';
import { reply, stubFetch } from './test-setup';

const LIVE_INFO = {
  live: true, checkpoint_epoch: 100, params: 16318467, window: 32, n_anchor: 512,
  target_mode: 'cv_residual', steps: 20, samples: 4, threads: 12, video: '3',
  origin_min: 240, origin_max: 296, n_points: 256, horizons: [1, 3], split_frame: 240,
};

// The health reply has to describe the payload the page actually loaded, or the
// mismatch guard refuses live mode -- which is the point of that guard.
function infoFor(d: Payload) {
  return { ...LIVE_INFO, video: d.video, n_points: d.n_points };
}

function liveRowFor(d: Payload, k: number, over: Record<string, unknown> = {}) {
  const f = d.frames[0];
  return {
    origin: f.t, k, alpha: 1.02, seconds: 9.4, steps: 20, samples: 4,
    ctx: f.ctx, pred: f[`pred${k}`], cv: f[`cv${k}`], gt: f[`gt${k}`],
    px_ctx: f.px_ctx, px_pred: f.px_ctx, px_cv: f.px_ctx, px_gt: f.px_ctx,
    mean_error: { ours: 0.003, cv: 0.004, still: 0.005 },
    ...over,
  };
}

beforeEach(() => stubFetch());

test('with no server the page is exactly the exported demonstrator', async () => {
  render(<App data={real()} placeholder={false} />);
  await waitFor(() => a.ok(screen.getByText('MEASURED')));
  a.doesNotMatch(document.body.textContent!, /local inference server/i);
  a.doesNotMatch(document.body.textContent!, /run the model now/i);
});

test('a server that is up but has no data bundle does not light the badge', async () => {
  stubFetch(() => ({ ...LIVE_INFO, live: false }));
  render(<App data={real()} placeholder={false} />);
  await waitFor(() => a.ok(screen.getByText('MEASURED')));
  a.doesNotMatch(document.body.textContent!, /local inference server/i);
});

test('a reachable server offers live inference without changing what is shown', async () => {
  const d = real();
  stubFetch(() => infoFor(d));
  render(<App data={d} placeholder={false} />);
  await waitFor(() => a.match(document.body.textContent!, /local inference server/i));
  // Nothing has been computed yet, so the badge must still say where the numbers came from.
  a.ok(screen.getByText('MEASURED'));
  a.match(document.body.textContent!, /Press L and the model runs here/);
});

test('running the model live marks the reading as live and says it was computed here',
     async () => {
  const d = real();
  stubFetch(url => url === '/api/health' ? infoFor(d) : liveRowFor(d, 3));
  render(<App data={d} placeholder={false} />);
  await waitFor(() => a.ok(screen.getByText('run the model now')));
  fireEvent.click(screen.getByText('run the model now'));
  await waitFor(() => a.ok(screen.getByText('LIVE')));
  a.match(document.body.textContent!, /Computed on this machine/);
  a.match(document.body.textContent!, /origin \d+ alone/);
});

test('a live prediction never restates the sequence-level table', async () => {
  const d = real();
  stubFetch(url => url === '/api/health' ? infoFor(d) : liveRowFor(d, 3));
  render(<App data={d} placeholder={false} />);
  await waitFor(() => a.ok(screen.getByText('run the model now')));
  for (let i = 0; i < 4; i++) fireEvent.keyDown(window, { key: 'ArrowRight' });
  const before = screen.getByText(/share of how far the tissue actually travelled/).textContent;
  fireEvent.keyDown(window, { key: 'L' });
  await waitFor(() => a.match(document.body.textContent!, /Computed on this machine/));
  a.match(document.body.textContent!, /averages the whole held-out sequence/);
  a.equal(screen.getByText(/share of how far the tissue actually travelled/).textContent,
          before, 'the sequence-level scope note changed when one origin was computed');
});

test('a server that dies mid-demo falls back to the exported data', async () => {
  const d = real();
  stubFetch(url => url === '/api/health' ? infoFor(d) : undefined);
  render(<App data={d} placeholder={false} />);
  await waitFor(() => a.ok(screen.getByText('run the model now')));
  fireEvent.click(screen.getByText('run the model now'));
  await waitFor(() => a.match(document.body.textContent!, /did not answer/));
  a.ok(screen.getByText('MEASURED'), 'the badge must fall back, not stay LIVE');
  a.match(document.body.textContent!, /still holds/);
  // The exported geometry is still what is drawn, so the scene is still usable.
  a.ok(screen.getByText(SCENES[0].title));
});

test('a refused origin is reported as the server explains it', async () => {
  const d = real();
  stubFetch(url => url === '/api/health' ? infoFor(d)
    : ({ error: 'origin 10 is outside the held-out range 240..296' }));
  render(<App data={d} placeholder={false} />);
  await waitFor(() => a.ok(screen.getByText('run the model now')));
  fireEvent.click(screen.getByText('run the model now'));
  await waitFor(() => a.match(document.body.textContent!, /held-out range/));
});

test('changing the horizon drops a live row rather than relabelling it', async () => {
  const d = real();
  stubFetch(url => url === '/api/health' ? infoFor(d) : liveRowFor(d, 3));
  render(<App data={d} placeholder={false} />);
  await waitFor(() => a.ok(screen.getByText('run the model now')));
  fireEvent.keyDown(window, { key: '3' });
  fireEvent.keyDown(window, { key: 'L' });
  await waitFor(() => a.ok(screen.getByText('LIVE')));
  fireEvent.keyDown(window, { key: '1' });
  await waitFor(() => a.ok(screen.getByText('MEASURED')));
});

test('a server serving a different video is refused on screen, not used', async () => {
  const d = real();
  stubFetch(() => ({ ...infoFor(d), video: 'workspace_video9' }));
  render(<App data={d} placeholder={false} />);
  await waitFor(() => a.match(document.body.textContent!, /was refused/));
  a.match(document.body.textContent!, /another sequence's geometry/);
  a.ok(screen.getByText('MEASURED'));
  a.doesNotMatch(document.body.textContent!, /run the model now/,
                 'live controls must not appear for a mismatched server');
});

// ── the doctor sign-in gate ──────────────────────────────────────────────────
// Reachable only when a server answered and asked for one. Double-clicked the page
// is on file://, makes no request, and none of this can appear.

async function settle(ms = 30) {
  await new Promise(r => setTimeout(r, ms));
}

test('a server that wants a sign-in replaces the demonstrator with the gate',
     async () => {
  stubFetch(() => ({ ok: true, live: false, auth_required: true, signed_in: false }));
  render(<App data={real()} placeholder={false} />);
  await settle();
  a.ok(screen.getByText(/Sign in to run the model/i));
  a.equal(screen.queryByText(SCENES[0].title), null,
          'the scenes rendered behind an unanswered sign-in');
});

test('the gate says what it does not protect, and offers the way past', async () => {
  stubFetch(() => ({ ok: true, live: false, auth_required: true, signed_in: false }));
  render(<App data={real()} placeholder={false} />);
  await settle();
  // Without this the screen implies the exported measurement is confidential. It is
  // on a USB stick and opens by double-click; claiming otherwise would be a lie.
  a.ok(screen.getByText(/not confidential/i));

  fireEvent.click(screen.getByText(/Continue to the exported demonstrator/i));
  await settle();
  a.ok(screen.getByText(SCENES[0].title), 'the way past the gate did not work');
});

test('no sign-in screen appears when the server does not ask for one', async () => {
  stubFetch(() => ({ ok: true, live: false, auth_required: false, signed_in: true }));
  render(<App data={real()} placeholder={false} />);
  await settle();
  a.equal(screen.queryByText(/Sign in to run the model/i), null);
  a.ok(screen.getByText(SCENES[0].title));
});

test('the gate never appears with no server at all', async () => {
  stubFetch();                       // every call throws: nothing is listening
  render(<App data={real()} placeholder={false} />);
  await settle();
  a.equal(screen.queryByText(/Sign in to run the model/i), null);
  a.ok(screen.getByText(SCENES[0].title));
});

test('a wrong password keeps the gate up and shows the reason', async () => {
  stubFetch(url => url === '/api/login'
    ? reply(401, { error: 'that roll number and password do not match' })
    : { ok: true, live: false, auth_required: true, signed_in: false });
  render(<App data={real()} placeholder={false} />);
  await settle();
  fireEvent.change(screen.getByLabelText(/Roll number/i),
                   { target: { value: 'doc-1' } });
  fireEvent.change(screen.getByLabelText(/Password/i),
                   { target: { value: 'wrong' } });
  fireEvent.click(screen.getByText(/^Sign in$/));
  await settle(80);
  a.ok(screen.getByText(/do not match/i));
  a.ok(screen.getByText(/Sign in to run the model/i),
       'a failed sign-in let the page through');
});
