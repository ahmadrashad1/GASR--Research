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
