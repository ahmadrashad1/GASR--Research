import { test } from 'vitest';
import a from 'node:assert/strict';
import * as G from './core';
import { SCENES, STR } from './scenes';
import { synth } from './synth';

test('unflat groups triples', () => {
  a.deepStrictEqual(G.unflat([1,2,3,4,5,6], 2), [[1,2,3],[4,5,6]]);
});

test('meanPointError averages euclidean distance', () => {
  a.strictEqual(G.meanPointError([[0,0,0],[0,0,0]], [[3,4,0],[0,0,0]]), 2.5);
});

test('motionTravelled measures ctx to gt distance', () => {
  a.strictEqual(G.motionTravelled([[0,0,0]], [[0,0,2]]), 2);
});

test('skillPercent is error as percent of motion', () => {
  a.strictEqual(G.skillPercent(0.5, 2), 25);
});

// Review Focus 2
test('skillPercent returns null when nothing moved', () => {
  a.strictEqual(G.skillPercent(0.5, 0), null);
  a.strictEqual(G.skillPercent(0.5, -1), null);
});

test('spanPercent is error as percent of span, and null without a span', () => {
  a.strictEqual(G.spanPercent(0.5, 2), 25);
  a.strictEqual(G.spanPercent(0.5, 0), null);
  a.strictEqual(G.spanPercent(0.5, null), null);
});

test('rankAtHorizon orders methods by error ascending', () => {
  const me = {'3': {ours: 0.010, cv: 0.014, still: 0.016}};
  a.deepStrictEqual(G.rankAtHorizon(me, 3)!.map(r => r.key), ['ours','cv','still']);
});

test('verdict names ours when ours is lowest', () => {
  const v = G.verdict({'3': {ours: 0.010, cv: 0.014, still: 0.016}}, 3)!;
  a.strictEqual(v.winner, 'ours');
  a.strictEqual(v.oursBeatsAll, true);
  a.match(v.text, /closest/i);
});

test('verdict names the baseline honestly when a baseline wins', () => {
  const v = G.verdict({'3': {ours: 0.0225, cv: 0.0187, still: 0.0222}}, 3)!;
  a.strictEqual(v.winner, 'cv');
  a.strictEqual(v.oursBeatsAll, false);
  a.strictEqual(v.oursBeatsStatic, false);
  a.match(v.text, /Straight-line motion/);
});

// Review Focus 1
test('verdict and rank return null when there is no measurement', () => {
  a.strictEqual(G.verdict(null, 3), null);
  a.strictEqual(G.rankAtHorizon(null, 3), null);
  a.strictEqual(G.verdict({'1': {}}, 3), null);
});

// C1
test('an exact tie is reported as indistinguishable, not as a win for ours', () => {
  const v = G.verdict({'1': {ours: 0.47478, cv: 0.47478, still: 0.51380}}, 1)!;
  a.strictEqual(v.tied, true);
  a.strictEqual(v.oursBeatsAll, false);
  a.doesNotMatch(v.text, /closest prediction/i);
  a.match(v.text, /indistinguishable/i);
});

test('horizonLabel states frames only without timing', () => {
  a.strictEqual(G.horizonLabel(3, null), '3 frames ahead');
});

// Review Focus 4
test('horizonLabel refuses to state time when timing is unusable', () => {
  a.strictEqual(G.horizonLabel(3, {timing_ok: false, fps: 30, gap_min: 2, gap_max: 10}),
                '3 frames ahead');
  a.strictEqual(G.horizonLabel(3, {timing_ok: true}), '3 frames ahead');
});

test('horizonLabel gives a range, never a single figure, when timing is good', () => {
  const s = G.horizonLabel(3, {timing_ok: true, fps: 30, gap_min: 2, gap_max: 10});
  a.match(s, /3 frames ahead/);
  a.match(s, /0\.2.*1\.0\s*s/);
});

test('displayGain amplifies motion below the resolution of the screen', () => {
  a.strictEqual(G.displayGain(2, 45), 20);
  a.strictEqual(G.displayGain(45, 45), 1);
});

test('displayGain never shrinks motion and never divides by zero', () => {
  a.strictEqual(G.displayGain(500, 45), 1);
  a.strictEqual(G.displayGain(0, 45), 1);
  a.strictEqual(G.displayGain(-3, 45), 1);
  a.strictEqual(G.displayGain(null, 45), 1);
});

test('displayGain returns a round number a presenter can say out loud', () => {
  for (const m of [0.1, 0.4, 1, 2, 7, 19, 44]) {
    a.ok([1,2,5,10,20,50,100,200,500].includes(G.displayGain(m, 45)));
  }
});

// Module 1 always writes camera_meta.json, always with the nominal 75-degree fx.
test('a nominal field of view is not treated as measured', () => {
  const W = 640, fx = (W/2)/Math.tan(75*Math.PI/360);
  a.strictEqual(G.measuredCamera({fx, fy: fx, cx: 320, cy: 180, W, H: 360,
                                  calibrated: true}), false);
});

test('genuinely measured intrinsics are honoured', () => {
  a.strictEqual(G.measuredCamera({fx: 503.2, fy: 499.7, cx: 322.4, cy: 178.9,
                                  W: 640, H: 360, calibrated: true,
                                  source: 'measured'}), true);
});

test('there are exactly seven scenes in spec order', () => {
  a.strictEqual(SCENES.length, 7);
  a.deepStrictEqual(SCENES.map(s => s.id),
    ['sees','reconstructs','why','prediction','baselines','uncertainty','standing']);
});

test('every scene declares only known layers', () => {
  const ok = new Set(['frame','overlay','ctx','gt','pred','cv','still','sigma']);
  for (const s of SCENES) {
    a.ok(s.title && s.body);
    for (const l of s.layers) a.ok(ok.has(l), `${s.id}: unknown layer ${l}`);
  }
});

test('endoscopic imagery appears only in the scenes that declare it', () => {
  a.deepStrictEqual(SCENES.filter(s => s.layers.includes('frame')).map(s => s.id),
                    ['sees','reconstructs']);
  a.deepStrictEqual(SCENES.filter(s => s.layers.includes('overlay')).map(s => s.id),
                    ['prediction','baselines','uncertainty']);
});

test('no scene hard-codes a comparative claim', () => {
  const banned = /\b(outperform|beats|better than|best|wins|superior|state of the art)\b/i;
  for (const s of SCENES) a.doesNotMatch(s.body, banned, `${s.id} states a verdict`);
});

test('no scene states an absolute physical unit', () => {
  const banned = /\b\d+(\.\d+)?\s*(mm|cm|millimet|ms|millisecond)/i;
  for (const s of SCENES) a.doesNotMatch(s.body, banned, `${s.id} invents a unit`);
});

test('the strings table states no verdict and no physical unit', () => {
  // Narrower than the scene-body pattern on purpose: STR.scope must define the metric
  // with the words "no better than not predicting", which is not a claim about a method.
  const units = /\b\d+(\.\d+)?\s*(mm|cm|millimet|ms|millisecond)/i;
  const wins  = /\b(outperform|beats|best|superior|most accurate|wins|state of the art)\b/i;
  for (const [k, v] of Object.entries(STR)) {
    a.doesNotMatch(v, units, `${k} invents a physical unit`);
    a.doesNotMatch(v, wins, `${k} states a verdict`);
  }
});

test('synth is shaped like the real payload and claims nothing', () => {
  const d = synth();
  a.strictEqual(d.synthetic, true);
  a.strictEqual(d.mean_error, null);
  a.deepStrictEqual(d.horizons, [1, 3]);
  a.strictEqual(d.frames.length, d.n_frames);
  for (const k of d.horizons)
    for (const tag of ['gt','pred','cv'])
      a.strictEqual((d.frames[0][tag+k] as number[]).length, d.n_points*3);
  a.strictEqual((d.frames[0].sigma1 as number[]).length, d.n_points);
  a.strictEqual((d.frames[0].sigma3 as number[]).length, d.n_points);
});

test('placeholder mode never yields a verdict', () => {
  const d = synth();
  for (const k of d.horizons) {
    a.strictEqual(G.verdict(d.mean_error, k), null);
    a.strictEqual(G.rankAtHorizon(d.mean_error, k), null);
  }
});

test('placeholder mode carries no imagery and no overlay coordinates', () => {
  const d = synth();
  a.deepStrictEqual(d.frame_files, {});
  a.strictEqual(d.frames[0].px_ctx, undefined);
});
