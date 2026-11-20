const t = require('node:test'), a = require('node:assert');
const G = require('../demo/clinical/core.js');

t.test('unflat groups triples', () => {
  a.deepStrictEqual(G.unflat([1,2,3,4,5,6], 2), [[1,2,3],[4,5,6]]);
});

t.test('meanPointError averages euclidean distance', () => {
  a.strictEqual(G.meanPointError([[0,0,0],[0,0,0]], [[3,4,0],[0,0,0]]), 2.5);
});

t.test('motionTravelled measures ctx to gt distance', () => {
  a.strictEqual(G.motionTravelled([[0,0,0]], [[0,0,2]]), 2);
});

t.test('skillPercent is error as percent of motion', () => {
  a.strictEqual(G.skillPercent(0.5, 2), 25);
});

// Review Focus 2
t.test('skillPercent returns null when nothing moved', () => {
  a.strictEqual(G.skillPercent(0.5, 0), null);
  a.strictEqual(G.skillPercent(0.5, -1), null);
});

// Not in the plan's test list, but spanPercent is in its Produces list and shares the
// divide-by-zero hazard that Review Focus 2 raises for skillPercent.
t.test('spanPercent is error as percent of span, and null without a span', () => {
  a.strictEqual(G.spanPercent(0.5, 2), 25);
  a.strictEqual(G.spanPercent(0.5, 0), null);
  a.strictEqual(G.spanPercent(0.5, null), null);
});

t.test('rankAtHorizon orders methods by error ascending', () => {
  const me = {'3': {ours: 0.010, cv: 0.014, still: 0.016}};
  a.deepStrictEqual(G.rankAtHorizon(me, 3).map(r => r.key), ['ours','cv','still']);
});

t.test('verdict names ours when ours is lowest', () => {
  const me = {'3': {ours: 0.010, cv: 0.014, still: 0.016}};
  const v = G.verdict(me, 3);
  a.strictEqual(v.winner, 'ours');
  a.strictEqual(v.oursBeatsAll, true);
  a.match(v.text, /closest/i);
});

t.test('verdict names the baseline honestly when a baseline wins', () => {
  const me = {'3': {ours: 0.0225, cv: 0.0187, still: 0.0222}};
  const v = G.verdict(me, 3);
  a.strictEqual(v.winner, 'cv');
  a.strictEqual(v.oursBeatsAll, false);
  a.strictEqual(v.oursBeatsStatic, false);
  a.match(v.text, /Straight-line motion/);
});

// Review Focus 1
t.test('verdict and rank return null when there is no measurement', () => {
  a.strictEqual(G.verdict(null, 3), null);
  a.strictEqual(G.rankAtHorizon(null, 3), null);
  a.strictEqual(G.verdict({'1': {}}, 3), null);
});

t.test('horizonLabel states frames only without timing', () => {
  a.strictEqual(G.horizonLabel(3, null), '3 frames ahead');
});

// Review Focus 4
t.test('horizonLabel refuses to state time when timing is unusable', () => {
  a.strictEqual(G.horizonLabel(3, {timing_ok: false, fps: 30, gap_min: 2, gap_max: 10}),
                '3 frames ahead');
  a.strictEqual(G.horizonLabel(3, {timing_ok: true}), '3 frames ahead');
});

t.test('horizonLabel gives a range, never a single figure, when timing is good', () => {
  const s = G.horizonLabel(3, {timing_ok: true, fps: 30, gap_min: 2, gap_max: 10});
  a.match(s, /3 frames ahead/);
  a.match(s, /0\.2.*1\.0\s*s/);
});
