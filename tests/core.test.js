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

// ── Task 2: scenes and placeholder geometry ──────────────────────────────────
const S = require('../demo/clinical/scenes.js');
const synth = require('../demo/clinical/synth.js');

t.test('there are exactly seven scenes in spec order', () => {
  a.strictEqual(S.length, 7);
  a.deepStrictEqual(S.map(s => s.id),
    ['sees','reconstructs','why','prediction','baselines','uncertainty','standing']);
});

t.test('every scene declares only known layers', () => {
  const ok = new Set(['frame','overlay','ctx','gt','pred','cv','still','sigma']);
  for (const s of S) {
    a.ok(s.title && s.body, `${s.id} needs title and body`);
    for (const l of s.layers) a.ok(ok.has(l), `${s.id}: unknown layer ${l}`);
  }
});

t.test('endoscopic imagery appears only in the scenes that declare it', () => {
  // The spec originally ruled out drawing predictions on the frame, assuming a true
  // overlay needed extrinsics out of the splatting workspace. Module 6's own
  // project() shows the trajectories are already in camera space, so only intrinsics
  // are involved and the overlay is cheap. Scenes 4-6 therefore declare it.
  a.deepStrictEqual(S.filter(s => s.layers.includes('frame')).map(s => s.id),
                    ['sees','reconstructs']);
  a.deepStrictEqual(S.filter(s => s.layers.includes('overlay')).map(s => s.id),
                    ['prediction','baselines','uncertainty']);
});

t.test('no scene hard-codes a comparative claim', () => {
  const banned = /\b(outperform|beats|better than|best|wins|superior|state of the art)\b/i;
  for (const s of S) a.doesNotMatch(s.body, banned, `${s.id} states a verdict`);
});

t.test('no scene states an absolute physical unit', () => {
  const banned = /\b\d+(\.\d+)?\s*(mm|cm|millimet|ms|millisecond)/i;
  for (const s of S) a.doesNotMatch(s.body, banned, `${s.id} invents a unit`);
});

t.test('synth is shaped like the real payload and claims nothing', () => {
  const d = synth();
  a.strictEqual(d.synthetic, true);
  a.strictEqual(d.mean_error, null);
  a.deepStrictEqual(d.horizons, [1, 3]);
  a.strictEqual(d.frames.length, d.n_frames);
  for (const k of d.horizons)
    for (const tag of ['gt','pred','cv'])
      a.strictEqual(d.frames[0][tag+k].length, d.n_points*3, `${tag}${k} wrong length`);
  a.strictEqual(d.frames[0].sigma1.length, d.n_points);
  a.strictEqual(d.frames[0].sigma3.length, d.n_points);
});

t.test('the strings table states no verdict and no physical unit', () => {
  // The scene bodies and this table are both clinician-facing, so both are reviewed.
  // The verdict pattern here is narrower than the one applied to scene bodies on
  // purpose: STR.scope has to explain the metric with the words "no better than not
  // predicting", which is a definition of the denominator, not a claim about a method.
  // What must never appear is a method named as winning.
  const units = /\b\d+(\.\d+)?\s*(mm|cm|millimet|ms|millisecond)/i;
  const wins  = /\b(outperform|beats|best|superior|most accurate|wins|state of the art)\b/i;
  a.ok(S.strings && Object.keys(S.strings).length, 'no strings table found');
  for (const [k, v] of Object.entries(S.strings)) {
    a.strictEqual(typeof v, 'string', k + ' is not a string');
    a.doesNotMatch(v, units, k + ' invents a physical unit');
    a.doesNotMatch(v, wins, k + ' states a verdict');
  }
});

t.test('no clinician-facing string is left behind in the page', () => {
  // I3: prose in index.html escapes the review that scenes.js exists to make possible.
  // These are the fragments that were moved; if one reappears as a literal, this fails.
  const fs2 = require('node:fs');
  const page = fs2.readFileSync('demo/clinical/index.html', 'utf8');
  for (const gone of ['Placeholder geometry carries no measurement',
                      'Spread across sampled futures',
                      'What actually happened',
                      'invented geometry for rehearsal',
                      'The first column is the error']) {
    a.ok(!page.includes(gone), 'moved back into the page: ' + gone);
  }
});

t.test('displayGain amplifies motion that is below the resolution of the screen', () => {
  // Tissue moves ~2 px over three frames at 640x360, and the methods differ by ~1 px.
  // Drawn at true scale every series lands inside one dot. Module 6 faced this and
  // records an arrow_gain of 5 and 100 on its own figures; the gain is honest only if
  // it is stated, which is why it is computed rather than hard-coded.
  a.strictEqual(G.displayGain(2, 45), 20);
  a.strictEqual(G.displayGain(45, 45), 1);
});

t.test('displayGain never shrinks motion and never divides by zero', () => {
  a.strictEqual(G.displayGain(500, 45), 1, 'large motion is already visible');
  a.strictEqual(G.displayGain(0, 45), 1);
  a.strictEqual(G.displayGain(-3, 45), 1);
  a.strictEqual(G.displayGain(null, 45), 1);
});

t.test('displayGain returns a round number a presenter can say out loud', () => {
  for (const m of [0.1, 0.4, 1, 2, 7, 19, 44]) {
    const g = G.displayGain(m, 45);
    a.ok([1,2,5,10,20,50,100,200,500].includes(g), `${m} -> ${g} is not a round gain`);
  }
});
