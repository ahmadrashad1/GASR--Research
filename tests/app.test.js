// Static assertions on the shell. The file:// constraints are precisely the ones a
// headless browser test could not check offline, and they are the ones that decide
// whether this page opens at all on a laptop in an examination room with no network.
const t = require('node:test'), a = require('node:assert');
const fs = require('node:fs');
const html = fs.readFileSync('demo/clinical/index.html', 'utf8');

t.test('loads no module scripts and fetches no local data', () => {
  a.doesNotMatch(html, /type\s*=\s*["']module["']/);
  a.doesNotMatch(html, /\bfetch\s*\(/);
  a.doesNotMatch(html, /XMLHttpRequest/);
});

t.test('pulls nothing from the network', () => {
  a.doesNotMatch(html, /src\s*=\s*["']https?:/);
  a.doesNotMatch(html, /@import|cdn\.|unpkg|jsdelivr/);
});

t.test('every script it references exists on disk', () => {
  const srcs = [...html.matchAll(/<script[^>]+src=["']([^"']+)["']/g)].map(m => m[1]);
  a.ok(srcs.includes('core.js') && srcs.includes('scenes.js') && srcs.includes('synth.js'));
  for (const s of srcs)
    if (s !== 'demo_data.js') a.ok(fs.existsSync('demo/clinical/' + s), `missing ${s}`);
});

t.test('demo_data.js is optional, so its tag cannot block the page', () => {
  a.match(html, /<script[^>]+src=["']demo_data\.js["'][^>]*>/);
  a.match(html, /onerror/, 'the optional data script needs an onerror fallback');
});

t.test('uses the fixed palette', () => {
  for (const c of ['#D62828','#5C0A0A','#C98276','#F5F5F7']) a.match(html, new RegExp(c, 'i'));
});

t.test('carries the placeholder banner and the no-imagery panel', () => {
  a.match(html, /PLACEHOLDER DATA/);
  a.match(html, /id=["']noimg["']/);
});

t.test('the research viewer is untouched by this task', () => {
  a.doesNotMatch(fs.readFileSync('demo/index.html','utf8'), /clinical/);
});


// ── A harness that can boot the page against REAL-mode payloads ───────────────
// The previous version only ever booted placeholder data, so Review Focus 3 and 5 --
// both assigned to this task -- had no automated coverage at all. This one injects
// window.DEMO_DATA, records classList/src/textContent transitions, and exposes the
// keyboard handler, the <img> error listener and the playback tick, so the degradation
// paths can be driven rather than reasoned about.
function harness(payload){
  const known = new Set([...html.matchAll(/id=["']([^"']+)["']/g)].map(m => m[1]));
  const arcs = [], ovlArcs = [];
  const mkCtx = sink => new Proxy({}, {
    get: (o, k) => (k === 'arc' ? ((x,y,r) => sink.push(r)) : (k in o ? o[k] : () => {})),
    set: (o, k, v) => (o[k] = v, true),
  });
  const made = new Map(), listeners = {};

  function mk(id){
    const cls = new Set();
    const e = {
      id, style:{}, textContent:'', innerHTML:'', className:'', width:0, height:0, _src:null,
      classList:{
        add: c => cls.add(c), remove: c => cls.delete(c),
        toggle: (c, on) => (on === undefined ? (cls.has(c) ? cls.delete(c) : cls.add(c))
                                             : (on ? cls.add(c) : cls.delete(c))),
        contains: c => cls.has(c),
      },
      _cls: cls,
      addEventListener: (t, fn) => { (listeners[id] = listeners[id] || {})[t] = fn; },
      removeAttribute: at => { if (at === 'src') e._src = null; },
      setAttribute: () => {}, setPointerCapture: () => {},
      getContext: () => mkCtx(id === 'ovl' ? ovlArcs : arcs),
      getBoundingClientRect: () => ({width:1280, height:720}),
    };
    Object.defineProperty(e, 'src', { get: () => e._src, set: v => { e._src = v; } });
    e.parentElement = { getBoundingClientRect: () => ({width:1280, height:720}) };
    return e;
  }

  const document = { getElementById: id => {
    if (!known.has(id)) return null;                 // what the browser returns
    if (!made.has(id)) made.set(id, mk(id));
    return made.get(id);
  }};

  const win = { synth: require('../demo/clinical/synth.js'), devicePixelRatio: 1 };
  if (payload) win.DEMO_DATA = payload;

  const keys = [];
  let tick = null;
  const src = [...html.matchAll(/<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/g)]
    .map(m => m[1]).filter(s => s.includes('SCENES')).pop();
  if (!src) throw new Error('inline script not found');

  new Function('GASR','SCENES','window','document','addEventListener',
               'setInterval','clearInterval', src)(
    require('../demo/clinical/core.js'),
    require('../demo/clinical/scenes.js'),
    win, document,
    (t, fn) => { if (t === 'keydown') keys.push(fn); },
    fn => { tick = fn; return 1; },
    () => { tick = null; });

  if (!keys.length) throw new Error('no keyboard handler registered');
  return {
    el: id => document.getElementById(id),
    key: k => keys[0]({ key: k, preventDefault(){} }),
    fire: (id, type) => listeners[id] && listeners[id][type] &&
                        listeners[id][type].call(document.getElementById(id)),
    tick: () => tick && tick(),
    arcs, ovlArcs,
  };
}

// A payload shaped like the real export: measured, with a motion denominator and the
// calibration metadata Task 4 added.
function realPayload(over){
  const d = require('../demo/clinical/synth.js')();
  d.synthetic = false;
  d.video = 'workspace_video4';
  d.checkpoint_epoch = 100;
  d.mean_error = { '1': {ours:0.00611, cv:0.00590, still:0.00944},
                   '3': {ours:0.02000, cv:0.02272, still:0.02255} };
  d.motion = { '1': 0.01000, '3': 0.02400 };
  d.timing = { timing_ok:false };
  d.frame_files = {};
  d.degenerate_horizons = [];
  Object.assign(d, over || {});
  return d;
}
function withFrameFiles(d){
  d.frame_files = {};
  for (const f of d.frames){
    for (const i of [f.img, f.img1, f.img3])
      if (i != null) d.frame_files[String(i)] = 'frames/' + String(i).padStart(5,'0') + '.jpg';
  }
  return d;
}
const SCENE_IX = {sees:0, reconstructs:1, why:2, prediction:3, baselines:4, uncertainty:5, standing:6};
function goTo(h, id){ for (let i = 0; i < SCENE_IX[id]; i++) h.key('ArrowRight'); return h; }

t.test('the page boots and every scene renders at every horizon', () => {
  const h = harness(null);
  for (let i = 0; i < 8; i++){
    for (const k of ['1','3','r',' ']) h.key(k);
    h.key('ArrowRight');
  }
  h.key('ArrowLeft');
});

t.test('placeholder mode never renders a verdict', () => {
  const G = require('../demo/clinical/core.js');
  const d = require('../demo/clinical/synth.js')();
  for (const k of d.horizons){
    a.strictEqual(G.verdict(d.mean_error, k), null);
    a.strictEqual(G.rankAtHorizon(d.mean_error, k), null);
  }
});

// ── C3 ───────────────────────────────────────────────────────────────────────
t.test('real endoscopic frames and the payload cannot be committed by accident', () => {
  const gi = fs.readFileSync('.gitignore', 'utf8');
  a.match(gi, /demo\/clinical\/frames/, 'frames/ holds real surgical stills');
  a.match(gi, /demo\/clinical\/demo_data\.js/);
});

// ── C2 ───────────────────────────────────────────────────────────────────────
t.test('a degenerate horizon draws no halo labelled as sampled spread', () => {
  const d = realPayload({ degenerate_horizons: ['1'] });
  for (const f of d.frames){ f.sigma1 = f.sigma1.map(() => 0); f.pred1 = f.cv1.slice(); }
  const h = harness(d);
  goTo(h, 'uncertainty');
  h.key('1');
  a.doesNotMatch(h.el('legend').innerHTML, /spread across sampled futures/i,
                 'a zero spread must not be presented as a measured one');
  a.match(h.el('warn').textContent, /\S/, 'the degenerate horizon must be stated on screen');
});

// ── C1 ───────────────────────────────────────────────────────────────────────
t.test('an exact tie is reported as indistinguishable, not as a win for ours', () => {
  const G = require('../demo/clinical/core.js');
  const v = G.verdict({ '1': {ours:0.47478, cv:0.47478, still:0.51380} }, 1);
  a.strictEqual(v.tied, true);
  a.strictEqual(v.oursBeatsAll, false);
  a.doesNotMatch(v.text, /closest prediction/i);
  a.match(v.text, /indistinguishable/i);
});

t.test('a real win is still reported as a win', () => {
  const G = require('../demo/clinical/core.js');
  const v = G.verdict({ '3': {ours:0.02000, cv:0.02272, still:0.02255} }, 3);
  a.strictEqual(v.tied, false);
  a.strictEqual(v.oursBeatsAll, true);
  a.match(v.text, /closest/i);
});

// ── I1 ───────────────────────────────────────────────────────────────────────
t.test('a frame that failed to load stays hidden across scene changes and ticks', () => {
  const h = harness(withFrameFiles(realPayload()));
  a.ok(h.el('shot').src, 'scene 1 should have tried to load a still');
  h.fire('shot', 'error');
  a.strictEqual(h.el('noimg')._cls.has('on'), true);
  h.key('ArrowRight');
  a.strictEqual(h.el('noimg')._cls.has('on'), true,
                'the panel must not be torn down on the next scene');
  h.tick();
  a.strictEqual(h.el('noimg')._cls.has('on'), true,
                'nor re-exposed nine times a second during playback');
});

// ── I2 ───────────────────────────────────────────────────────────────────────
t.test('a frame-only scene clears the legend and does not split the stage', () => {
  const h = harness(realPayload());
  goTo(h, 'standing');
  a.match(h.el('legend').innerHTML, /\S/, 'scene 7 should have a legend');
  h.key('ArrowRight');                               // wraps to scene 1, frame only
  a.strictEqual(h.el('legend').innerHTML, '',
                'a stale legend must not float over an empty stage');
  a.strictEqual(h.el('stage')._cls.has('withframe'), false);
});

// ── I3 ───────────────────────────────────────────────────────────────────────
t.test('a missing horizon produces a grammatical sentence', () => {
  const d = realPayload();
  for (const f of d.frames){ delete f.gt3; delete f.sigma3; }
  const h = harness(d);
  goTo(h, 'prediction');
  const body = h.el('body').textContent;
  a.doesNotMatch(body, /\bno the\b/i, 'articles doubled up');
  a.match(body, /recorded outcome/i);
});

// ── M1, promoted ─────────────────────────────────────────────────────────────
t.test('scene prose states the horizon that is actually selected', () => {
  const h = harness(realPayload());
  goTo(h, 'prediction');
  a.match(h.el('body').textContent, /3 frames ahead/);
  h.key('1');
  a.match(h.el('body').textContent, /1 frame ahead/);
  a.doesNotMatch(h.el('body').textContent, /three frames/i);
});

// ── M4, promoted ─────────────────────────────────────────────────────────────
t.test('a payload with no horizons list still renders instead of going blank', () => {
  const d = realPayload();
  delete d.horizons;
  const h = harness(d);
  a.match(h.el('title').textContent, /\S/);
});

// ── Overlay: the prediction drawn on the endoscopic frame ────────────────────
// The export projects; the page only draws where it is told. These check that it draws
// when it can, degrades when it cannot, and never presents uncalibrated intrinsics with
// the confidence of measured ones.
function withOverlay(d){
  d = withFrameFiles(d);
  d.camera = {fx:503.2, fy:499.7, cx:322.4, cy:178.9, W:640, H:360,
              calibrated:true, source:'measured'};
  for (const f of d.frames){
    const n = d.n_points;
    const uv = () => { const o = []; for (let i=0;i<n;i++) o.push(0.2 + 0.6*i/n, 0.3 + 0.4*i/n); return o; };
    f.px_ctx = uv();
    for (const k of [1,3]){ f['px_gt'+k] = uv(); f['px_pred'+k] = uv(); f['px_cv'+k] = uv(); }
  }
  return d;
}

t.test('a scene with the overlay layer draws onto the frame', () => {
  const h = harness(withOverlay(realPayload()));
  goTo(h, 'prediction');
  a.ok(h.el('ovl'), 'the overlay canvas must exist');
  a.strictEqual(h.el('stage')._cls.has('withframe'), true,
                'the overlay scene must show the endoscopic still');
  a.ok(h.ovlArcs.length > 0, 'nothing was drawn on the overlay');
});

t.test('an export without pixel coordinates still renders, without an overlay', () => {
  // The payload that is on Drive today predates the projection. Loading it must not
  // throw and must not leave an empty canvas floating over the video.
  const h = harness(withFrameFiles(realPayload()));
  goTo(h, 'prediction');
  a.strictEqual(h.ovlArcs.length, 0, 'drew an overlay with no coordinates to draw');
  a.match(h.el('body').textContent, /\S/);
});

t.test('an uncalibrated overlay says so rather than passing as measured', () => {
  const d = withOverlay(realPayload());
  d.camera.calibrated = false;
  const h = harness(d);
  goTo(h, 'prediction');
  a.match(h.el('warn').textContent, /nominal|not a measurement|assumed/i,
          'a guessed field of view must be declared on screen');
});

t.test('a calibrated overlay raises no warning of its own', () => {
  const h = harness(withOverlay(realPayload()));
  goTo(h, 'prediction');
  a.doesNotMatch(h.el('warn').textContent, /calibrat/i);
});

t.test('placeholder mode never draws an overlay', () => {
  const h = harness(null);
  for (let i = 0; i < 7; i++){ h.key('ArrowRight'); }
  a.strictEqual(h.ovlArcs.length, 0, 'invented geometry must not be drawn on imagery');
});

t.test('an assumed field of view is declared even when the payload claims calibration', () => {
  // The export already on Drive was written before camera_from learned to detect this,
  // so it carries calibrated:true over Module 1's nominal 75-degree intrinsics. The
  // page re-derives the signature rather than trusting the flag, so an existing
  // payload does not have to be regenerated to stop overclaiming.
  const d = withOverlay(realPayload());
  const W = 640;
  d.camera = {fx: (W/2)/Math.tan(75*Math.PI/360), fy: (W/2)/Math.tan(75*Math.PI/360),
              cx: 320, cy: 180, W: W, H: 360, calibrated: true};
  const h = harness(d);
  goTo(h, 'prediction');
  a.match(h.el('warn').textContent, /assumed|not .*measured|nominal/i,
          'a nominal field of view must not pass as a measured one');
});

t.test('genuinely measured intrinsics raise no camera warning', () => {
  const d = withOverlay(realPayload());
  d.camera = {fx: 503.2, fy: 499.7, cx: 322.4, cy: 178.9, W: 640, H: 360,
              calibrated: true, source: 'measured'};
  const h = harness(d);
  goTo(h, 'prediction');
  a.doesNotMatch(h.el('warn').textContent, /field of view/i);
});
