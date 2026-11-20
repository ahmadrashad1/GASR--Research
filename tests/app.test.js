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

// ── Boot smoke test ──────────────────────────────────────────────────────────
// The plan's Step 4 asks for a human to double-click the file and confirm the scenes
// advance and the console is clean. That cannot run here, so this executes the page's
// own inline script against a stub DOM instead and walks every scene at every horizon.
// It will not catch a layout or colour mistake, but it does catch the failures that
// would end the demonstration in the room: a syntax error, a mistyped element id, or an
// exception thrown mid-render. A human still has to look at it.
function stubDom(){
  const noop = () => {};
  const ctx = new Proxy({}, { get: (o,k) => (k in o ? o[k] : noop), set: (o,k,v) => (o[k]=v, true) });
  // Only ids that genuinely appear in the markup resolve. A stub that fabricates an
  // element for any string cannot fail on a mistyped id, which is one of the two
  // failures this test exists to catch.
  const known = new Set([...html.matchAll(/id=["']([^"']+)["']/g)].map(m => m[1]));
  const made = new Map();
  const el = (id) => {
    if (!known.has(id)) return null;          // what the browser returns
    if (made.has(id)) return made.get(id);
    const e = {
      id, style:{}, textContent:'', innerHTML:'', className:'',
      classList:{ add:noop, remove:noop, toggle:noop, contains:()=>false },
      addEventListener:noop, removeAttribute:noop, setAttribute:noop,
      setPointerCapture:noop, getContext:()=>ctx,
      getBoundingClientRect:()=>({width:1280, height:720}),
      width:0, height:0,
    };
    e.parentElement = { getBoundingClientRect: () => ({width:1280, height:720}) };
    made.set(id, e);
    return e;
  };
  return { el, document:{ getElementById: el } };
}

t.test('the page boots and every scene renders at every horizon', () => {
  const src = [...html.matchAll(/<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/g)]
    .map(m => m[1]).filter(s => s.includes('SCENES')).pop();
  a.ok(src, 'inline script not found');

  const dom = stubDom();
  const keyHandlers = [];
  const win = {
    synth: require('../demo/clinical/synth.js'),
    devicePixelRatio: 1,
  };
  const addEventListener = (type, fn) => { if (type === 'keydown') keyHandlers.push(fn); };

  const run = new Function('GASR','SCENES','window','document','addEventListener',
                           'setInterval','clearInterval', src);
  run(require('../demo/clinical/core.js'),
      require('../demo/clinical/scenes.js'),
      win, dom.document, addEventListener, () => 0, () => {});

  a.strictEqual(keyHandlers.length, 1, 'no keyboard handler was registered');

  // Walk forward through every scene, toggling the horizon at each one. Any throw here
  // is a scene that would go blank on the projector.
  for (let i = 0; i < 8; i++){
    for (const key of ['1','3','r',' ']) keyHandlers[0]({key, preventDefault(){}});
    keyHandlers[0]({key:'ArrowRight', preventDefault(){}});
  }
  keyHandlers[0]({key:'ArrowLeft', preventDefault(){}});
});

t.test('placeholder mode never renders a verdict', () => {
  const synth = require('../demo/clinical/synth.js');
  const G = require('../demo/clinical/core.js');
  const d = synth();
  for (const k of d.horizons){
    a.strictEqual(G.verdict(d.mean_error, k), null);
    a.strictEqual(G.rankAtHorizon(d.mean_error, k), null);
  }
});
