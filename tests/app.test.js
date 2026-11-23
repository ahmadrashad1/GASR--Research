// Repo-level guard on the BUILT demonstrator.
//
// demo/clinical/index.html is a build artefact now, produced by demo/clinical-src.
// Its logic and components are tested there with Vitest; what is checked here is the
// one property no unit test can see and no browser test can check offline: that the
// shipped file still opens by double-click, with no server and no network.
//
// This duplicates the build's own install guard on purpose. The build check stops a
// bad artefact being produced; this stops a bad artefact being committed, which is
// the failure that would actually reach the viva.
const t = require('node:test'), a = require('node:assert');
const fs = require('node:fs');

const PAGE = 'demo/clinical/index.html';
const html = fs.readFileSync(PAGE, 'utf8');

t.test('the built page loads no module script', () => {
  // Blocked on the file:// origin. It would work over a dev server and be blank on
  // the examiner's laptop, which is the worst way for this to fail.
  a.doesNotMatch(html, /<script[^>]+type\s*=\s*["']module["']/);
});

t.test('the bundle runs after the element it mounts into exists', () => {
  // This one shipped. The build downgrades Vite's <script type="module"> to a classic
  // script so file:// will run it, but Vite had hoisted that tag into <head>, where
  // being deferred was the only thing making it safe. A classic script in <head> runs
  // the moment the parser reaches it -- before <div id="root"> -- so createRoot() got
  // null, React threw, and the page rendered nothing at all. Not on file:// only: over
  // http too. It was invisible to every other check here, because the file was still
  // self-contained, still loaded nothing, and still had no module syntax.
  //
  // `defer` cannot fix it: the attribute is defined only for scripts with a src and is
  // ignored on an inline one. The script has to physically follow the div.
  const rootAt = html.indexOf('id="root"');
  const mountAt = html.search(/getElementById\(\s*["'`]root["'`]\s*\)/);
  a.notStrictEqual(rootAt, -1, 'the page has no <div id="root"> to mount into');
  a.notStrictEqual(mountAt, -1, 'the bundle never mounts into #root');
  a.ok(mountAt > rootAt,
       `the bundle runs at offset ${mountAt}, before #root at ${rootAt}: createRoot() ` +
       'would be handed null and the page would come up blank');
});

t.test('the built page loads nothing from the network', () => {
  const loads = html.match(/(?:\bsrc|\bhref)\s*=\s*["']https?:\/\/[^"']+/gi) || [];
  a.deepStrictEqual(loads.filter(u => !/w3\.org/.test(u)), [],
                    'the page must not load anything over a network');
  a.doesNotMatch(html, /XMLHttpRequest/);
});

t.test('the page talks to the local server only, and only over http', () => {
  // The page may now call a local inference server -- but only when it was itself
  // served over http by serve/server.py, which makes the API same-origin. Opened by
  // double-click it is on file://, and src/live.ts returns early on the protocol
  // before any request is made (src/live.test.ts counts the calls and proves it).
  //
  // This asserts the two literals that gate it. It deliberately does NOT assert the
  // absence of `fetch(`: the bundle calls fetch through a local alias, so that
  // assertion passed while meaning nothing, which is worse than not having it.
  a.match(html, /\/api\/health/, 'the live probe is missing from the built page');
  a.match(html, /\/api\/predict/, 'the live prediction call is missing');
  a.match(html, /protocol/,
          'the protocol gate that stops file:// from requesting is gone');

  // Any DIRECT fetch call that survives minification must target the API. A new
  // fetch of a local file is the mistake this catches: file:// refuses it, so the
  // page would work over the server and come up blank when double-clicked.
  const direct = html.match(/\bfetch\s*\(\s*[`'"][^`'"]*/g) || [];
  a.deepStrictEqual(direct.filter(f => !/[`'"]\/api\//.test(f)), [],
                    'the page fetches something other than the local API');
});

t.test('the built page is self-contained', () => {
  const srcs = [...html.matchAll(/<script[^>]+src=["']([^"']+)["']/g)].map(m => m[1]);
  a.deepStrictEqual(srcs, [], 'scripts must be inlined; data is loaded at runtime');
  a.doesNotMatch(html, /<link[^>]+stylesheet/, 'CSS must be inlined');
});

t.test('the page still loads its data the only way file:// allows', () => {
  // A classic <script> injected at runtime. Not asserted as a literal tag, because
  // the loader builds it; asserted as the filename plus the element it creates.
  a.match(html, /demo_data\.js/);
  // The minifier may quote with ' " or a backtick, so all three are allowed.
  a.match(html, /createElement\([`'"]script[`'"]\)/,
          'the loader must inject a classic script element');
});

t.test('placeholder and measured are both reachable states', () => {
  a.match(html, /PLACEHOLDER DATA/);
  a.match(html, /MEASURED/);
});

t.test('the research viewer is untouched', () => {
  a.doesNotMatch(fs.readFileSync('demo/index.html', 'utf8'), /clinical/);
});

t.test('the build source is not inside the folder that goes on a USB stick', () => {
  a.ok(!fs.existsSync('demo/clinical/node_modules'),
       'node_modules must never travel with the deliverable');
  a.ok(!fs.existsSync('demo/clinical/package.json'));
  a.ok(fs.existsSync('demo/clinical-src/package.json'), 'source tree missing');
});
