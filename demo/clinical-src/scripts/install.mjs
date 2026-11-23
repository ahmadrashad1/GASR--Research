// Put the built page where the data already lives.
//
// The deliverable is demo/clinical/index.html, beside demo_data.js and frames/. The
// source tree stays outside that folder so node_modules never travels onto the USB
// stick that goes to the viva.
//
// This also refuses to ship a build that cannot open from file://. Vite emits an ES
// module by default; a module script is blocked on the file:// origin, so the page
// would work perfectly over the dev server and be blank on the examiner's laptop.
// That failure is invisible until the worst possible moment, so it is checked here.
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const built = join(root, 'dist', 'index.html');
const target = join(root, '..', 'clinical', 'index.html');

if (!existsSync(built)) {
  console.error(`no build output at ${built}`);
  process.exit(1);
}
let html = readFileSync(built, 'utf8');

// Vite always emits <script type="module"> for an HTML entry, and a module script is
// refused on the file:// origin -- the page would work over the dev server and be
// blank on the examiner's laptop. The inlined bundle is a single self-contained chunk
// with no module syntax, so it is a valid classic script; this downgrades the tag
// AFTER proving that, rather than assuming it.
const mod = html.match(/<script type="module"[^>]*>([\s\S]*?)<\/script>/);
if (mod) {
  const js = mod[1];
  const needsModule =
    /(^|[;\n])\s*import\s*[{*'"(]/.test(js) ||
    /(^|[;\n])\s*export\s*[{ ]/.test(js) ||
    js.includes('import.meta') ||
    /[^.\w]import\s*\(/.test(js);
  if (needsModule) {
    console.error('\n  The bundle uses module syntax and cannot be served from file://.');
    console.error('  Remove the dynamic import or top-level export that caused it.');
    process.exit(1);
  }
  html = html.replace(/<script type="module"[^>]*>/, '<script>');
}

const faults = [];
if (/<script[^>]+type\s*=\s*["']module["']/.test(html)) {
  faults.push('still contains a module script, which file:// refuses to load');
}
if (/<script[^>]+\bsrc\s*=\s*["'](?!demo_data\.js)/.test(html)) {
  faults.push('references an external script; the bundle must be self-contained');
}
if (/<link[^>]+stylesheet/.test(html)) {
  faults.push('references an external stylesheet; CSS must be inlined');
}
// fetch is allowed, but only to the local inference server's API, and only because
// the page checks location.protocol before calling it.
//
// WHERE THE REAL GUARANTEE LIVES: src/live.test.ts asserts that probeLive() on a
// file:// origin makes no request AT ALL, by counting calls against a stubbed fetch.
// That is a behavioural proof and it is what this constraint actually rests on.
//
// The checks below are a backstop with a known blind spot, stated rather than
// implied: the bundle calls fetch through a local alias, so the call-site regex sees
// nothing in our own code and would only catch a NEW, direct `fetch('something')`.
// That is still worth catching -- fetching a local file is exactly the mistake that
// makes the page work over the dev server and come up blank when double-clicked --
// but it is not what proves the file:// path safe.
if (/XMLHttpRequest/.test(html)) {
  faults.push('uses XMLHttpRequest; the live path uses fetch to /api/ and nothing else');
}
const fetches = html.match(/\bfetch\s*\(\s*["'`][^"'`]*/g) ?? [];
const badFetch = fetches.filter(f => !/["'`]\/api\//.test(f));
if (badFetch.length) {
  faults.push(`fetches something other than the live API: ${badFetch.slice(0, 2)
    .map(f => f.slice(0, 40)).join(', ')}`);
}
if (/\bfetch\s*\(\s*[^"'`\s)]/.test(html)) {
  faults.push('calls fetch with a computed URL; the guard cannot verify it stays local');
}
if (!/createElement\(\s*["'`]script["'`]\s*\)/.test(html)) {
  faults.push('no injected script tag: demo_data.js must load without fetch');
}
// The live path must be present and must still be gated on the protocol. If either
// literal has gone, a refactor has either dropped live mode or -- worse -- dropped
// the check that keeps it from firing on file://.
for (const [needle, why] of [
  ['/api/health', 'the live health probe is missing from the bundle'],
  ['/api/predict', 'the live prediction call is missing from the bundle'],
  ['protocol', 'the location.protocol check that keeps file:// from requesting is gone'],
]) {
  if (!html.includes(needle)) faults.push(why);
}
// Only references that actually LOAD something count. React's minified error URL and
// a Tailwind credit in a CSS comment are inert strings, and the w3.org XML namespaces
// are identifiers rather than requests.
const loads = html.match(/(?:\bsrc|\bhref)\s*=\s*["']https?:\/\/[^"']+/gi) ?? [];
const external = loads.filter(u => !/w3\.org/.test(u));
if (external.length) {
  faults.push(`loads from the network: ${external.slice(0, 2).join(', ')}`);
}
if (faults.length) {
  console.error('\n  BUILD WILL NOT OPEN BY DOUBLE-CLICK:');
  for (const f of faults) console.error(`    - ${f}`);
  process.exit(1);
}

writeFileSync(target, html);
const kb = (Buffer.byteLength(html) / 1024).toFixed(0);
console.log(`  -> ${target}  (${kb} KB, self-contained, opens from file://)`);
