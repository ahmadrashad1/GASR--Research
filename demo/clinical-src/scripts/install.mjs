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
if (/\bfetch\s*\(|XMLHttpRequest/.test(html)) {
  faults.push('uses fetch or XHR, which cannot read a local file');
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
