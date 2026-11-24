// FYP-2 final presentation — theme and design sampler for supervisor approval.
//
// Built to the academic-pptx skill: white content slides, one sans-serif, action
// titles, one exhibit per results slide, figure left / interpretation right,
// conclusions last before references.
//
// Theme: white and blood red. The palette is the demonstrator's own, so the deck and
// the live demo look like one project. Everything restyles from COLORS below.
const pptxgen = require('pptxgenjs');
const path = require('path');

const OUT = process.argv[2] || path.join(__dirname, 'GASR_FYP2_theme.pptx');

const COLORS = {
  bg:      'FFFFFF',
  primary: '5C0A0A',   // deep crimson — titles, dark slide grounds
  accent:  'D62828',   // blood red — the signal colour, used only to direct the eye
  body:    '2D2D2D',   // near-black
  muted:   '7A7A80',   // citations, captions
  rule:    'E2DCDC',   // warm light grey dividers
  onDark:  'FFFFFF',
  onDarkMuted: 'E8BFBF',
  grid:    'EFEBEB',
};
const F = { face: 'Arial', title: 25, sect: 19, body: 18, label: 15, cite: 12 };
const M = 0.5;

const pres = new pptxgen();
pres.layout = 'LAYOUT_16x9';          // 10 x 5.625 in — set BEFORE any slide
pres.author = 'Ahmad Rashad, Shehroz Kashif, Humair Khan';
pres.title = 'GASR FYP-2 Final Presentation';

// ── layouts ────────────────────────────────────────────────────────────────
// Frames only. Content is composed per slide, which is what the academic skill's
// patterns assume; the layouts exist so the theme changes in one place.
pres.defineSlideMaster({
  title: 'DARK',
  background: { color: COLORS.primary },
  objects: [
    { rect: { x: 0, y: 0, w: 0.14, h: 5.625, fill: { color: COLORS.accent } } },
  ],
});
pres.defineSlideMaster({
  title: 'CONTENT',
  background: { color: COLORS.bg },
  objects: [
    { rect: { x: 0, y: 5.46, w: 10, h: 0.03, fill: { color: COLORS.rule } } },
  ],
  slideNumber: { x: 9.3, y: 5.16, w: 0.5, h: 0.25, fontFace: F.face,
                 fontSize: 11, color: COLORS.muted, align: 'right' },
});

// ── helpers ────────────────────────────────────────────────────────────────
function actionTitle(slide, text) {
  slide.addText(text, {
    x: M, y: 0.26, w: 9.0, h: 0.86, isTextBox: true,
    fontSize: F.title, fontFace: F.face, color: COLORS.primary,
    bold: true, valign: 'top', align: 'left',
  });
  slide.addShape(pres.shapes.RECTANGLE, {
    x: M, y: 1.16, w: 9.0, h: 0.022, fill: { color: COLORS.rule },
    line: { color: COLORS.rule },
  });
}
function cite(slide, text) {
  slide.addText(text, {
    x: M, y: 5.06, w: 9.0, h: 0.3, isTextBox: true,
    fontSize: F.cite, fontFace: F.face, color: COLORS.muted, align: 'left',
  });
}
function sectionHeader(slide, text, x, y, w) {
  slide.addText(text, {
    x, y, w, h: 0.32, isTextBox: true,
    fontSize: F.sect, fontFace: F.face, color: COLORS.accent, bold: true,
  });
}
// Fresh options object every call: pptxgenjs mutates them in place.
// pptxgenjs renders bullets per RUN, not per text box: `bullet: true` in these
// parent options marked only the first run and left the rest unmarked. Lead-in
// paragraph blocks therefore carry no bullet at all (which reads better), and the
// right-hand take-away lists set bullet: true on each run individually.
const bullets = (x, y, w, h, runs, size) => ({
  x, y, w, h, isTextBox: true, fontSize: size || F.body, fontFace: F.face,
  color: COLORS.body, paraSpaceAfter: 10, valign: 'top',
});

// ───────────────────────────── 1. TITLE ────────────────────────────────────
let s = pres.addSlide({ masterName: 'DARK' });
s.addText('Predicting future tissue geometry\nfrom monocular laparoscopic video', {
  x: 0.75, y: 1.15, w: 8.6, h: 1.7, isTextBox: true,
  fontSize: 31, fontFace: F.face, color: COLORS.onDark, bold: true, valign: 'top',
});
s.addText('4D surgical scene reconstruction and deformation prediction', {
  x: 0.75, y: 2.78, w: 8.6, h: 0.35, isTextBox: true,
  fontSize: 15, fontFace: F.face, color: COLORS.onDarkMuted,
});
s.addShape(pres.shapes.RECTANGLE, {
  x: 0.75, y: 3.36, w: 1.6, h: 0.04,
  fill: { color: COLORS.accent }, line: { color: COLORS.accent },
});
s.addText('Ahmad Rashad  22i-1175   ·   Shehroz Kashif  22i-0771   ·   Humair Khan  22i-2623\n' +
          'Supervisor: Dr Akhtar Jamil   ·   FAST NUCES, Islamabad', {
  x: 0.75, y: 3.55, w: 8.6, h: 0.7, isTextBox: true,
  fontSize: 14, fontFace: F.face, color: COLORS.onDark, lineSpacingMultiple: 1.2,
});
s.addText('FYP-2 Final Presentation   ·   Project S26-017-R-GASR', {
  x: 0.75, y: 4.45, w: 8.6, h: 0.35, isTextBox: true,
  fontSize: 13, fontFace: F.face, color: COLORS.onDarkMuted,
});
s.addNotes('Theme sampler for supervisor approval. Content is real but the slide set ' +
           'is a sample of every slide type the final deck needs, not the final deck.');

// ─────────────────────────── 2. MOTIVATION ─────────────────────────────────
s = pres.addSlide({ masterName: 'CONTENT' });
actionTitle(s, 'A robot that reacts to where tissue is has already reacted to where it was');
s.addText([
  { text: 'Tissue never holds still. ', options: { bold: true, breakLine: false } },
  { text: 'It moves with breathing, with the pulse, and with every retraction.',
    options: { breakLine: true } },
  { text: 'Reconstruction is retrospective. ', options: { bold: true, breakLine: false } },
  { text: 'Surgical 4D methods rebuild frames already observed; none forecast the next one.',
    options: { breakLine: true } },
  { text: 'Latency is the clinical problem. ', options: { bold: true, breakLine: false } },
  { text: 'Guidance drawn on a surgeon’s screen must act on where tissue will be, not where it was.',
    options: { breakLine: false } },
], bullets(M, 1.35, 9.0, 2.6));
cite(s, 'Wang et al. (2022) EndoNeRF  ·  Liu et al. (2024) EndoGaussian  ·  Yang et al. (2024) Deform3DGS');

// ──────────────────── 3. RESEARCH QUESTION ────────────────────────────────
s = pres.addSlide({ masterName: 'CONTENT' });
actionTitle(s, 'Can future tissue geometry be predicted\nfrom monocular video alone?');
s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: 1.15, y: 1.5, w: 7.7, h: 1.45, rectRadius: 0.08,
  fill: { color: 'FBF2F2' }, line: { color: COLORS.accent, pt: 1.5 },
});
s.addText('Given only the frames observed so far, where will every tracked point of ' +
          'tissue be k frames from now — with no depth sensor, no stereo pair, no ' +
          'tool trajectory, and no human annotation?', {
  x: 1.35, y: 1.62, w: 7.3, h: 1.2, isTextBox: true,
  fontSize: 17, fontFace: F.face, color: COLORS.primary, align: 'center', valign: 'middle',
});
s.addText([
  { text: 'Contribution: ', options: { bold: true, breakLine: false } },
  { text: 'a conditional diffusion model that forecasts 3D tissue geometry from monocular ' +
          'endoscopy and reports calibrated uncertainty — which no compared method does.',
    options: {} },
], { x: M, y: 3.25, w: 9.0, h: 1.0, isTextBox: true,
     fontSize: F.body, fontFace: F.face, color: COLORS.body });

// ──────────────────────── 4. SECTION: METHOD ──────────────────────────────
s = pres.addSlide({ masterName: 'DARK' });
s.addText('01', { x: 0.75, y: 1.9, w: 1.2, h: 0.7, isTextBox: true,
  fontSize: 34, fontFace: F.face, color: COLORS.accent, bold: true });
s.addText('Method', { x: 0.75, y: 2.6, w: 8.0, h: 0.8, isTextBox: true,
  fontSize: 30, fontFace: F.face, color: COLORS.onDark, bold: true });

// ────────────────────────── 5. METHODS ────────────────────────────────────
s = pres.addSlide({ masterName: 'CONTENT' });
actionTitle(s, 'Monocular video becomes tracked 3D geometry,\nwhich a diffusion model forecasts forward');
sectionHeader(s, 'Pipeline', M, 1.32, 4.2);
s.addText([
  { text: 'Reconstruct — ', options: { bold: true, breakLine: false } },
  { text: '3D Gaussian scene from monocular frames', options: { breakLine: true } },
  { text: 'Track — ', options: { bold: true, breakLine: false } },
  { text: '512 tissue anchors, identity preserved over time', options: { breakLine: true } },
  { text: 'Window — ', options: { bold: true, breakLine: false } },
  { text: '32 frames of context, 80/20 split in time', options: { breakLine: true } },
  { text: 'Forecast — ', options: { bold: true, breakLine: false } },
  { text: 'conditional diffusion, horizon-conditioned', options: { breakLine: false } },
], bullets(M, 1.72, 4.2, 2.6, null, 16));
sectionHeader(s, 'Evaluation protocol', 5.3, 1.32, 4.2);
s.addText([
  { text: '10 videos, 1,298 held-out windows', options: { breakLine: true } },
  { text: 'Split temporally — later frames never seen in training', options: { breakLine: true } },
  { text: 'Every baseline retrained on identical windows, identical budget', options: { breakLine: true } },
  { text: 'Two metrics: Chamfer Distance and Mean Point Error', options: { breakLine: false } },
], bullets(5.3, 1.72, 4.2, 2.6, null, 16));
cite(s, 'Model predicts a correction to constant-velocity extrapolation, so its worst case is a baseline that already beats doing nothing.');

// ───────────────────── 6. SECTION: RESULTS ────────────────────────────────
s = pres.addSlide({ masterName: 'DARK' });
s.addText('02', { x: 0.75, y: 1.9, w: 1.2, h: 0.7, isTextBox: true,
  fontSize: 34, fontFace: F.face, color: COLORS.accent, bold: true });
s.addText('Results', { x: 0.75, y: 2.6, w: 8.0, h: 0.8, isTextBox: true,
  fontSize: 30, fontFace: F.face, color: COLORS.onDark, bold: true });

// ─────────── 7. RESULT: overall ranking (one exhibit) ─────────────────────
s = pres.addSlide({ masterName: 'CONTENT' });
actionTitle(s, 'Against ten classical and published baselines, the model ranks first');
s.addChart(pres.charts.BAR, [{
  name: 'Mean Point Error',
  labels: ['Ours — diffusion', 'MLP regressor', 'LSTM (per-point)', 'GRU (per-point)',
           'Constant velocity', 'PointLSTM (base paper)', 'Transformer', 'Zero motion'],
  values: [0.007461, 0.008053, 0.008056, 0.008087, 0.008211, 0.008392, 0.009290, 0.009395],
}], {
  x: 0.35, y: 1.3, w: 5.9, h: 3.6,
  barDir: 'bar',
  chartColors: [COLORS.accent, 'C9C4C4', 'C9C4C4', 'C9C4C4',
                'C9C4C4', 'C9C4C4', 'C9C4C4', 'C9C4C4'],
  chartArea: { fill: { color: COLORS.bg } },
  plotArea: { fill: { color: COLORS.bg } },
  catAxisLabelColor: COLORS.body, catAxisLabelFontSize: 11, catAxisLabelFontFace: F.face,
  valAxisLabelColor: COLORS.muted, valAxisLabelFontSize: 10, valAxisLabelFontFace: F.face,
  valGridLine: { color: COLORS.grid, size: 0.5 },
  catGridLine: { style: 'none' },
  showValue: false, showLegend: false,
  valAxisTitle: 'Mean Point Error  (lower is better)',
  showValAxisTitle: true, valAxisTitleColor: COLORS.muted,
  valAxisTitleFontSize: 11, valAxisTitleFontFace: F.face,
});
sectionHeader(s, 'What to take away', 6.45, 1.3, 3.1);
s.addText([
  { text: '20.6% better than assuming no motion', options: { bullet: true, breakLine: true } },
  { text: '11.1% better than PointLSTM, the base paper', options: { bullet: true, breakLine: true } },
  { text: '9.1% better than constant velocity', options: { bullet: true, breakLine: true } },
  { text: 'First on both metrics, CD and MPE', options: { bullet: true, breakLine: false } },
], bullets(6.45, 1.72, 3.1, 2.6, null, 15));
cite(s, 'Module 7 · 1,298 held-out windows across 10 videos · baselines trained 60 epochs, one protocol');

// ─────────── 8. RESULT: where the advantage is (honest) ───────────────────
s = pres.addSlide({ masterName: 'CONTENT' });
actionTitle(s, 'The advantage grows with horizon — simple extrapolation wins at one frame');
s.addChart(pres.charts.LINE, [
  { name: 'Ours — diffusion', labels: ['k = 1', 'k = 2', 'k = 3'],
    values: [0.003108, 0.007327, 0.011960] },
  { name: 'Constant velocity', labels: ['k = 1', 'k = 2', 'k = 3'],
    values: [0.002996, 0.007838, 0.013811] },
  { name: 'MLP regressor', labels: ['k = 1', 'k = 2', 'k = 3'],
    values: [0.004113, 0.008093, 0.011962] },
], {
  x: 0.35, y: 1.3, w: 5.9, h: 3.5,
  chartColors: [COLORS.accent, '8A8A90', 'C2A0A0'],
  chartArea: { fill: { color: COLORS.bg } },
  plotArea: { fill: { color: COLORS.bg } },
  lineSize: 3, lineSmooth: false,
  catAxisLabelColor: COLORS.body, catAxisLabelFontSize: 12, catAxisLabelFontFace: F.face,
  valAxisLabelColor: COLORS.muted, valAxisLabelFontSize: 10, valAxisLabelFontFace: F.face,
  valGridLine: { color: COLORS.grid, size: 0.5 },
  catGridLine: { style: 'none' },
  showLegend: true, legendPos: 'b', legendColor: COLORS.body, legendFontSize: 11,
  valAxisTitle: 'Mean Point Error', showValAxisTitle: true,
  valAxisTitleColor: COLORS.muted, valAxisTitleFontSize: 11,
});
sectionHeader(s, 'Read this honestly', 6.45, 1.3, 3.1);
s.addText([
  { text: 'k=1: constant velocity wins by 3.7%. Over 33 ms motion is near-linear.',
    options: { breakLine: true } },
  { text: 'k=2: we win by 6.5% — where the advantage actually lives.',
    options: { breakLine: true } },
  { text: 'k=3: we lead constant velocity by 13.4% as linear extrapolation breaks down.',
    options: { breakLine: false } },
], bullets(6.45, 1.72, 3.1, 2.8, null, 14));
cite(s, 'Module 7, per-horizon Mean Point Error · same held-out windows as the overall ranking');

// ─────────── 9. RESULT: reconstruction (table, exact values) ──────────────
s = pres.addSlide({ masterName: 'CONTENT' });
actionTitle(s, 'Reconstruction from monocular input matches stereo methods');
s.addTable([
  [{ text: 'Method', options: { bold: true, color: COLORS.onDark, fill: { color: COLORS.primary } } },
   { text: 'Input', options: { bold: true, color: COLORS.onDark, fill: { color: COLORS.primary } } },
   { text: 'PSNR (dB)', options: { bold: true, color: COLORS.onDark, fill: { color: COLORS.primary }, align: 'right' } },
   { text: 'SSIM', options: { bold: true, color: COLORS.onDark, fill: { color: COLORS.primary }, align: 'right' } }],
  ['LerPlane', 'stereo', { text: '29.46', options: { align: 'right' } }, { text: '—', options: { align: 'right' } }],
  ['EndoGaussian', 'stereo', { text: '30.25', options: { align: 'right' } }, { text: '—', options: { align: 'right' } }],
  ['Deform3DGS', 'stereo', { text: '30.48', options: { align: 'right' } }, { text: '—', options: { align: 'right' } }],
  [{ text: 'Ours', options: { bold: true, color: COLORS.accent } },
   { text: 'monocular', options: { bold: true, color: COLORS.accent } },
   { text: '30.10', options: { bold: true, color: COLORS.accent, align: 'right' } },
   { text: '0.9037', options: { bold: true, color: COLORS.accent, align: 'right' } }],
], {
  x: M, y: 1.5, w: 9.0, colW: [3.2, 2.0, 2.0, 1.8],
  fontSize: 15, fontFace: F.face, color: COLORS.body,
  border: { type: 'solid', color: COLORS.rule, pt: 0.5 },
  rowH: 0.42, valign: 'middle',
});
s.addText('Published figures are those each method reports on stereo sequences — different data, ' +
          'with depth supervision. Indicative, not a controlled comparison.', {
  x: M, y: 4.0, w: 9.0, h: 0.6, isTextBox: true,
  fontSize: 14, fontFace: F.face, color: COLORS.body, italic: true,
});
cite(s, 'Ruan et al. (2023); Liu et al. (2024); Yang et al. (2024) · ours: 219 frames, single sequence');

// ─────────────────── 10. DISCUSSION / LIMITATIONS ─────────────────────────
s = pres.addSlide({ masterName: 'CONTENT' });
actionTitle(s, 'The margin is real but narrow, and the cost is high');
sectionHeader(s, 'What holds', M, 1.32, 9.0);
s.addText([
  { text: 'Beats the base paper under one protocol, on both metrics.', options: { breakLine: true } },
  { text: 'Only method of the eleven that reports uncertainty — spread tracks error.', options: { breakLine: false } },
], bullets(M, 1.72, 9.0, 1.0, null, 17));
sectionHeader(s, 'What does not', M, 2.82, 9.0);
s.addText([
  { text: 'At three frames a 157k-parameter MLP matches us to 0.017% — we use 104× the parameters and 31 minutes of sampling.',
    options: { breakLine: true } },
  { text: 'Normalised units, not millimetres: nothing in the pipeline is calibrated to physical scale.',
    options: { breakLine: true } },
  { text: 'Feasibility study on reconstructed recordings — it supports no decision about a patient.',
    options: { breakLine: false } },
], bullets(M, 3.22, 9.0, 1.7, null, 17));

// ───────────────────────── 11. CONCLUSIONS ────────────────────────────────
s = pres.addSlide({ masterName: 'DARK' });
s.addText('Conclusions', { x: 0.75, y: 0.3, w: 8.6, h: 0.45, isTextBox: true,
  fontSize: 19, fontFace: F.face, color: COLORS.onDarkMuted });
s.addShape(pres.shapes.RECTANGLE, { x: 0.75, y: 0.78, w: 8.6, h: 0.035,
  fill: { color: COLORS.accent }, line: { color: COLORS.accent } });
s.addText([
  { text: '1.  Future tissue geometry is predictable from monocular video alone.  ',
    options: { bold: true, breakLine: false } },
  { text: 'No depth sensor, no stereo pair, no annotation.', options: { breakLine: true } },
  { text: '2.  It beats ten classical and published baselines under one protocol.  ',
    options: { bold: true, breakLine: false } },
  { text: '20.6% better than no motion; 11.1% better than the base paper.', options: { breakLine: true } },
  { text: '3.  The advantage grows with how far ahead you look.  ',
    options: { bold: true, breakLine: false } },
  { text: 'Linear extrapolation wins at one frame and degrades; the model does not.', options: { breakLine: true } },
  { text: '4.  It is the only compared method that says how sure it is.  ',
    options: { bold: true, breakLine: false } },
  { text: 'Sampled spread correlates with actual error.', options: { breakLine: false } },
], { x: 0.75, y: 1.0, w: 8.6, h: 3.5, isTextBox: true,
     fontSize: 16, fontFace: F.face, color: COLORS.onDark, paraSpaceAfter: 14 });
s.addText('S26-017-R-GASR  ·  FAST NUCES Islamabad  ·  live demonstrator runs offline from a USB stick', {
  x: 0.75, y: 4.72, w: 8.6, h: 0.4, isTextBox: true,
  fontSize: 13, fontFace: F.face, color: COLORS.onDarkMuted });

// ───────────────────────── 12. REFERENCES ─────────────────────────────────
s = pres.addSlide({ masterName: 'CONTENT' });
actionTitle(s, 'References');
s.addText([
  { text: 'Kerbl, B. et al. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM TOG.', options: { breakLine: true } },
  { text: 'Wang, Y. et al. (2022). Neural Rendering for Stereo 3D Reconstruction of Deformable Tissues. MICCAI.', options: { breakLine: true } },
  { text: 'Ruan, C. et al. (2023). LerPlane: Neural Representations for Fast 4D Reconstruction. MICCAI.', options: { breakLine: true } },
  { text: 'Liu, Y. et al. (2024). EndoGaussian: Real-time Gaussian Splatting for Dynamic Endoscopic Scenes.', options: { breakLine: true } },
  { text: 'Yang, S. et al. (2024). Deform3DGS: Flexible Deformation for Fast Surgical Scene Reconstruction.', options: { breakLine: true } },
  { text: 'Lu, F. et al. (2022). MoNet: Motion-Based Point Cloud Prediction Network. IEEE T-ITS 23(8), 13794–13804.', options: { breakLine: true } },
  { text: 'Luo, Z. et al. (2023). PCPNet: Efficient Semantic-Enhanced Transformer for Point Cloud Prediction.', options: { breakLine: true } },
  { text: 'Fan, H. & Yang, Y. (2019). PointRNN: Point Recurrent Neural Network for Moving Point Cloud Processing.', options: { breakLine: false } },
], { x: M, y: 1.35, w: 9.0, h: 3.6, isTextBox: true,
     fontSize: 12.5, fontFace: F.face, color: COLORS.body, paraSpaceAfter: 8 });

pres.writeFile({ fileName: OUT })
  .then(() => console.log('  wrote ' + OUT))
  .catch(e => { console.error(e); process.exit(1); });
