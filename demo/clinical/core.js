// Pure logic for the clinical demonstrator. No DOM, no canvas, no data.
//
// Everything comparative the app says out loud is computed here, from the mean_error
// block of whatever demo_data.js was loaded. Nothing in this file, and nothing in the
// page that uses it, states who wins. The repo's only other MPE-style comparison
// (paper/measured.json -> comparison) is from the superseded epoch-60 W=4 checkpoint and
// was read off a chart by eye; it must not shape what this app claims. Running the real
// export through these functions is the only thing that decides the wording on screen.
//
// The same file is loaded two ways: as a classic <script> by index.html, because file://
// forbids both fetch() and <script type="module">, and as a CommonJS module by
// node --test. The footer at the bottom is what makes both work.

const GASR = {};

GASR.METHOD_LABELS = {
  ours:  'GASR (ours)',
  cv:    'Straight-line motion',
  still: 'No movement',
};

// A finite number, or null. Everything below routes through this rather than trusting
// the payload: a missing horizon, a null from a partial export, or a NaN out of a
// degenerate calibration must all come back as "unknown" rather than as a number the
// page would render with total confidence.
function num(v) {
  return (typeof v === 'number' && Number.isFinite(v)) ? v : null;
}

GASR.unflat = function (flat, n) {
  const out = [];
  for (let i = 0; i < n; i++) out.push([flat[i * 3], flat[i * 3 + 1], flat[i * 3 + 2]]);
  return out;
};

function meanDistance(a, b) {
  if (!a || !b || !a.length || a.length !== b.length) return null;
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const dx = a[i][0] - b[i][0], dy = a[i][1] - b[i][1], dz = a[i][2] - b[i][2];
    sum += Math.sqrt(dx * dx + dy * dy + dz * dz);
  }
  return sum / a.length;
}

GASR.meanPointError = meanDistance;

// How far the tissue actually went over the horizon: the distance from the last observed
// frame to what really happened. This is the denominator the app leads with.
GASR.motionTravelled = meanDistance;

// Error as a percentage of the distance the tissue actually travelled. A skill score:
// 100 % means the prediction was no better than not predicting, under 100 % beats
// standing still, over 100 % is worse than standing still.
//
// Returns null when nothing moved. A static window makes the denominator zero, and
// Infinity rendered into a sentence reads as a catastrophic error rather than as an
// undefined one.
GASR.skillPercent = function (err, motion) {
  const e = num(err), m = num(motion);
  if (e === null || m === null || m <= 0) return null;
  return e / m * 100;
};

// Error as a percentage of the reconstructed tissue's own span. Secondary context only.
// Every method in this project, including assuming the tissue never moves, scores under
// half a percent against this denominator, so it distinguishes nothing and flatters
// everything. The app shows it once, labelled, and never leads with it.
GASR.spanPercent = function (err, span) {
  const e = num(err), s = num(span);
  if (e === null || s === null || s <= 0) return null;
  return e / s * 100;
};

GASR.rankAtHorizon = function (meanError, k) {
  if (!meanError) return null;
  const row = meanError[String(k)];
  if (!row) return null;
  const out = [];
  for (const key of Object.keys(GASR.METHOD_LABELS)) {
    const e = num(row[key]);
    if (e !== null) out.push({ key: key, err: e });
  }
  if (!out.length) return null;
  return out.sort((x, y) => x.err - y.err);
};

// Two errors this close are the same number, not a ranking. The case is not
// hypothetical: where the calibration fit returns alpha = 0 the exporter states that
// pred{k} is EXACTLY cv{k}, and its own run has printed
// "k=1  ours 0.47478   const-vel 0.47478". A stable sort over METHOD_LABELS puts ours
// first on equal keys, so without this the page announces the model as the closest
// prediction at a horizon where its prediction IS the straight-line baseline -- a
// verdict computed at runtime and still wrong, which is the one outcome the
// no-authored-verdict rule exists to prevent.
//
// The threshold is numerical indistinguishability only. A wider one would be a
// significance threshold, and there is nothing in this pipeline that would justify a
// particular value for it.
const TIE_REL = 1e-6;

GASR.verdict = function (meanError, k) {
  const rank = GASR.rankAtHorizon(meanError, k);
  if (!rank) return null;

  const winner = rank[0];
  const label = GASR.METHOD_LABELS[winner.key];
  const ours = rank.find(r => r.key === 'ours');
  const still = rank.find(r => r.key === 'still');

  const scale = Math.max.apply(null, rank.map(r => Math.abs(r.err))) || 1;
  const level = rank.filter(r => Math.abs(r.err - winner.err) <= TIE_REL * scale);
  const tied = level.length > 1;

  const oursBeatsAll = winner.key === 'ours' && !tied;
  const oursBeatsStatic = (ours && still) ? ours.err < still.err : null;

  let text;
  if (tied) {
    const names = level.map(r => GASR.METHOD_LABELS[r.key]);
    text = `${names.slice(0, -1).join(', ')} and ${names[names.length - 1]} are ` +
           `numerically indistinguishable ${GASR.horizonLabel(k, null)}: to the ` +
           `precision reported here they are the same prediction.`;
  } else if (oursBeatsAll) {
    text = `${label} is the closest prediction ${GASR.horizonLabel(k, null)}.`;
  } else {
    // Say it plainly. Standing in front of an examination panel claiming a win the
    // number on the same screen contradicts is a worse outcome than reading out a loss.
    text = `${label} is closest ${GASR.horizonLabel(k, null)}; ` +
           `${GASR.METHOD_LABELS.ours} is behind it here. ` +
           `The model is the only method on screen that also reports how uncertain it ` +
           `is at each point.`;
  }

  return {
    winner: winner.key,
    winnerLabel: label,
    text: text,
    tied: tied,
    tiedWith: level.map(r => r.key),
    oursBeatsAll: oursBeatsAll,
    oursBeatsStatic: oursBeatsStatic,
  };
};

// The horizon in frames, which is exact, plus elapsed time as a RANGE when and only when
// the export recovered usable timing.
//
// Kept frames are not evenly spaced: Module 1 takes every second raw frame and then drops
// blurred ones, so consecutive frames can be 2 or 10 raw frames apart. A single averaged
// millisecond figure over origins whose real gaps differ fivefold is the same class of
// fabrication as stating the error in millimetres, and it is the kind of number a panel
// asks a second question about.
GASR.horizonLabel = function (k, timing) {
  const frames = `${k} frame${k === 1 ? '' : 's'} ahead`;
  if (!timing || timing.timing_ok !== true) return frames;
  const fps = num(timing.fps), lo = num(timing.gap_min), hi = num(timing.gap_max);
  if (fps === null || lo === null || hi === null || fps <= 0 || lo <= 0 || hi <= 0) {
    return frames;
  }
  const a = (k * lo / fps).toFixed(1), b = (k * hi / fps).toFixed(1);
  return `${frames} — between about ${a} and ${b} s at this sequence's recovered frame rate`;
};

if (typeof module !== 'undefined' && module.exports) module.exports = GASR;
if (typeof window !== 'undefined') window.GASR = GASR;
