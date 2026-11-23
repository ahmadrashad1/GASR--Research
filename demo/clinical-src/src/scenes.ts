// Every sentence a clinician reads lives in this file and nowhere else, so the whole
// script of the demonstration can be reviewed in one sitting without reading any code.
//
// Two rules hold for every body below, and tests enforce both:
//
//   1. No comparative claim. Which method came closest is computed at runtime from the
//      loaded mean_error and rendered beneath the scene. The same readout that shows a
//      win shows a loss, which is the point: the alternative is asserting something on
//      a projector that the number on the same screen contradicts.
//
//   2. No absolute physical unit. The trajectories live in a per-video normalised space
//      and nothing in this pipeline is calibrated to millimetres. Horizons are stated
//      in frames, which is exact.
//
// {horizon}, {points} and {gain} are substituted at render time.

export type Layer =
  | 'frame' | 'overlay' | 'ctx' | 'gt' | 'pred' | 'cv' | 'still' | 'sigma';

export interface Scene {
  id: string;
  title: string;
  body: string;
  layers: Layer[];
  horizon: number | null;
}

export const SCENES: Scene[] = [
  {
    id: 'sees',
    title: 'What the endoscope sees',
    body:
      'This is real laparoscopic footage from one of the reconstructed sequences. ' +
      'Nothing in it is still — the tissue moves with breathing, with the pulse, and ' +
      'with every retraction. A system that responds to where the tissue is has ' +
      'already responded to where it was.',
    layers: ['frame'],
    horizon: null,
  },
  {
    id: 'reconstructs',
    title: 'What the system reconstructs',
    body:
      'From that video alone — one camera, no depth sensor, no stereo pair — the ' +
      'pipeline rebuilds the scene as a moving three-dimensional surface and follows ' +
      'the same patches of tissue through it. It tracks 512 points per frame; this ' +
      'view draws an evenly spaced {points} of them, because a browser canvas cannot ' +
      'usefully show more.',
    layers: ['frame', 'ctx'],
    horizon: null,
  },
  {
    id: 'why',
    title: 'Why prediction matters',
    body:
      'Everything so far describes motion that has already happened. A robot holding ' +
      'a retractor, or guidance drawn onto the surgeon’s screen, has to act on where ' +
      'the tissue will be by the time it moves, not on where it was when the camera ' +
      'last looked. Closing that gap is what the rest of this demonstration is about.',
    layers: ['ctx', 'gt'],
    horizon: null,
  },
  {
    id: 'prediction',
    title: 'The prediction',
    body:
      'Given only the frames before this moment, the model proposes where every ' +
      'tracked point will be {horizon}. White is what actually happened; red is the ' +
      'model’s answer. These are held-out frames — this sequence’s later portion was ' +
      'never used to train the model.',
    layers: ['ctx', 'gt', 'pred', 'overlay'],
    horizon: 3,
  },
  {
    id: 'baselines',
    title: 'Against the obvious guesses',
    body:
      'Two simpler answers are always available: assume the tissue carries on exactly ' +
      'as it was moving, or assume it stays where it is. Both are reasonable, and over ' +
      'a frame or two both are hard to improve on. The figures below are computed from ' +
      'this sequence as you watch, so they report whichever of the three came closest ' +
      'here rather than whichever we would prefer.',
    layers: ['ctx', 'gt', 'pred', 'cv', 'still', 'overlay'],
    horizon: null,
  },
  {
    id: 'uncertainty',
    title: 'How sure it is',
    body:
      'The model does not return a single answer. It samples several possible futures, ' +
      'and the halo around each point shows how far apart they were there. A wide halo ' +
      'is the system saying it does not know — which is information a surgeon can act ' +
      'on, and which neither of the simpler guesses can express at all.',
    layers: ['ctx', 'pred', 'sigma', 'overlay'],
    horizon: null,
  },
  {
    id: 'standing',
    title: 'Where it stands',
    body:
      'These figures are the average distance between a prediction and what actually ' +
      'happened, over the exported points of this one sequence. They are in the ' +
      'reconstruction’s own normalised units, not millimetres: no part of this ' +
      'pipeline is calibrated to a physical scale, so a distance in this demonstration ' +
      'cannot be converted into one. This is a feasibility study on reconstructed ' +
      'recordings, not a clinical evaluation, and it does not support any decision ' +
      'about a patient.',
    layers: ['ctx', 'gt', 'pred'],
    horizon: null,
  },
];

export const STR: Record<string, string> = {
  noImageTitle: 'Endoscopic imagery comes from the export',
  noImageBody:
    'No frames were found beside this page, so only the video stills are missing.',
  noImageBodyWithGeometry:
    'No frames were found beside this page. The geometry is complete; only the video ' +
    'stills are missing.',

  noMeasurePlaceholder:
    'Placeholder geometry carries no measurement, so nothing is compared here.',
  noMeasureReal:
    'This export carries no measurement at {horizon}, so nothing is compared here.',

  degenerate:
    'At {horizon} the calibration fit gave the model no weight, so its prediction here ' +
    'is the straight-line one, point for point. The sampled spread is zero and is not ' +
    'drawn: there is no uncertainty to report, rather than none found.',

  uncalibrated:
    'The points are placed using the same nominal 75° field of view the reconstruction ' +
    'itself was built with. That figure is the textbook value for a standard ' +
    'laparoscope, not a measurement of this one, so the overlay is consistent with the ' +
    'model rather than calibrated to the real scope. The geometry and every figure ' +
    'beside it are unaffected.',

  gainNote:
    'Motion on the video is drawn ×{gain} life size. Over this many frames the tissue ' +
    'moves about a pixel, and the three answers differ by less than that, so at true ' +
    'scale they would all sit inside one dot. The exaggeration is applied equally to ' +
    'every series, so what you are comparing is unchanged.',

  legendGt: 'What actually happened',
  legendSigma: 'Spread across sampled futures',

  metaPlaceholder: 'invented geometry for rehearsal — no measurement, no imagery',

  scope:
    'The first column is the error as a share of how far the tissue actually travelled: ' +
    'at 100% a prediction is no better than not predicting. The second is the same ' +
    'error against the width of the reconstructed tissue, for scale only — every method ' +
    'here looks small against it. Averaged over {points} points of this one sequence, ' +
    'in normalised units.',

  missingPrefix: 'This export has no ',
  missingSuffix: ' at {horizon}.',
  GAP_GT: 'recorded outcome',
  GAP_PRED: 'prediction',
  GAP_CV: 'straight-line comparison',
  GAP_SIGMA: 'sampled spread',
};
