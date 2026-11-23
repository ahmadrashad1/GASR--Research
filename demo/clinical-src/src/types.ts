import { METHOD_LABELS } from './core';
export type { Payload, Frame, Camera, Timing, MeanError } from './core';
export type { Scene, Layer } from './scenes';

// One fixed colour per series across all seven scenes, so the panel learns the
// legend once. These are the palette the research viewer already established.
export const COLOUR = {
  gt:    '#F5F5F7',
  pred:  '#D62828',
  cv:    '#C98276',
  still: '#75757E',
  sigma: '#5C0A0A',
} as const;

export const SERIES_LABEL = {
  gt:    'What actually happened',
  pred:  METHOD_LABELS.ours,
  cv:    METHOD_LABELS.cv,
  still: METHOD_LABELS.still,
} as const;
