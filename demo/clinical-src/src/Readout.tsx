import { METHOD_LABELS, horizonLabel, rankAtHorizon, skillPercent, spanPercent,
         verdict } from './core';
import type { Payload } from './types';
import { COLOUR } from './types';
import { STR } from './scenes';

// Every comparative sentence on this panel is computed from the loaded mean_error.
// None of it is written into the page, so the same readout that shows a win shows a
// loss. That is the point: the alternative is asserting something on a projector that
// the number underneath it contradicts.
export function Readout({ data, k, placeholder, fill }:
                        { data: Payload; k: number; placeholder: boolean;
                          fill: (s: string) => string }) {
  const v = verdict(data.mean_error, k);
  const rank = rankAtHorizon(data.mean_error, k);

  if (!v || !rank) {
    return (
      <div className="border-t border-edge pt-4 text-lg">
        {placeholder ? STR.noMeasurePlaceholder : fill(STR.noMeasureReal)}
      </div>
    );
  }

  const motion = data.motion?.[String(k)] ?? null;
  const pct = (x: number | null, dp: number) => x === null ? '—' : `${x.toFixed(dp)}%`;

  return (
    <div className="border-t border-edge pt-4">
      <p className="mb-3 text-lg">{v.text}</p>
      <table className="w-full text-base">
        <thead>
          <tr className="text-xs uppercase tracking-wide text-dim">
            <th className="py-1 pr-2 text-left font-semibold">{horizonLabel(k, data.timing)}</th>
            <th className="py-1 pr-2 text-right font-semibold">of distance moved</th>
            <th className="py-1 text-right font-semibold">of tissue span</th>
          </tr>
        </thead>
        <tbody>
          {rank.map(r => (
            <tr key={r.key}>
              <td className="py-1 pr-2">
                <i className="mr-2 inline-block h-3 w-3 rounded-sm"
                   style={{ background: r.key === 'ours' ? COLOUR.pred
                                      : r.key === 'cv' ? COLOUR.cv : COLOUR.still }} />
                {METHOD_LABELS[r.key]}
              </td>
              <td className="py-1 pr-2 text-right tabular-nums">
                {pct(skillPercent(r.err, motion), 0)}
              </td>
              <td className="py-1 text-right tabular-nums">
                {pct(spanPercent(r.err, data.span), 2)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-3 text-sm text-dim">{fill(STR.scope)}</p>
    </div>
  );
}
