import { useCallback, useEffect, useMemo, useState } from 'react';
import { horizonLabel, measuredCamera } from './core';
import type { Payload } from './types';
import { SCENES, STR } from './scenes';
import { Stage, overlayGain } from './Stage';
import { Readout } from './Readout';

export function App({ data, placeholder }: { data: Payload; placeholder: boolean }) {
  // An older or hand-edited payload can lack horizons entirely. Reading it unguarded
  // throws before the first render, and a blank page with nothing on it to explain why
  // is the worst failure available in a room.
  const horizons = data.horizons?.length ? data.horizons : [1];

  const [si, setSi] = useState(0);
  const [frame, setFrame] = useState(0);
  const [k, setK] = useState(() => horizons.includes(3) ? 3 : horizons[0]);
  const [playing, setPlaying] = useState(false);
  const [view, setView] = useState({ yaw: 0.6, pitch: 0.25, zoom: 1 });

  const scene = SCENES[si];

  const fill = useCallback((t: string) => String(t)
    .replace(/\{horizon\}/g, horizonLabel(k, null))
    .replace(/\{points\}/g, String(data.n_points))
    .replace(/\{gain\}/g, String(overlayGain(data, k))), [data, k]);

  const go = useCallback((n: number) => {
    const next = (n + SCENES.length) % SCENES.length;
    setSi(next);
    const h = SCENES[next].horizon;
    if (h !== null && horizons.includes(h)) setK(h);
  }, [horizons]);

  const pickK = useCallback((want: number) => {
    if (horizons.includes(want)) setK(want);
  }, [horizons]);

  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => setFrame(f => (f + 1) % data.n_frames), 110);
    return () => clearInterval(id);
  }, [playing, data.n_frames]);

  // Controls stay live in every scene, so an interruption can be answered inside the
  // scene on screen rather than dead-ending.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'ArrowRight') { go(si + 1); e.preventDefault(); }
      else if (e.key === 'ArrowLeft') { go(si - 1); e.preventDefault(); }
      else if (e.key === ' ') { setPlaying(p => !p); e.preventDefault(); }
      else if (e.key === '1') pickK(1);
      else if (e.key === '3') pickK(3);
      else if (e.key === 'r' || e.key === 'R') setView({ yaw: 0.6, pitch: 0.25, zoom: 1 });
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [go, si, pickK]);

  const notes = useMemo(() => {
    const out: string[] = [];
    const f = data.frames[frame];
    if ((data.degenerate_horizons ?? []).map(String).includes(String(k))) {
      out.push(fill(STR.degenerate));
    }
    if (!placeholder && scene.layers.includes('overlay') && f?.px_ctx) {
      // An overlay drawn from assumed intrinsics is plausible, not measured, and must
      // not reach a projector presented as the latter.
      if (!measuredCamera(data.camera)) out.push(STR.uncalibrated);
      if (overlayGain(data, k) > 1) out.push(fill(STR.gainNote));
    }
    return out;
  }, [data, frame, k, scene, placeholder, fill]);

  // An old demo_data.js can be missing a horizon. Draw what is there and name what is
  // not, rather than throwing mid-render.
  const missing = useMemo(() => {
    const f = data.frames[frame];
    if (!f) return '';
    const gone: string[] = [];
    const absent = (key: string) => !(f[key] as number[] | undefined)?.length;
    if (scene.layers.includes('gt') && absent(`gt${k}`)) gone.push(STR.GAP_GT);
    if (scene.layers.includes('pred') && absent(`pred${k}`)) gone.push(STR.GAP_PRED);
    if (scene.layers.includes('cv') && absent(`cv${k}`)) gone.push(STR.GAP_CV);
    if (scene.layers.includes('sigma') && absent(`sigma${k}`)) gone.push(STR.GAP_SIGMA);
    if (!gone.length) return '';
    const list = gone.length > 1
      ? `${gone.slice(0, -1).join(', ')} or ${gone[gone.length - 1]}` : gone[0];
    return ` ${STR.missingPrefix}${list}${fill(STR.missingSuffix)}`;
  }, [data, frame, k, scene, fill]);

  const btn = 'rounded border border-edge bg-[#1E1E24] px-4 py-2 text-[15px] ' +
              'hover:border-blood cursor-pointer';
  const on = 'border-blood bg-crimson';

  return (
    <div className="flex h-full flex-col">
      <header className="flex flex-wrap items-baseline gap-4 border-b border-edge
                         bg-panel px-6 py-3">
        <h1 className="m-0 text-[19px] font-semibold uppercase tracking-[0.14em]">
          GASR<span className="text-blood">.</span> predicting tissue motion
        </h1>
        <div className="text-sm text-dim">
          {placeholder ? STR.metaPlaceholder
            : `${data.video} · ${data.n_points} points · ${data.n_frames} held-out ` +
              `moments · checkpoint epoch ${data.checkpoint_epoch}`}
        </div>
        <div className={'ml-auto rounded px-3 py-1 text-[13px] font-bold tracking-[0.11em] ' +
          (placeholder ? 'border border-[#6E6420] bg-[#3A3A10] text-[#F2E27A]'
                       : 'bg-crimson text-bone')}>
          {placeholder ? 'PLACEHOLDER DATA' : 'MEASURED'}
        </div>
      </header>

      <main className="grid min-h-0 flex-1 grid-cols-[minmax(380px,34%)_1fr]">
        <section className="overflow-y-auto border-r border-edge px-7 py-6">
          <div className="text-sm font-bold tracking-[0.17em] text-blood">
            SCENE {si + 1} OF {SCENES.length}
          </div>
          <h2 className="mb-4 mt-2 text-[33px] font-bold leading-tight">{scene.title}</h2>
          <p className="mb-5 text-lg text-[#D8D8DE]">{fill(scene.body)}{missing}</p>

          {notes.map((n, i) => (
            <div key={i} className="mb-4 border-l-[3px] border-blood bg-blood/10
                                    px-4 py-3 text-base text-[#F0CFCF]">{n}</div>
          ))}

          {(scene.id === 'baselines' || scene.id === 'standing') && (
            <Readout data={data} k={k} placeholder={placeholder} fill={fill} />
          )}
        </section>

        <Stage
          data={data} scene={scene} k={k} frame={frame} placeholder={placeholder}
          yaw={view.yaw} pitch={view.pitch} zoom={view.zoom}
          onOrbit={(dy, dp) => setView(v => ({
            ...v, yaw: v.yaw + dy,
            pitch: Math.max(-1.5, Math.min(1.5, v.pitch + dp)),
          }))}
          onZoom={fct => setView(v => ({
            ...v, zoom: Math.max(0.35, Math.min(4, v.zoom * fct)),
          }))}
        />
      </main>

      <footer className="flex items-center gap-2 border-t border-edge bg-panel px-6 py-3">
        <button className={btn} onClick={() => go(si - 1)}>← back</button>
        <button className={btn} onClick={() => go(si + 1)}>next →</button>
        <button className={btn} onClick={() => setPlaying(p => !p)}>
          {playing ? 'pause' : 'play'}
        </button>
        <button className={`${btn} ${k === 1 ? on : ''}`} onClick={() => pickK(1)}>1 frame</button>
        <button className={`${btn} ${k === 3 ? on : ''}`} onClick={() => pickK(3)}>3 frames</button>
        <button className={btn}
                onClick={() => setView({ yaw: 0.6, pitch: 0.25, zoom: 1 })}>reset view</button>
        <div className="ml-auto text-[13px] text-dim">
          ← → scene · space play · 1 / 3 horizon · R view
        </div>
      </footer>
    </div>
  );
}
