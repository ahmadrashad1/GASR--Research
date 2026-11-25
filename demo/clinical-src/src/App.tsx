import { useCallback, useEffect, useMemo, useState } from 'react';
import { horizonLabel, measuredCamera } from './core';
import type { Payload } from './types';
import { SCENES, STR } from './scenes';
import { Stage, overlayGain } from './Stage';
import { Readout } from './Readout';
import { SignIn } from './SignIn';
import { liveMismatch, liveSkill, mergeLiveRow, needsSignIn, predictLive, probeLive,
         signOut, type LiveInfo, type LiveRow } from './live';

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

  // Live inference. All four of these stay null/false unless a local server answered,
  // which it can only do when this page was served over http by serve/server.py.
  const [live, setLive] = useState<LiveInfo | null>(null);
  const [liveRow, setLiveRow] = useState<LiveRow | null>(null);
  const [liveBusy, setLiveBusy] = useState(false);
  const [liveError, setLiveError] = useState<string | null>(null);
  const [liveOrigin, setLiveOrigin] = useState<number | null>(null);
  const [liveRefused, setLiveRefused] = useState<string | null>(null);

  // Sign-in. `gate` is set only when a server answered and asked for one, which can
  // never happen on file:// because no request is made there at all.
  const [gate, setGate] = useState<LiveInfo | null>(null);
  const [doctor, setDoctor] = useState<string | null>(null);
  const [past, setPast] = useState(false);

  const scene = SCENES[si];

  // One place decides what a health reply means, because sign-in produces a second
  // reply that has to be read by exactly the same rules as the first.
  const applyInfo = useCallback((info: LiveInfo | null) => {
    if (!info) return;
    if (needsSignIn(info)) { setGate(info); return; }
    setGate(null);
    setDoctor(info.doctor ?? null);
    if (info.live !== true) return;
    // A server serving a different video, or a different point count, must not be
    // allowed to draw over this payload. Refuse, and say which.
    const why = liveMismatch(info, data);
    if (why) { setLiveRefused(why); return; }
    setLive(info);
  }, [data]);

  useEffect(() => {
    let on = true;
    probeLive().then(info => { if (on) applyInfo(info); });
    return () => { on = false; };
  }, [data, applyInfo]);

  const onSignedIn = useCallback(async () => {
    // Health said almost nothing while signed out. Ask again now that it will answer.
    applyInfo(await probeLive());
  }, [applyInfo]);

  const onSignOut = useCallback(async () => {
    await signOut();
    setLive(null); setLiveRow(null); setDoctor(null); setPast(false);
    applyInfo(await probeLive());
  }, [applyInfo]);

  const exportedOrigin = data.frames[frame]?.t ?? 0;
  const askOrigin = liveOrigin ?? exportedOrigin;

  const fill = useCallback((t: string) => String(t)
    .replace(/\{horizon\}/g, horizonLabel(k, null))
    .replace(/\{points\}/g, String(data.n_points))
    .replace(/\{gain\}/g, String(overlayGain(data, k)))
    .replace(/\{steps\}/g, String(live?.steps ?? ''))
    .replace(/\{samples\}/g, String(live?.samples ?? ''))
    .replace(/\{seconds\}/g, String(liveRow?.seconds ?? ''))
    .replace(/\{lo\}/g, String(live?.origin_min ?? ''))
    .replace(/\{hi\}/g, String(live?.origin_max ?? '')), [data, k, live, liveRow]);

  const go = useCallback((n: number) => {
    const next = (n + SCENES.length) % SCENES.length;
    setSi(next);
    const h = SCENES[next].horizon;
    if (h !== null && horizons.includes(h)) setK(h);
  }, [horizons]);

  const pickK = useCallback((want: number) => {
    if (horizons.includes(want)) setK(want);
  }, [horizons]);

  // A live row is a prediction for ONE origin at ONE horizon. The moment either
  // changes it is no longer a description of what is on screen, so it is dropped
  // rather than left to be read as current.
  useEffect(() => {
    setLiveRow(null);
    setLiveError(null);
    setLiveOrigin(null);
  }, [k, frame]);

  const compute = useCallback(async (wanted: number) => {
    if (!live || liveBusy) return;
    const lo = live.origin_min ?? wanted;
    const hi = live.origin_max ?? wanted;
    const origin = Math.max(lo, Math.min(hi, wanted));
    setLiveBusy(true);
    setLiveError(null);
    try {
      // The uncertainty halo costs four more stochastic trajectories, so it is only
      // asked for in the scene that draws it.
      const row = await predictLive(origin, k,
        { hypotheses: scene.layers.includes('sigma') ? 4 : 0 });
      setLiveRow(row);
      setLiveOrigin(row.origin);
    } catch (err) {
      // The exported payload is still in `data` and still on screen. A failed live
      // request must cost nothing but a line of explanation.
      setLiveError(err instanceof Error ? err.message : String(err));
      setLiveRow(null);
    } finally {
      setLiveBusy(false);
    }
  }, [live, liveBusy, k, scene]);

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
      else if (live && (e.key === 'l' || e.key === 'L')) { void compute(askOrigin); }
      else if (live && e.key === '[') { void compute(askOrigin - 1); }
      else if (live && e.key === ']') { void compute(askOrigin + 1); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [go, si, pickK, live, compute, askOrigin]);

  // What the stage and the readout actually render. A live row replaces this frame's
  // arrays at this horizon and nothing else -- mean_error, motion and span stay as
  // exported, because the readout's sentences are about the whole sequence.
  const shown = useMemo(
    () => (liveRow && liveRow.k === k ? mergeLiveRow(data, frame, k, liveRow) : data),
    [data, frame, k, liveRow]);

  const isLive = Boolean(liveRow && liveRow.k === k);

  const notes = useMemo(() => {
    const out: string[] = [];
    const f = shown.frames[frame];
    if ((shown.degenerate_horizons ?? []).map(String).includes(String(k))) {
      out.push(fill(STR.degenerate));
    }
    if (!placeholder && scene.layers.includes('overlay') && f?.px_ctx) {
      // An overlay drawn from assumed intrinsics is plausible, not measured, and must
      // not reach a projector presented as the latter.
      if (!measuredCamera(shown.camera)) out.push(STR.uncalibrated);
      if (overlayGain(data, k) > 1) out.push(fill(STR.gainNote));
    }
    return out;
  }, [shown, data, frame, k, scene, placeholder, fill]);

  // An old demo_data.js can be missing a horizon. Draw what is there and name what is
  // not, rather than throwing mid-render.
  const missing = useMemo(() => {
    const f = shown.frames[frame];
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
  }, [shown, frame, k, scene, fill]);

  const btn = 'rounded border border-rule bg-swab px-4 py-2 text-[15px] text-ink ' +
              'hover:border-signal cursor-pointer';
  const on = 'border-signal bg-oxblood text-drape';

  const badge = placeholder ? 'PLACEHOLDER DATA'
              : liveBusy ? 'COMPUTING…'
              : isLive ? 'LIVE' : 'MEASURED';
  const badgeStyle = placeholder
    ? 'border border-[#C9A227] bg-[#FBF3DC] text-[#6B4E00]'
    : liveBusy ? 'border border-[#C9A227] bg-[#FBF3DC] text-[#6B4E00]'
    : isLive ? 'bg-[#0E5A3A] text-drape' : 'bg-oxblood text-drape';

  // A server asked for a sign-in and nobody has signed in or chosen to go past it.
  // Unreachable on file://, where no request is ever made.
  if (gate && !past) {
    return (
      <div className="flex h-full flex-col">
        <header className="flex flex-wrap items-baseline gap-4 border-b border-rule
                           bg-swab px-6 py-3">
          <h1 className="m-0 font-display text-[21px] font-bold tracking-[0.01em]">
            GASR<span className="text-signal">.</span> predicting tissue motion
          </h1>
        </header>
        <main className="min-h-0 flex-1">
          <SignIn onSignedIn={onSignedIn} onSkip={() => setPast(true)} />
        </main>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col">
      <header className="flex flex-wrap items-baseline gap-4 border-b border-rule
                         bg-swab px-6 py-3">
        <h1 className="m-0 font-display text-[21px] font-bold tracking-[0.01em]">
          GASR<span className="text-signal">.</span> predicting tissue motion
        </h1>
        <div className="text-sm text-serosa">
          {placeholder ? STR.metaPlaceholder
            : `${data.video} · ${data.n_points} points · ${data.n_frames} held-out ` +
              `moments · checkpoint epoch ${data.checkpoint_epoch}` +
              (isLive ? ` · origin ${askOrigin} computed here` : '')}
        </div>
        <div className={'ml-auto rounded px-3 py-1 text-[13px] font-bold ' +
                        'tracking-[0.11em] ' + badgeStyle}>
          {badge}
        </div>
      </header>

      {/* grid-rows-[minmax(0,1fr)] is load-bearing. A grid's implicit row is `auto`,
          which sizes to its content, so without it the row grew past this element's
          own height and the 3D canvas spilled down over the footer: the canvas is
          sized h-full, that percentage resolved against an auto-height parent, and it
          fell back to the canvas's intrinsic height -- which paint() had just set from
          the measured box. The loop settled at 1200px inside a 599px window at 720p.
          minmax(0,1fr) pins the row to the available height and lets it shrink. */}
      <main className="grid min-h-0 flex-1 grid-cols-[minmax(380px,34%)_1fr]
                       grid-rows-[minmax(0,1fr)]">
        {/* `narration` is the hook for the short-viewport rules in index.css. This
            column holds more than a 720p projector has room for, so below 900px of
            height it tightens rather than silently scrolling four of seven scenes. */}
        <section className="narration overflow-y-auto border-r border-rule px-7 py-6">
          <div className="mb-3 flex items-center gap-3">
            <div className="rail" role="presentation">
              {SCENES.map((_, i) => <i key={i} data-on={i <= si ? '1' : '0'} />)}
            </div>
            <span className="text-sm text-serosa">
              Scene {si + 1} of {SCENES.length}
            </span>
          </div>
          <h2 className="mb-4 mt-2 font-display text-[33px] font-bold leading-[1.15] text-ink">{scene.title}</h2>
          <p className="mb-5 text-lg text-ink">{fill(scene.body)}{missing}</p>

          {notes.map((n, i) => (
            <div key={i} className="caveat mb-4 border-l-[3px] border-signal bg-signal/8
                                    px-4 py-3 text-base text-oxblood">{n}</div>
          ))}

          {/* Ordered after the caveats here and before them on a short viewport -- see
              the `order` rules in index.css. At 720p the caveats pushed this table far
              enough down that only its first row, our own, was above the fold: the
              page would have shown "GASR 82%" with the two baselines it is being
              compared against hidden. A comparative claim with the comparison out of
              sight is the one thing this demonstrator must not do. The caveats
              describe how the overlay is drawn; they lose nothing by following. */}
          <div className="readout-slot">
            {(scene.id === 'baselines' || scene.id === 'standing') && (
              <Readout data={shown} k={k} placeholder={placeholder} fill={fill} />
            )}
          </div>

          <div className="live-slot">
            {live && !placeholder && (
              <LivePanel info={live} row={isLive ? liveRow : null} busy={liveBusy}
                         error={liveError} origin={askOrigin} fill={fill} />
            )}

            {liveRefused && !placeholder && (
              <div className="mt-6 border-l-[3px] border-signal bg-signal/8 px-4 py-3
                              text-base text-oxblood">
                Live inference is available but was refused: {liveRefused}. The exported
                measurement on this page is unaffected.
              </div>
            )}
          </div>
        </section>

        <Stage
          data={shown} scene={scene} k={k} frame={frame} placeholder={placeholder}
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

      <footer className="flex items-center gap-2 border-t border-rule bg-swab px-6 py-3">
        <button className={btn} onClick={() => go(si - 1)}>← back</button>
        <button className={btn} onClick={() => go(si + 1)}>next →</button>
        <button className={btn} onClick={() => setPlaying(p => !p)}>
          {playing ? 'pause' : 'play'}
        </button>
        <button className={`${btn} ${k === 1 ? on : ''}`} onClick={() => pickK(1)}>1 frame</button>
        <button className={`${btn} ${k === 3 ? on : ''}`} onClick={() => pickK(3)}>3 frames</button>
        <button className={btn}
                onClick={() => setView({ yaw: 0.6, pitch: 0.25, zoom: 1 })}>reset view</button>
        {live && (
          <>
            <button className={`${btn} ${isLive ? on : ''}`} disabled={liveBusy}
                    onClick={() => void compute(askOrigin)}>
              {liveBusy ? 'computing…' : 'run the model now'}
            </button>
            <button className={btn} disabled={liveBusy}
                    onClick={() => void compute(askOrigin - 1)}>◂ origin</button>
            <button className={btn} disabled={liveBusy}
                    onClick={() => void compute(askOrigin + 1)}>origin ▸</button>
          </>
        )}
        {/* Sign-out lives in the status line, not the button row. As a tenth button it
            pushed the row onto two lines at 1280x720 and 1600x1000 -- measured, 88px
            of footer instead of 66 -- and the stage lost the difference. It also
            belongs next to the name it signs out. */}
        <div className="ml-auto text-[13px] text-serosa">
          {doctor && (
            <span className="mr-3">
              {fill(STR.signedInAs).replace('{doctor}', doctor)}
              {' · '}
              <button onClick={() => void onSignOut()}
                      className="underline underline-offset-2 hover:text-ink">
                sign out
              </button>
            </span>
          )}
          {/* The keyboard hints are the first thing to give at narrow widths: every
              one of them duplicates a button that is still on screen, and below
              1536px keeping them wrapped the control row onto a second line, which
              cost the stage 22px. Who is signed in stays visible at every width. */}
          <span className="hidden 2xl:inline">
            ← → scene · space play · 1 / 3 horizon · R view
            {live ? ' · L run live · [ ] origin' : ''}
          </span>
        </div>
      </footer>
    </div>
  );
}

// The live panel. Deliberately separate from Readout: that one reports the whole
// held-out sequence and its sentences say so, while this reports a single origin and
// says THAT. Collapsing the two would let a sequence-level claim be made from one
// frame without a word of the sentence changing.
function LivePanel({ info, row, busy, error, origin, fill }:
                   { info: LiveInfo; row: LiveRow | null; busy: boolean;
                     error: string | null; origin: number;
                     fill: (s: string) => string }) {
  const skill = row ? liveSkill(row) : null;
  const pct = (x: number | null | undefined) =>
    x === null || x === undefined ? '—' : `${x.toFixed(0)}%`;

  return (
    <div className="mt-6 border-t border-rule pt-4">
      <div className="mb-2 text-xs font-semibold tracking-[0.01em] text-serosa">
        local inference server · {info.params.toLocaleString()} parameters ·
        epoch {info.checkpoint_epoch}
      </div>

      {busy && <p className="text-base text-[#6B4E00]">{fill(STR.liveComputing)}</p>}

      {!busy && error && (
        <div className="border-l-[3px] border-signal bg-signal/8 px-4 py-3 text-base
                        text-oxblood">
          {STR.liveFailed}
          <span className="mt-1 block text-sm text-serosa">{error}</span>
        </div>
      )}

      {!busy && !error && !row && (
        <p className="text-base text-ink">{fill(STR.liveOffer)}</p>
      )}

      {!busy && row && (
        <>
          <p className="mb-3 text-base text-ink">{fill(STR.liveFresh)}</p>
          {skill && (
            <table className="w-full text-base">
              <thead>
                <tr className="text-xs tracking-[0.01em] text-serosa">
                  <th className="py-1 pr-2 text-left font-semibold">
                    origin {origin} alone
                  </th>
                  <th className="py-1 text-right font-semibold">of distance moved</th>
                </tr>
              </thead>
              <tbody>
                <tr><td className="py-1">GASR (ours), computed now</td>
                    <td className="py-1 text-right tabular-nums">{pct(skill.ours)}</td></tr>
                <tr><td className="py-1">Straight-line motion</td>
                    <td className="py-1 text-right tabular-nums">{pct(skill.cv)}</td></tr>
                <tr><td className="py-1">No movement</td>
                    <td className="py-1 text-right tabular-nums">{pct(skill.still)}</td></tr>
              </tbody>
            </table>
          )}
          <p className="mt-3 text-sm text-serosa">{STR.liveScope}</p>
        </>
      )}

      <p className="mt-3 text-sm text-serosa">{fill(STR.liveRange)}</p>
    </div>
  );
}
