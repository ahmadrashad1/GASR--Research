// The doctor sign-in screen.
//
// Shown only when a local server answered and asked for one. Double-clicked, the page
// is on the file:// origin, makes no request at all, and this component is therefore
// unreachable -- the offline demonstrator is exactly what it was before sign-in
// existed.
//
// WHAT IT IS HONEST ABOUT
// The gate is real, and it is on the server: /api/predict and the bundle metadata in
// /api/health return 401 without a session, so no amount of poking at this page gets
// a prediction out of the checkpoint. What it is NOT is protection of the exported
// measurement. That is in demo_data.js beside the page, it ships on a USB stick, and
// it opens by double-click with no server in sight. A screen implying otherwise would
// be claiming a confidentiality that does not exist, so the screen says what it
// covers and offers the way past.
import { useState } from 'react';
import { signIn } from './live';
import { STR } from './scenes';

export function SignIn({ onSignedIn, onSkip }: {
  onSignedIn: () => void;
  onSkip: () => void;
}) {
  const [roll, setRoll] = useState('');
  const [password, setPassword] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (busy) return;
    setBusy(true);
    setError(null);
    const why = await signIn(roll, password);
    setBusy(false);
    if (why) { setError(why); setPassword(''); return; }
    onSignedIn();
  }

  const field = 'w-full rounded border border-edge bg-stage px-3 py-2 text-base ' +
                'text-bone outline-none focus:border-blood';

  return (
    <div className="grid h-full place-content-center px-6">
      <div className="w-[min(92vw,430px)]">
        <div className="mb-1 text-sm font-bold uppercase tracking-[0.17em] text-blood">
          {STR.signInEyebrow}
        </div>
        <h2 className="mb-2 mt-0 text-[28px] font-bold leading-tight">
          {STR.signInTitle}
        </h2>
        <p className="mb-5 text-base text-[#D8D8DE]">{STR.signInBody}</p>

        <form onSubmit={submit}>
          <label className="mb-1 block text-sm text-dim" htmlFor="roll">
            Roll number
          </label>
          <input id="roll" name="roll" className={`${field} mb-3`} value={roll}
                 autoComplete="username" autoFocus disabled={busy}
                 onChange={e => setRoll(e.target.value)} />

          <label className="mb-1 block text-sm text-dim" htmlFor="password">
            Password
          </label>
          <input id="password" name="password" type="password" className={`${field} mb-4`}
                 value={password} autoComplete="current-password" disabled={busy}
                 onChange={e => setPassword(e.target.value)} />

          {error && (
            <div role="alert" className="mb-4 border-l-[3px] border-blood bg-blood/10
                                         px-4 py-3 text-base text-[#F0CFCF]">
              {error}
            </div>
          )}

          <button type="submit" disabled={busy || !roll || !password}
                  className="w-full rounded border border-blood bg-blood px-4 py-2
                             text-base font-semibold text-bone
                             disabled:cursor-not-allowed disabled:opacity-40">
            {busy ? 'Signing in…' : 'Sign in'}
          </button>
        </form>

        <div className="mt-6 border-t border-edge pt-4">
          <p className="mb-3 text-sm text-dim">{STR.signInScope}</p>
          <button type="button" onClick={onSkip}
                  className="rounded border border-edge px-3 py-1.5 text-sm text-bone
                             hover:border-blood">
            {STR.signInSkip}
          </button>
        </div>
      </div>
    </div>
  );
}
