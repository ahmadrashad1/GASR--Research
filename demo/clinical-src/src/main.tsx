import { StrictMode, useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './index.css';
import { App } from './App';
import { loadPayload } from './data';
import { synth } from './synth';
import type { Payload } from './core';

function Boot() {
  const [data, setData] = useState<Payload | null>(null);

  useEffect(() => {
    let live = true;
    loadPayload().then(p => { if (live) setData(p ?? synth()); });
    return () => { live = false; };
  }, []);

  if (!data) {
    return <div className="grid h-full place-content-center text-dim">loading…</div>;
  }
  return <App data={data} placeholder={data.synthetic === true} />;
}

createRoot(document.getElementById('root')!).render(
  <StrictMode><Boot /></StrictMode>
);
