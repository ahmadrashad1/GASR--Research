import type { Payload } from './core';

// demo_data.js is loaded by injecting a classic <script>, which is the only mechanism
// a page on the file:// origin still has: fetch() of a local file is refused, and a
// module script cannot be loaded either. The payload assigns window.DEMO_DATA.
//
// It is optional on purpose. Before Module 9 has been exported, or on a copy of this
// folder without it, the load fails and the app runs on placeholder geometry and says
// so on every screen. A missing optional file must never take the page down with it.
declare global {
  interface Window { DEMO_DATA?: Payload }
}

export function loadPayload(): Promise<Payload | null> {
  return new Promise(resolve => {
    if (window.DEMO_DATA) { resolve(window.DEMO_DATA); return; }
    const s = document.createElement('script');
    s.src = 'demo_data.js';
    s.onload = () => resolve(window.DEMO_DATA ?? null);
    s.onerror = () => resolve(null);
    document.head.appendChild(s);
  });
}
