// happy-dom has no canvas 2D context. Without a stub the components return early and
// the drawing code -- where a NaN coordinate or a bad index would surface -- is never
// executed by any test. This records every arc so a test can assert something was
// drawn, and so a non-finite radius or coordinate fails loudly.
export const arcs: { x: number; y: number; r: number }[] = [];

const ctx = new Proxy({} as Record<string, unknown>, {
  get: (o, k) => {
    if (k === 'arc') return (x: number, y: number, r: number) => {
      if (![x, y, r].every(Number.isFinite)) {
        throw new Error(`non-finite arc: ${x}, ${y}, ${r}`);
      }
      arcs.push({ x, y, r });
    };
    return k in o ? o[k as string] : () => {};
  },
  set: (o, k, v) => { o[k as string] = v; return true; },
});

(globalThis.HTMLCanvasElement as unknown as { prototype: Record<string, unknown> })
  .prototype.getContext = () => ctx;

// ── the network ───────────────────────────────────────────────────────────────
// No test may reach the network. The default below stands in for "no local server is
// running", which is the configuration the demonstrator has to survive and the one it
// will be in when it is double-clicked.
//
// This stub is not optional housekeeping: happy-dom reports location.protocol as
// http:, so the live probe DOES fire under test. Without it the suite makes a real
// connection attempt to localhost on every render of App.
export const fetchCalls: string[] = [];

export function stubFetch(handler?: (url: string, init?: RequestInit) => unknown) {
  fetchCalls.length = 0;
  globalThis.fetch = (async (url: string | URL | Request, init?: RequestInit) => {
    const u = String(typeof url === 'string' ? url : (url as Request).url ?? url);
    fetchCalls.push(u);
    if (!handler) throw new TypeError('fetch failed');
    const body = handler(u, init);
    if (body === undefined) throw new TypeError('fetch failed');
    return {
      ok: true,
      status: 200,
      json: async () => body,
    } as Response;
  }) as typeof fetch;
}

stubFetch();
