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
