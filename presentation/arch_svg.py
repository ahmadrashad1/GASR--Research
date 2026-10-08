"""
Architecture figure for slide 14, drawn from the actual implementation:
ConditionalDenoiser.forward() in module5_diffusion.py and the sampler and
evaluation path in module6_prediction.py. Every shape annotation and layer
width below is taken from the source, not approximated.
"""

BONE, DIM, TEAL, CORAL = '#E9E6DE', '#6E7E7C', '#5FA8A0', '#D2563F'
GHOST = 'rgba(233,230,222,.30)'


def esc(t):
    return (t.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))


def box(x, y, w, h, title, sub=None, tone='n', dash=False):
    """A labelled node. `sub` is the tensor shape / layer widths, in mono."""
    stroke = {'t': TEAL, 'c': CORAL, 'n': 'rgba(233,230,222,.22)'}[tone]
    fill = {'t': 'rgba(95,168,160,.10)', 'c': 'rgba(210,86,63,.10)',
            'n': 'rgba(7,16,15,.74)'}[tone]
    d = ' stroke-dasharray="4 4"' if dash else ''
    o = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" '
         f'stroke="{stroke}" stroke-width="1"{d}/>')
    cx = x + w / 2
    if sub:
        o += (f'<text x="{cx}" y="{y + h/2 - 4}" class="nt">{esc(title)}</text>'
              f'<text x="{cx}" y="{y + h/2 + 15}" class="ns">{esc(sub)}</text>')
    else:
        o += f'<text x="{cx}" y="{y + h/2 + 5}" class="nt">{esc(title)}</text>'
    return o


def arrow(x1, y1, x2, y2, tone='n', dash=False):
    col = {'t': TEAL, 'c': CORAL, 'n': GHOST}[tone]
    d = ' stroke-dasharray="5 4"' if dash else ''
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" '
            f'stroke-width="1.4" marker-end="url(#ah)"{d}/>')


def elbow(x1, y1, x2, y2, tone='n'):
    """Right-angle connector: vertical first, then horizontal into the target."""
    col = {'t': TEAL, 'c': CORAL, 'n': GHOST}[tone]
    return (f'<path d="M {x1} {y1} V {y2} H {x2}" fill="none" stroke="{col}" '
            f'stroke-width="1.4" marker-end="url(#ah)"/>')


def plus(cx, cy):
    return (f'<circle cx="{cx}" cy="{cy}" r="13" fill="rgba(7,16,15,.9)" '
            f'stroke="{TEAL}" stroke-width="1.2"/>'
            f'<text x="{cx}" y="{cy + 6}" class="plus">+</text>')


def label(x, y, t, cls='lb'):
    return f'<text x="{x}" y="{y}" class="{cls}">{esc(t)}</text>'


def build():
    p = []
    # ── defs ────────────────────────────────────────────────────────
    p.append(f'''<defs>
<marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7"
        orient="auto-start-reverse"><path d="M0,1 L9,5 L0,9 z" fill="{GHOST}"/></marker>
<marker id="aht" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7"
        orient="auto-start-reverse"><path d="M0,1 L9,5 L0,9 z" fill="{TEAL}"/></marker>
</defs>''')

    # ════ MODULE 5 ═════════════════════════════════════════════════
    p.append(f'<rect x="0" y="0" width="1620" height="372" fill="rgba(95,168,160,.045)"/>')
    p.append(label(16, 26, 'MODULE 5 — CONDITIONAL DENOISER  ·  one forward pass', 'band t'))

    CY, TY = 86, 206          # context-branch and target-branch row centres
    H = 56
    # context branch
    p.append(box(16, CY - H / 2, 148, H, 'past 4 frames', '(B, 4, 512, 3)'))
    p.append(arrow(164, CY, 186, CY))
    p.append(box(186, CY - H / 2, 132, H, 'GeomMLP', '3→1024³→512'))
    p.append(arrow(318, CY, 340, CY))
    p.append(box(340, CY - H / 2, 118, H, '+ temporal PE', '(4, 512)'))
    p.append(arrow(458, CY, 480, CY))
    p.append(box(480, CY - H / 2, 212, H, 'temporal transformer',
                 '3 layers · 8 heads · FF 1024'))
    p.append(arrow(692, CY, 714, CY))
    p.append(box(714, CY - H / 2, 126, H, 'last-token pool', '(B, 512, 512)'))
    p.append(arrow(840, CY, 876, CY))
    # attention is over the 4-frame axis only - worth stating, it is unusual
    p.append(label(482, CY + 48, 'attention over the W = 4 axis only', 'note'))

    # target branch
    p.append(box(16, TY - H / 2, 148, H, 'noisy target  xₜ', '(B, 512, 3)'))
    p.append(arrow(164, TY, 186, TY))
    p.append(box(186, TY - H / 2, 132, H, 'GeomMLP', '3→1024³→512'))
    p.append(arrow(318, TY, 340, TY))
    p.append(box(340, TY - H / 2, 212, H, '+ sinusoidal t', 'diffusion step embedding'))
    p.append(arrow(552, TY, 876, TY))

    # horizon conditioning, injected into BOTH branches (the fix that made it learnable)
    p.append(box(596, 296, 262, 54, 'horizon  k ∈ {1,2,3}',
                 'Embedding(512) → MLP', tone='c'))
    p.append(f'<path d="M 727 296 V 262 H 889 V {TY + 20}" fill="none" stroke="{CORAL}" '
             f'stroke-width="1.3" marker-end="url(#ah)"/>')
    p.append(f'<path d="M 727 296 V 276 H 905 V {CY + 20}" fill="none" stroke="{CORAL}" '
             f'stroke-width="1.3" marker-end="url(#ah)"/>')
    p.append(plus(889, CY))
    p.append(plus(889, TY))
    p.append(label(600, 366, 'injected into both branches — without it the same window has '
                             'three different right answers', 'note'))

    # bidirectional cross-attention over the point axis
    p.append(box(962, CY - H / 2, 186, H, 'cross-attn  c ← t', 'query: context · 8 heads'))
    p.append(box(962, TY - H / 2, 186, H, 'cross-attn  t ← c', 'query: target · 8 heads'))
    # each branch is the query on its own row and the key/value on the other
    p.append(arrow(902, CY, 962, CY))
    p.append(arrow(902, TY, 962, TY))
    p.append(f'<path d="M 920 {CY + 16} V {TY - 16} H 958" fill="none" stroke="{GHOST}" '
             f'stroke-width="1.1" stroke-dasharray="4 3" marker-end="url(#ah)"/>')
    p.append(f'<path d="M 936 {TY - 16} V {CY + 16} H 958" fill="none" stroke="{GHOST}" '
             f'stroke-width="1.1" stroke-dasharray="4 3" marker-end="url(#ah)"/>')
    p.append(label(905, 272, 'dashed = key / value', 'note'))

    p.append(f'<path d="M 1148 {CY} H 1180 V 140" fill="none" stroke="{GHOST}" '
             f'stroke-width="1.4"/>')
    p.append(f'<path d="M 1148 {TY} H 1180 V 152" fill="none" stroke="{GHOST}" '
             f'stroke-width="1.4"/>')
    p.append(arrow(1180, 146, 1212, 146))
    p.append(box(1212, 118, 152, H, 'concat', '(B, 512, 1024)'))
    p.append(arrow(1364, 146, 1386, 146))
    p.append(box(1386, 118, 148, H, 'output MLP', '1024→512→3'))
    p.append(arrow(1534, 146, 1556, 146))
    p.append(box(1556, 118, 56, H, 'v̂', None, tone='t'))
    p.append(label(1212, 212, 'v-prediction · cosine schedule T = 500', 'note'))
    p.append(label(1212, 230, 'Min-SNR-γ = 5 loss weighting', 'note'))
    p.append(label(1212, 248, 'target = correction to constant velocity', 'note c'))

    # ════ checkpoint hand-off ══════════════════════════════════════
    p.append(f'<path d="M 1584 174 V 402 H 150" fill="none" stroke="{TEAL}" '
             f'stroke-width="1.4" marker-end="url(#aht)"/>')
    p.append(label(1266, 392, 'trained checkpoint  +  architecture descriptor', 'note t'))

    # ════ MODULE 6 ═════════════════════════════════════════════════
    p.append(f'<rect x="0" y="416" width="1620" height="244" fill="rgba(210,86,63,.05)"/>')
    p.append(label(16, 442, 'MODULE 6 — INFERENCE & EVALUATION  ·  trains nothing', 'band c'))

    RY, H6 = 506, 54
    p.append(box(16, RY - H6 / 2, 168, H6, 'trajectory replay', '+ QT0 integrity gate', tone='c'))
    p.append(arrow(184, RY, 206, RY))
    p.append(box(206, RY - H6 / 2, 124, H6, 'x_T ~ 𝒩(0, I)', None))
    p.append(arrow(330, RY, 352, RY))
    p.append(box(352, RY - H6 / 2, 196, H6, 'DDIM sampler', '20 of 500 steps · η = 0'))
    # the denoising loop
    p.append(f'<path d="M 470 {RY + H6/2} V 566 H 392 V {RY + H6/2}" fill="none" '
             f'stroke="{GHOST}" stroke-width="1.2" marker-end="url(#ah)"/>')
    p.append(label(412, 584, 'iterate × 20', 'note'))
    p.append(arrow(548, RY, 570, RY))
    p.append(box(570, RY - H6 / 2, 176, H6, '+ base prediction', 'constant velocity'))
    p.append(arrow(746, RY, 782, RY))

    # three query modes
    for i, (t, s) in enumerate([('direct', 'k ≤ 3'),
                                ('rollout', '10 frames, fed back'),
                                ('5 futures', 'spread = uncertainty')]):
        y = 448 + i * 62
        p.append(box(782, y, 206, 50, t, s, tone='c' if i == 2 else 'n'))
        p.append(f'<path d="M 770 {RY} V {y + 25} H 782" fill="none" stroke="{GHOST}" '
                 f'stroke-width="1.2" marker-end="url(#ah)"/>')
        p.append(f'<path d="M 988 {y + 25} H 1020 V {RY}" fill="none" stroke="{GHOST}" '
                 f'stroke-width="1.2"/>')

    p.append(arrow(1020, RY, 1056, RY))
    p.append(box(1056, RY - H6 / 2, 200, H6, 'Chamfer · MPE · cosine', 'spread vs realised error'))
    p.append(arrow(1256, RY, 1278, RY))
    p.append(box(1278, RY - H6 / 2, 180, H6, 'QT0 – QT5', 'figures · exports'))
    p.append(arrow(1458, RY, 1480, RY))
    p.append(box(1480, RY - H6 / 2, 132, H6, 'Module 7', 'comparison', tone='c', dash=True))

    # the two baselines score on identical windows
    p.append(box(1016, 598, 280, 44, 'nothing moves  ·  constant velocity', None, dash=True))
    p.append(f'<path d="M 1156 598 V {RY + H6/2}" fill="none" stroke="{GHOST}" '
             f'stroke-width="1.2" marker-end="url(#ah)"/>')
    p.append(label(1016, 660, 'baselines, scored on identical windows', 'note'))

    style = f"""
.nt{{font-family:'IBM Plex Sans',sans-serif;font-size:16px;fill:{BONE};text-anchor:middle}}
.ns{{font-family:'IBM Plex Mono',monospace;font-size:12.5px;fill:{DIM};text-anchor:middle}}
.plus{{font-family:'IBM Plex Sans',sans-serif;font-size:17px;fill:{TEAL};text-anchor:middle}}
.band{{font-family:'IBM Plex Mono',monospace;font-size:14px;letter-spacing:.2em}}
.band.t{{fill:{TEAL}}} .band.c{{fill:{CORAL}}}
.note{{font-family:'IBM Plex Mono',monospace;font-size:12.5px;fill:{DIM}}}
.note.t{{fill:{TEAL}}} .note.c{{fill:{CORAL}}}
.lb{{font-family:'IBM Plex Mono',monospace;font-size:13px;fill:{DIM}}}
"""
    return (f'<svg class="archsvg" viewBox="0 0 1620 666" width="100%" '
            f'preserveAspectRatio="xMidYMid meet" role="img" '
            f'aria-label="Architecture of Modules 5 and 6">'
            f'<style>{style}</style>' + ''.join(p) + '</svg>')


if __name__ == '__main__':
    print(build()[:400])
