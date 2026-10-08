"""
Architecture figure for slide 14, drawn in the idiom of the Transformer paper
figure: light plate, pastel blocks with thin outlines, bottom-up flow, grey
repeat containers with an N-times label.

Content is taken from the implementation, not sketched -- ConditionalDenoiser
in module5_diffusion.py for the left plate, the sampler and evaluation path in
module6_prediction.py for the right one.
"""

# paper-figure palette
PAPER   = '#F6F3EC'
INK     = '#1C1C1C'
LINE    = '#2A2A2A'
GREY_T  = '#5C5C5C'
CONT_F, CONT_S = '#EDEAE3', '#C6C1B7'
PINK    = ('#F7D6D8', '#C28A8E')   # inputs / encoders
ORANGE  = ('#FBDCB2', '#C4934D')   # attention
BLUE    = ('#C9DFF2', '#6E9BC4')   # feed-forward / compute
OLIVE   = ('#E9EFC6', '#A4B36C')   # add & norm
PURPLE  = ('#D8D1E8', '#8E85B6')   # linear
GREEN   = ('#C8E3BB', '#7CA56D')   # outputs
WHITE   = ('#FFFFFF', '#9A958C')

P = []


def esc(t):
    return t.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def rbox(cx, y, w, h, lines, pal, small=False):
    f, s = pal
    x = cx - w / 2
    P.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="{f}" '
             f'stroke="{s}" stroke-width="1.1"/>')
    cls = 'bs' if small else 'bl'
    n = len(lines)
    lh = 15 if small else 16
    y0 = y + h / 2 - (n - 1) * lh / 2 + (4 if small else 5)
    for i, t in enumerate(lines):
        P.append(f'<text x="{cx}" y="{y0 + i * lh}" class="{cls}">{esc(t)}</text>')


def cont(x, y, w, h, times, side='left'):
    P.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="{CONT_F}" '
             f'stroke="{CONT_S}" stroke-width="1"/>')
    tx = x - 16 if side == 'left' else x + w + 16
    P.append(f'<text x="{tx}" y="{y + h / 2 + 5}" class="nx">{esc(times)}</text>')


def up(cx, y1, y2, head=True):
    """Vertical flow arrow, drawn upward (y1 below y2)."""
    m = ' marker-end="url(#a)"' if head else ''
    P.append(f'<line x1="{cx}" y1="{y1}" x2="{cx}" y2="{y2}" stroke="{LINE}" '
             f'stroke-width="1.5"{m}/>')


def path(d, head=True, dash=False, col=None):
    m = ' marker-end="url(#a)"' if head else ''
    da = ' stroke-dasharray="5 4"' if dash else ''
    P.append(f'<path d="{d}" fill="none" stroke="{col or LINE}" stroke-width="1.5"{m}{da}/>')


def oplus(cx, cy):
    P.append(f'<circle cx="{cx}" cy="{cy}" r="11" fill="#fff" stroke="{LINE}" stroke-width="1.4"/>')
    P.append(f'<line x1="{cx-6}" y1="{cy}" x2="{cx+6}" y2="{cy}" stroke="{LINE}" stroke-width="1.4"/>')
    P.append(f'<line x1="{cx}" y1="{cy-6}" x2="{cx}" y2="{cy+6}" stroke="{LINE}" stroke-width="1.4"/>')


def squiggle(cx, cy):
    """The sine motif the reference uses for positional encoding."""
    P.append(f'<path d="M {cx-15} {cy} q 3.75 -11 7.5 0 t 7.5 0" fill="none" '
             f'stroke="{LINE}" stroke-width="1.4"/>')
    P.append(f'<circle cx="{cx}" cy="{cy}" r="15" fill="none" stroke="{LINE}" stroke-width="1.3"/>')


def txt(x, y, t, cls='lbl', anchor='middle'):
    P.append(f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{esc(t)}</text>')


def build():
    P.clear()
    P.append(f'''<defs>
<marker id="a" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5"
 orient="auto-start-reverse"><path d="M0,1.2 L8.5,5 L0,8.8 z" fill="{LINE}"/></marker>
<marker id="p" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5"
 orient="auto-start-reverse"><path d="M0,1.2 L8.5,5 L0,8.8 z" fill="#8E85B6"/></marker>
</defs>''')
    P.append(f'<rect x="0" y="0" width="780" height="812" rx="6" fill="{PAPER}"/>')
    P.append(f'<rect x="840" y="0" width="780" height="812" rx="6" fill="{PAPER}"/>')

    # ═══════════ PLATE A — Module 5, read bottom to top ═══════════
    txt(390, 28, 'MODULE 5  —  Conditional Denoiser', 'plate')
    txt(390, 47, 'one forward pass  ·  16.3 M parameters', 'plsub')
    L, R = 215, 565

    rbox(390, 62, 250, 34, ['Predicted velocity  v̂'], GREEN)
    txt(390, 112, 'v-prediction · cosine schedule · Min-SNR-γ', 'shape')
    up(390, 134, 96)
    rbox(390, 140, 250, 34, ['Concat  →  Output Head'], PURPLE)
    txt(390, 190, '(B, 512, 1024) → 512 → 3', 'shape')

    # bidirectional cross-attention
    cont(100, 212, 576, 116, '', 'left')

    rbox(L, 222, 196, 28, ['Add & Norm'], OLIVE, small=True)
    rbox(R, 222, 196, 28, ['Add & Norm'], OLIVE, small=True)
    rbox(L, 266, 196, 34, ['Cross-Attn  c ← t'], ORANGE, small=True)
    rbox(R, 266, 196, 34, ['Cross-Attn  t ← c'], ORANGE, small=True)
    up(L, 266, 250); up(R, 266, 250)
    path(f'M {L} 222 V 196 H 390 V 174', head=True)
    path(f'M {R} 222 V 196 H 390', head=False)
    # each branch is query on its own side, key/value on the other
    path(f'M {L+98} 276 H {R-98}', dash=True)
    path(f'M {R-98} 290 H {L+98}', dash=True)
    txt(390, 344, 'BIDIRECTIONAL CROSS-ATTENTION   ·   dashed = key / value', 'grp')

    # left branch: temporal transformer over the 4-frame axis
    up(L, 300, 288)
    oplus(L, 372)
    up(L, 361, 334)
    rbox(L, 386, 196, 34, ['Last-frame pool'], WHITE, small=True)
    cont(100, 434, 230, 196, '3 ×', 'left')
    rbox(L, 444, 196, 28, ['Add & Norm'], OLIVE, small=True)
    rbox(L, 488, 196, 34, ['Feed Forward'], BLUE, small=True)
    rbox(L, 544, 196, 28, ['Add & Norm'], OLIVE, small=True)
    rbox(L, 588, 196, 34, ['Self-Attention over the 4 frames'], ORANGE, small=True)
    up(L, 444, 420); up(L, 488, 472); up(L, 544, 522); up(L, 588, 572)
    path(f'M {L-108} 605 V 558 H {L-98}')
    path(f'M {L-108} 505 V 458 H {L-98}')
    up(L, 658, 622)

    # right branch is a single frame, so no temporal stack
    up(R, 658, 300)
    txt(R + 118, 470, 'the candidate is', 'note')
    txt(R + 118, 484, 'a single frame —', 'note')
    txt(R + 118, 498, 'no temporal stack', 'note')

    # conditioning
    oplus(L, 669); oplus(R, 669)
    squiggle(L - 84, 669); txt(L - 84, 706, 'Temporal', 'cond'); txt(L - 84, 720, 'Encoding', 'cond')
    squiggle(R + 84, 669); txt(R + 84, 706, 'Diffusion', 'cond'); txt(R + 84, 720, 'step  t', 'cond')
    path(f'M {L-69} 669 H {L-11}')
    path(f'M {R+69} 669 H {R+11}')
    rbox(390, 684, 152, 32, ['Horizon  k ∈ {1,2,3}'], ORANGE, small=True)
    txt(390, 732, 'Embedding → MLP', 'shape')
    path(f'M 390 684 V 372 H {L+11}')
    path(f'M 390 684 V 669 H {R-11}')

    up(L, 730, 700); up(R, 730, 700)
    rbox(L, 736, 182, 32, ['Point Encoder'], PINK, small=True)
    rbox(R, 736, 182, 32, ['Point Encoder'], PINK, small=True)
    txt(L, 794, 'Past 4 frames   (B, 4, 512, 3)', 'io')
    txt(R, 794, 'Noisy displacement xₜ   (B, 512, 3)', 'io')
    up(L, 782, 770); up(R, 782, 770)

    # ═══════════ PLATE B — Module 6 ═══════════
    txt(1230, 28, 'MODULE 6  —  Inference & Evaluation', 'plate')
    txt(1230, 47, 'loads the checkpoint  ·  trains nothing', 'plsub')
    C = 1230

    rbox(C, 62, 300, 34, ['Metrics · figures · exports'], GREEN)
    up(C, 118, 96)
    rbox(C, 124, 300, 34, ['QT0 – QT5  quality gates'], OLIVE, small=True)
    up(C, 180, 158)
    rbox(C, 186, 300, 34, ['Chamfer · MPE · cosine · spread'], PURPLE, small=True)
    rbox(C + 232, 186, 150, 34, ['Baselines'], WHITE, small=True)
    txt(C + 232, 234, 'nothing moves ·', 'shape')
    txt(C + 232, 248, 'constant velocity', 'shape')
    path(f'M {C+157} 203 H {C+151}')

    for cx, t, sub in ((C - 176, 'Direct', 'k ≤ 3'),
                       (C, 'Rollout', '10 frames'),
                       (C + 176, '5 futures', 'uncertainty')):
        rbox(cx, 282, 158, 34, [t], BLUE, small=True)
        txt(cx, 330, sub, 'shape')
        path(f'M {cx} 282 V 258 H {C} V 220', head=(cx == C))
    path(f'M {C-176} 258 H {C+176}', head=False)
    up(C, 258, 220)

    rbox(C, 352, 260, 34, ['Predicted displacement  x̂₀'], WHITE, small=True)
    up(C, 352, 316)
    oplus(C, 404)
    rbox(C + 196, 388, 158, 34, ['Constant-velocity base'], OLIVE, small=True)
    path(f'M {C+117} 404 H {C+11}')
    up(C, 393, 368)

    cont(1080, 444, 300, 160, '20 ×', 'right')
    rbox(C, 456, 260, 34, ['DDIM update   xₜ → xₜ₋₁'], BLUE, small=True)
    rbox(C, 540, 260, 34, ['Frozen denoiser  (Module 5)'], ORANGE, small=True)
    up(C, 540, 496)
    up(C, 456, 434)
    path(f'M {C+140} 473 V 557 H {C+132}')

    rbox(C - 160, 648, 230, 34, ['Trajectory replay + QT0'], PINK, small=True)
    rbox(C + 170, 648, 170, 34, ['xₜ ~ 𝒩(0, I)'], WHITE, small=True)
    path(f'M {C-160} 648 V 616 H {C-40} V 580', head=True)
    path(f'M {C+170} 648 V 616 H {C+40} V 580', head=True)
    txt(C - 160, 704, 'held-out context, verified', 'shape')
    txt(C - 160, 718, 'against Module 4 first', 'shape')
    txt(C + 170, 704, 'fresh noise per sample', 'shape')

    # checkpoint crosses the gap between the plates
    path('M 782 557 H 1074', col='#5FA8A0')
    P.append('<text x="808" y="545" class="ckpt" text-anchor="middle" '
             'transform="rotate(-90 808 545)">trained checkpoint</text>')

    style = f"""
.bl{{font-family:'IBM Plex Sans',Helvetica,sans-serif;font-size:14.5px;fill:{INK};text-anchor:middle}}
.bs{{font-family:'IBM Plex Sans',Helvetica,sans-serif;font-size:13px;fill:{INK};text-anchor:middle}}
.shape{{font-family:'IBM Plex Mono',monospace;font-size:11px;fill:{GREY_T};text-anchor:middle}}
.io{{font-family:'IBM Plex Sans',Helvetica,sans-serif;font-size:14px;fill:{INK};text-anchor:middle}}
.cond{{font-family:'IBM Plex Sans',Helvetica,sans-serif;font-size:12px;fill:{INK};text-anchor:middle}}
.nx{{font-family:'IBM Plex Sans',Helvetica,sans-serif;font-size:15px;fill:{INK};text-anchor:middle}}
.plate{{font-family:'IBM Plex Sans',Helvetica,sans-serif;font-size:17px;font-weight:600;
  fill:{INK};text-anchor:middle;letter-spacing:.04em}}
.plsub{{font-family:'IBM Plex Mono',monospace;font-size:11.5px;fill:{GREY_T};text-anchor:middle}}
.grp{{font-family:'IBM Plex Sans',Helvetica,sans-serif;font-size:12.5px;fill:{GREY_T};
  text-anchor:middle;letter-spacing:.08em}}
.note{{font-family:'IBM Plex Mono',monospace;font-size:10.5px;fill:{GREY_T};text-anchor:middle}}
.ckpt{{font-family:'IBM Plex Mono',monospace;font-size:11px;fill:#5FA8A0;
  letter-spacing:.12em}}
"""
    return (f'<svg class="archsvg" viewBox="0 0 1620 824" width="100%" '
            f'preserveAspectRatio="xMidYMid meet" role="img" '
            f'aria-label="Architecture of Modules 5 and 6">'
            f'<style>{style}</style>' + ''.join(P) + '</svg>')


if __name__ == '__main__':
    open('/tmp/arch_preview.html', 'w').write(
        '<body style="margin:0;background:#07100F">' + build() + '</body>')
    print('preview written')
