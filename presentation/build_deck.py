#!/usr/bin/env python3
"""FYP-2 mid-evaluation deck.

Built as flat ODF (a single XML file) and converted to .pptx by LibreOffice,
because neither pptxgenjs nor python-pptx is installable in this environment.
The result is a normal, fully editable PowerPoint file.
"""
import html, subprocess, sys, os

W, H = 33.867, 19.05          # 16:9 in cm
INK   = '11191B'
TEAL  = '0E5C58'
TEAL2 = '1F6F6B'
MINT  = 'CFE3E1'
PAPER = 'F7F9F9'
GOLD  = 'A97B10'
BRICK = '9A3B2E'
GREY  = '5B6B72'
RULE  = 'D7E0DF'

styles, bodies = [], []
_uid = [0]


def uid(p):
    _uid[0] += 1
    return f'{p}{_uid[0]}'


def gstyle(fill=None, stroke='none', stroke_color=None, stroke_w=0.03, arrow=False,
           valign='middle'):
    n = uid('gr')
    props = [f'draw:fill="{"solid" if fill else "none"}"']
    if fill:
        props.append(f'draw:fill-color="#{fill}"')
    props.append(f'draw:stroke="{stroke}"')
    if stroke != 'none':
        props.append(f'svg:stroke-color="#{stroke_color or INK}"')
        props.append(f'svg:stroke-width="{stroke_w}cm"')
    if arrow:
        props.append('draw:marker-end="Arrow" draw:marker-end-width="0.28cm"')
    props.append('draw:auto-grow-height="false" draw:auto-grow-width="false"')
    props.append('fo:padding-top="0.12cm" fo:padding-bottom="0.12cm"')
    props.append('fo:padding-left="0.22cm" fo:padding-right="0.22cm"')
    props.append(f'draw:textarea-vertical-align="{valign}"')
    styles.append(f'<style:style style:name="{n}" style:family="graphic">'
                  f'<style:graphic-properties {" ".join(props)}/></style:style>')
    return n


def pstyle(size, color=INK, bold=False, align='left', font='Calibri', space_after=0.0,
           italic=False):
    n = uid('P')
    styles.append(
        f'<style:style style:name="{n}" style:family="paragraph">'
        f'<style:paragraph-properties fo:text-align="{align}" '
        f'fo:margin-bottom="{space_after}cm"/>'
        f'<style:text-properties fo:font-size="{size}pt" fo:color="#{color}" '
        f'fo:font-weight="{"bold" if bold else "normal"}" '
        f'fo:font-style="{"italic" if italic else "normal"}" '
        f'style:font-name="{font}"/></style:style>')
    return n


def esc(t):
    return html.escape(str(t), quote=False)


def box(x, y, w, h, paras, fill=None, stroke='none', stroke_color=None, stroke_w=0.03,
        valign='middle'):
    """paras: list of (text, pstyle_name). Empty text -> blank spacer line."""
    g = gstyle(fill, stroke, stroke_color, stroke_w, valign=valign)
    ps = ''.join(
        f'<text:p text:style-name="{s}">{esc(t) if t else ""}</text:p>' for t, s in paras)
    return (f'<draw:frame draw:style-name="{g}" svg:x="{x}cm" svg:y="{y}cm" '
            f'svg:width="{w}cm" svg:height="{h}cm"><draw:text-box>{ps}</draw:text-box>'
            f'</draw:frame>')


def rect(x, y, w, h, fill, stroke='none', stroke_color=None):
    g = gstyle(fill, stroke, stroke_color)
    return (f'<draw:frame draw:style-name="{g}" svg:x="{x}cm" svg:y="{y}cm" '
            f'svg:width="{w}cm" svg:height="{h}cm"><draw:text-box/></draw:frame>')


def arrow(x1, y1, x2, y2, color=TEAL2, w=0.05):
    g = gstyle(None, 'solid', color, w, arrow=True)
    return (f'<draw:line draw:style-name="{g}" svg:x1="{x1}cm" svg:y1="{y1}cm" '
            f'svg:x2="{x2}cm" svg:y2="{y2}cm"/>')


def line(x1, y1, x2, y2, color=RULE, w=0.03):
    g = gstyle(None, 'solid', color, w)
    return (f'<draw:line draw:style-name="{g}" svg:x1="{x1}cm" svg:y1="{y1}cm" '
            f'svg:x2="{x2}cm" svg:y2="{y2}cm"/>')


_pages = {}


def slide(name, content, dark=False, notes=''):
    """Record a slide. Position in the deck is decided by ORDER, not by call order."""
    _pages[name] = (content, dark, notes)


def _render(name, number):
    content, dark, notes = _pages[name]
    bg = INK if dark else 'FFFFFF'
    if not dark:
        # Anchors the bottom margin, so space below the content reads as a deliberate
        # margin rather than an unfinished slide.
        fl = pstyle(9.5, 'A3B2B0')
        fr = pstyle(9.5, 'A3B2B0', align='right')
        content += (line(1.6, 17.45, 32.3, 17.45, 'ECF1F0') +
                    box(1.6, 17.55, 22, 0.75,
                        [('GASR  /  S26-017-R-GASR  /  FYP-2 Mid Evaluation', fl)]) +
                    box(26.3, 17.55, 6.0, 0.75, [(str(number), fr)]))
    n = uid('dp')
    styles.append(f'<style:style style:name="{n}" style:family="drawing-page">'
                  f'<style:drawing-page-properties draw:fill="solid" '
                  f'draw:fill-color="#{bg}"/></style:style>')
    note = ''
    if notes:
        np_ = pstyle(12)
        note = (f'<presentation:notes><draw:frame svg:x="2cm" svg:y="2cm" '
                f'svg:width="24cm" svg:height="12cm"><draw:text-box>'
                f'<text:p text:style-name="{np_}">{esc(notes)}</text:p>'
                f'</draw:text-box></draw:frame></presentation:notes>')
    bodies.append(f'<draw:page draw:name="{esc(name)}" draw:master-page-name="Default" '
                  f'draw:style-name="{n}">{content}{note}</draw:page>')


def emit(order):
    missing = [n for n in order if n not in _pages]
    extra = [n for n in _pages if n not in order]
    if missing or extra:
        raise SystemExit(f'ORDER mismatch -- missing: {missing}, never shown: {extra}')
    for i, name in enumerate(order, 1):
        _render(name, i)


# ── reusable slide furniture ────────────────────────────────────────
def title_bar(text, kicker=None):
    s = pstyle(30, INK, True, font='Cambria')
    out = box(1.6, 0.9, 30.7, 2.3, [(text, s)])
    if kicker:
        k = pstyle(12.5, TEAL2, True)
        out = box(1.6, 0.5, 30.7, 0.72, [(kicker.upper(), k)]) + \
              box(1.6, 1.25, 30.7, 2.2, [(text, s)])
    return out


def card(x, y, w, h, head, lines, accent=TEAL2, fill=PAPER, head_size=14.5, body_size=12.5):
    hs = pstyle(head_size, accent, True)
    bs = pstyle(body_size, INK, space_after=0.08)
    paras = [(head, hs)] + [(l, bs) for l in lines]
    return box(x, y, w, h, paras, fill=fill, valign='top')


def stat(x, y, w, value, label, color=TEAL):
    v = pstyle(34, color, True, align='center', font='Cambria')
    l = pstyle(11.5, GREY, align='center')
    return box(x, y, w, 3.3, [(value, v), (label, l)], fill=PAPER)


# ════════════════════════════════════════════════════════════════════
# 1 — title
# ════════════════════════════════════════════════════════════════════
t1 = pstyle(40, 'FFFFFF', True, font='Cambria')
t2 = pstyle(17, MINT)
t3 = pstyle(13, 'A9BFBD')
t4 = pstyle(13, 'FFFFFF', True)
slide('Title', (
    rect(0, 0, 0.55, H, TEAL2) +
    box(2.4, 4.4, 28, 3.4, [
        ('From Reconstruction to Anticipation', t1),
        ('Predicting future tissue geometry from monocular laparoscopic video', t2)]) +
    box(2.4, 9.3, 28, 1.0, [('FYP-2 Mid Evaluation  ·  Modules 5 and 6', t4)]) +
    box(2.4, 12.6, 28, 2.6, [
        ('Ahmad Rashad  ·  Shehroz  ·  Humair', t3),
        ('Supervisor: Dr Akhtar Jamil', t3),
        ('Department of Computer Science, FAST NUCES Islamabad', t3)])
), dark=True, notes='Mid-evaluation for FYP-2. FYP-1 delivered reconstruction; this '
                    'presentation is about Modules 5 and 6, the prediction stage.')

# ════════════════════════════════════════════════════════════════════
# 2 — the problem
# ════════════════════════════════════════════════════════════════════
b = pstyle(15, INK, space_after=0.28)
bi = pstyle(14.5, GREY, italic=True)
slide('Problem', (
    title_bar('The laparoscope shows the present, never the next moment',
              'motivation') +
    box(1.6, 3.9, 15.4, 7.0, [
        ('A surgeon operates through one camera. It gives a flat image with no depth, '
         'no 3D structure, and no indication of how tissue will respond to the '
         'instrument about to touch it.', b),
        ('', b),
        ('Reconstruction research has solved the first problem well. It has not '
         'addressed the second at all.', b),
        ('', b),
        ('Every published surgical 4D reconstruction system is retrospective: it '
         'models deformation that has already been recorded.', bi)]) +
    card(18.0, 3.9, 14.3, 3.1, 'What exists today', [
        'Reconstruct the scene in 3D, for frames already captured'], TEAL2) +
    card(18.0, 7.6, 14.3, 3.3, 'What is missing', [
        'Anticipate where tissue will be a moment from now,',
        'with an honest indication of confidence'], BRICK) +
    stat(1.6, 12.3, 7.1, '0', 'depth sensors available') +
    stat(9.4, 12.3, 7.1, '1–2 px', 'tissue motion per frame') +
    stat(17.2, 12.3, 7.1, '0', 'deformation labels in clinical video') +
    stat(25.0, 12.3, 7.3, '10', 'videos reconstructed')
), notes='Set up the gap: reconstruction is solved, anticipation is not. The motion we '
         'must predict is one to two pixels per frame, which is the core difficulty.')

# ════════════════════════════════════════════════════════════════════
# 3 — where we are: FYP-1 recap
# ════════════════════════════════════════════════════════════════════
sm = pstyle(11.5, INK, True, align='center')
smg = pstyle(10, GREY, align='center')
done = pstyle(10.5, TEAL2, True, align='center')
focus = pstyle(10.5, BRICK, True, align='center')
stage = []
names = [('M1', 'Frames\n+ masks'), ('M2', '4D Gaussian\nreconstruction'),
         ('M3', 'Trajectory\nextraction'), ('M4', 'Self-supervised\nwindows'),
         ('M5', 'Diffusion\npredictor'), ('M6', 'Inference\n+ evaluation')]
x = 1.6
for i, (tag, nm) in enumerate(names):
    is_focus = i >= 4
    stage.append(card(x, 5.2, 4.75, 3.2, tag,
                      nm.split('\n'),
                      accent=BRICK if is_focus else TEAL2,
                      fill='F3E7E4' if is_focus else PAPER, head_size=16, body_size=11.5))
    stage.append(box(x, 8.6, 4.75, 0.8,
                     [('FYP-2 focus' if is_focus else 'FYP-1 complete',
                       focus if is_focus else done)]))
    if i < 5:
        stage.append(arrow(x + 4.82, 6.8, x + 5.25, 6.8,
                           BRICK if i >= 4 else TEAL2))
    x += 5.3
slide('Pipeline', (
    title_bar('FYP-1 built the reconstruction. FYP-2 adds the prediction',
              'where we are') +
    ''.join(stage) +
    stat(1.6, 11.6, 7.1, '30.10 dB', 'reconstruction PSNR') +
    stat(9.4, 11.6, 7.1, '0.904', 'reconstruction SSIM') +
    stat(17.2, 11.6, 7.1, '57,943', 'Gaussians tracked') +
    stat(25.0, 11.6, 7.3, '7,371', 'windows built (train + val)')
), notes='One minute on FYP-1. Modules 1 to 4 are complete and validated. Everything '
         'after this slide is Modules 5 and 6.')

# ════════════════════════════════════════════════════════════════════
# 4 — literature review
# ════════════════════════════════════════════════════════════════════
th = pstyle(11.5, 'FFFFFF', True)
td = pstyle(11.5, INK)
tdm = pstyle(11.5, GREY)
tdb = pstyle(11.5, TEAL, True)
rows = [
    ('Surgical 4D reconstruction', 'EndoNeRF, LerPlane, EndoGaussian, Deform3DGS',
     'No — retrospective', '3D geometry', 'No'),
    ('Surgical video anticipation', 'TPG-VAE, SurgVista',
     'Yes', '2D pixels only', 'No'),
    ('Point cloud forecasting', 'PointRNN, MoNet, PCPNet',
     'Yes', '3D, rigid outdoor LiDAR', 'No'),
    ('Medical point diffusion', 'Xie et al. 2025 (liver)',
     'No — present state', '3D points', 'No'),
]
tbl = [rect(1.6, 4.5, 30.7, 1.25, TEAL)]
cols = [1.9, 8.4, 16.4, 22.4, 29.2]
widths = [6.3, 7.8, 5.8, 6.6, 3.0]
heads = ['Line of work', 'Representative', 'Predicts future?', 'Output space', 'Uncertainty']
for cx, cw, hd in zip(cols, widths, heads):
    tbl.append(box(cx, 4.5, cw, 1.25, [(hd, th)]))
y = 5.95
for r in rows:
    for cx, cw, val in zip(cols, widths, r):
        tbl.append(box(cx, y, cw, 1.6, [(val, tdm if val.startswith('No') else td)]))
    tbl.append(line(1.6, y + 1.6, 32.3, y + 1.6))
    y += 1.65
tbl.append(rect(1.6, y + 0.2, 30.7, 1.85, MINT))
ours = [('This work', pstyle(12, TEAL, True)), ('—', tdb), ('Yes', tdb),
        ('3D tissue geometry', tdb), ('Calibrated', tdb)]
for cx, cw, (val, st) in zip(cols, widths, ours):
    tbl.append(box(cx, y + 0.2, cw, 1.85, [(val, st)]))
gap = pstyle(13.5, INK, italic=True)
slide('Literature', (
    title_bar('Four adjacent fields. None occupies the cell we need',
              'literature review') +
    ''.join(tbl) +
    box(1.6, y + 2.5, 30.7, 1.9, [
        ('The contribution is the combination in the last row, not any single column: '
         'future 3D tissue geometry, from monocular video, with no annotation, '
         'and a calibrated confidence estimate.', gap)])
), notes='Be precise about the gap. SurgVista and TPG-VAE do predict the surgical '
         'future, but in pixel space and conditioned on instrument actions. Xie et al. '
         'use point cloud diffusion on tissue but estimate the current state, not a '
         'future one.')

# ════════════════════════════════════════════════════════════════════
# 5 — what Module 5 is
# ════════════════════════════════════════════════════════════════════
lead = pstyle(15.5, INK, space_after=0.3)
slide('Module5', (
    title_bar('Module 5 — learning how tissue moves', 'module 5') +
    box(1.6, 3.7, 30.7, 1.3, [
        ('A conditional diffusion model that learns the deformation of 512 tracked '
         'tissue points, trained entirely without clinical labels.', lead)]) +
    card(1.6, 5.4, 9.8, 4.4, 'Input', [
        '•  Last 4 frames of 512 tracked points',
        '•  How far ahead to predict (1, 2 or 3)',
        '•  A patch of random noise to shape'], TEAL2) +
    card(12.05, 5.4, 9.8, 4.4, 'Model', [
        '•  16.3 M parameters',
        '•  Temporal transformer per point',
        '•  Two-way cross-attention',
        '•  Denoises over 20 steps'], TEAL2) +
    card(22.5, 5.4, 9.8, 4.4, 'Output', [
        '•  Movement of each point',
        '•  Sample repeatedly → several',
        '    plausible futures'], TEAL2) +
    card(1.6, 10.3, 14.9, 4.4, 'Why diffusion and not direct regression', [
        'Soft tissue under contact is genuinely uncertain. A single',
        'estimate hides that. Sampling the model repeatedly gives a',
        'spread of outcomes — which Module 6 shows is a usable',
        'confidence signal.'], GOLD, fill='FBF4E6') +
    card(17.4, 10.3, 14.9, 4.4, 'Why the horizon is an input', [
        'The same 4 frames appear in training with targets 1, 2 and 3',
        'frames ahead — three different right answers. Without being',
        'told which, the task is unlearnable no matter the capacity.'], GOLD,
         fill='FBF4E6') +
    box(1.6, 15.1, 30.7, 1.4,
        [('Built from 7,371 windows across 10 reconstructed videos  —  5,910 train, '
          '1,461 validation  —  with no human annotation at any stage.',
          pstyle(13.5, TEAL, True, align='center', italic=True))], fill=MINT)
), notes='Explain the model at a level the panel can follow. The two yellow cards are '
         'the two design decisions most likely to be questioned.')

# ════════════════════════════════════════════════════════════════════
# 6 — what Module 6 is
# ════════════════════════════════════════════════════════════════════
slide('Module6', (
    title_bar('Module 6 — using the model, and measuring it honestly', 'module 6') +
    box(1.6, 3.7, 30.7, 1.3, [
        ('Module 6 trains nothing. It loads the trained model, asks it three different '
         'questions, and grades every answer against estimators that need no learning '
         'at all.', lead)]) +
    card(1.6, 5.4, 9.8, 3.9, 'Ask once', [
        'Where will tissue be 1, 2 or 3',
        'frames from now?',
        '',
        'Most reliable mode.'], TEAL2) +
    card(12.05, 5.4, 9.8, 3.9, 'Ask repeatedly', [
        'Feed each answer back in to',
        'reach 10 frames ahead.',
        '',
        'Error compounds — measured.'], TEAL2) +
    card(22.5, 5.4, 9.8, 3.9, 'Ask five times', [
        'Five possible futures from one',
        'input. How much they disagree',
        'is the confidence estimate.'], TEAL2) +
    card(1.6, 9.9, 14.9, 4.2, 'Graded against two simple rules', [
        '•  "Nothing moves"  —  the naive baseline',
        '•  "Everything continues straight"  —  constant velocity',
        '',
        'If a 16 M-parameter model cannot beat these, we say so.'], BRICK,
         fill='F3E7E4') +
    card(17.4, 9.9, 14.9, 4.2, 'Six quality tests', [
        'QT0 verifies the evaluation data is the data the model trained on,',
        'before anything is predicted. QT1–QT5 cover accuracy, rollout,',
        'diversity, smoothness and cross-scene generalisation.'], BRICK,
         fill='F3E7E4') +
    box(1.6, 14.9, 30.7, 1.5,
        [('Module 6 is not there to make the model look good. It is there to establish, '
          'with evidence, exactly where the model stands.',
          pstyle(13.5, TEAL, True, align='center', italic=True))], fill=MINT)
), notes='Stress that Module 6 is an evaluation framework, not just an inference script. '
         'QT0 as a gate is worth calling out.')

# ════════════════════════════════════════════════════════════════════
# 7 — architecture diagram (both modules)
# ════════════════════════════════════════════════════════════════════
nd = pstyle(11, INK, True, align='center')
ndw = pstyle(11, 'FFFFFF', True, align='center')
ndd = pstyle(9.5, GREY, align='center')
lab = pstyle(11.5, TEAL2, True)
lab2 = pstyle(11.5, BRICK, True)
arch = []
# Module 5 band
arch.append(rect(1.5, 3.95, 30.9, 6.1, 'F2F7F6'))
arch.append(box(1.9, 4.15, 8, 0.8, [('MODULE 5  —  TRAINING', lab)]))
m5 = [(2.1, 'Past 4 frames', '(B, 4, 512, 3)'), (7.3, 'Point encoder', 'shared MLP'),
      (12.5, 'Temporal\ntransformer', '3 layers, 8 heads'),
      (17.7, 'Cross-attention', 'both directions'),
      (22.9, 'Output head', 'predicted velocity')]
for i, (x0, nm, sub) in enumerate(m5):
    parts = nm.split('\n')
    arch.append(box(x0, 5.5, 4.4, 2.0, [(p, nd) for p in parts] + [(sub, ndd)],
                    fill='FFFFFF', stroke='solid', stroke_color=TEAL2))
    if i < len(m5) - 1:
        arch.append(arrow(x0 + 4.45, 6.5, x0 + 5.05, 6.5, TEAL2))
arch.append(box(28.1, 5.5, 4.3, 2.0, [('Trained', ndw), ('checkpoint', ndw)], fill=TEAL))
arch.append(arrow(27.35, 6.5, 28.05, 6.5, TEAL2))
# conditioning inputs
arch.append(box(12.5, 7.75, 9.6, 1.1,
                [('conditioned on:  diffusion step t   ·   horizon k ∈ {1,2,3}',
                  pstyle(10.5, GOLD, True, align='center'))], fill='FBF4E6'))
arch.append(arrow(17.3, 7.7, 17.3, 7.55, GOLD))
# Module 6 band
arch.append(rect(1.5, 10.5, 30.9, 6.35, 'FAF2F0'))
arch.append(box(1.9, 10.7, 9, 0.8, [('MODULE 6  —  INFERENCE & EVALUATION', lab2)]))
arch.append(box(2.1, 11.9, 4.4, 2.4, [('Trajectory', nd), ('replay', nd), ('+ QT0 gate', ndd)],
                fill='FFFFFF', stroke='solid', stroke_color=BRICK))
arch.append(arrow(6.55, 13.1, 7.15, 13.1, BRICK))
arch.append(box(7.2, 11.9, 4.4, 2.4, [('DDIM', nd), ('sampler', nd), ('20 steps', ndd)],
                fill='FFFFFF', stroke='solid', stroke_color=BRICK))
arch.append(arrow(11.65, 13.1, 12.25, 13.1, BRICK))
for i, (nm, sub) in enumerate([('Direct', 'k ≤ 3'), ('Rollout', '10 frames'),
                               ('5 futures', 'uncertainty')]):
    x0 = 12.3 + i * 4.3
    arch.append(box(x0, 11.9, 4.0, 2.4, [(nm, nd), (sub, ndd)],
                    fill='FFFFFF', stroke='solid', stroke_color=BRICK))
arch.append(arrow(25.3, 13.1, 25.9, 13.1, BRICK))
arch.append(box(25.95, 11.9, 6.45, 2.4,
                [('Metrics  ·  figures', ndw), ('exports for Module 7', ndw)], fill=BRICK))
arch.append(box(12.3, 14.6, 12.7, 1.1,
                [('all three compared against: nothing moves  ·  constant velocity',
                  pstyle(10.5, BRICK, True, align='center'))], fill='FFFFFF'))
arch.append(arrow(31.7, 7.6, 31.7, 11.85, GREY))
arch.append(box(26.2, 9.0, 5.1, 1.0, [('checkpoint', pstyle(10, GREY, italic=True,
                                                            align='right'))]))
slide('Architecture', title_bar('Architecture — Modules 5 and 6', 'architecture')
      + ''.join(arch),
      notes='One slide for both modules. Module 5 trains and emits a checkpoint; Module 6 '
            'consumes that checkpoint and never trains. The horizon conditioning arrow is '
            'the fix that made the objective well-posed.')

# ════════════════════════════════════════════════════════════════════
# 8, 9 — problems and solutions
# ════════════════════════════════════════════════════════════════════
ph = pstyle(12.5, BRICK, True, space_after=0.05)
pb = pstyle(11.5, INK, space_after=0.0)
sh = pstyle(12.5, TEAL, True, space_after=0.05)


def ps_rows(items, y0=4.3, row_h=2.42, gap=0.2):
    out = []
    y = y0
    for prob, probd, sol, sold in items:
        out.append(box(1.6, y, 14.1, row_h, [(prob, ph), (probd, pb)], fill='F7F2F1'))
        out.append(arrow(15.85, y + row_h / 2, 16.55, y + row_h / 2, GREY))
        out.append(box(16.7, y, 15.6, row_h, [(sol, sh), (sold, pb)], fill='F0F6F5'))
        y += row_h + gap
    return ''.join(out)


slide('Problems5', (
    title_bar('Module 5 — what went wrong, and what we did', 'problems → solutions') +
    ps_rows([
        ('The task had no single right answer',
         'One context appeared with targets 1, 2 and 3 frames ahead.',
         'Condition on the horizon',
         'Horizon embedded and injected into both branches of the network.'),
        ('The signal was smaller than the noise',
         'Tissue moves 0.004 units; diffusion adds noise at scale 1.',
         'Rescale the target to unit variance',
         'Inverted before every reported metric, so results stay in physical units.'),
        ('Raw trajectories had almost no momentum',
         'Correlation between consecutive motions was 0.077 — nearly noise.',
         'Adaptive temporal smoothing',
         'Per-video Savitzky–Golay window; correlation rose to 0.89.'),
        ('Training made the model worse after epoch 60',
         'Held-out error rose while training loss kept falling.',
         'Shorter schedule, stronger regularisation',
         '500 → 200 epochs, weight decay 0.01 → 0.05, best checkpoint kept.'),
        ('A model that gave up scored the same as doing nothing',
         'Predicting zero displacement equalled the naive baseline.',
         'Predict the correction to constant velocity',
         'Now giving up falls back to an estimator 41% better than naive.'),
    ])
), notes='Five problems, each with the measurement that revealed it. The last one is the '
         'most important design change in the project.')

slide('Problems6', (
    title_bar('Module 6 — what went wrong, and what we did', 'problems → solutions') +
    ps_rows([
        ('Evaluation could have measured the wrong data',
         'Rollout needs continuous trajectories; Module 4 stores fixed windows.',
         'Rebuild, then verify before predicting',
         'QT0 checks the rebuild against Module 4 output and runs as a gate.'),
        ('The metric was penalising the model unfairly',
         'Chamfer Distance is minimised by the average, not by one sample.',
         'Average several sampled trajectories',
         'Removes a ~40% penalty the model was being charged for free.'),
        ('Diagnostics were measuring one easy video',
         'Taking the first N windows drew almost entirely from one recording.',
         'Evenly spread the evaluation set',
         'Every video contributes; earlier scores had been 4× too optimistic.'),
        ('Our constant-acceleration baseline was wrong',
         'A backward difference gives the velocity at t − ½, not at t.',
         'Derive the velocity at t properly',
         'v = (3\u00b7x(t) − 4\u00b7x(t−1) + x(t−2)) / 2, verified exact on quadratic motion.'),
        ('What we called a Kalman filter was not one',
         'It was an alpha–beta filter with fixed gains and no covariance.',
         'Implement a real two-state Kalman filter',
         'Full 2×2 covariance; recovers a known velocity to 2.6e−06.'),
    ])
), notes='Module 6 problems are mostly about measurement integrity rather than modelling. '
         'That is the right emphasis for an evaluation module.')

# ════════════════════════════════════════════════════════════════════
# 10 — results: training
# ════════════════════════════════════════════════════════════════════
slide('ResultsTrain', (
    title_bar('Training converged cleanly — and then overfitted', 'results  1 / 4') +
    stat(1.6, 4.1, 7.1, '16.3 M', 'parameters') +
    stat(9.4, 4.1, 7.1, '500', 'epochs completed') +
    stat(17.2, 4.1, 7.1, '0.012', 'training loss, from 0.36') +
    stat(25.0, 4.1, 7.3, '~60', 'best epoch') +
    card(1.6, 7.9, 14.9, 3.5, 'What went right', [
        'Smooth convergence, no instability, no divergence.',
        'Multi-session training resumes exactly where it stopped,',
        'surviving Colab disconnection.'], TEAL2) +
    card(17.4, 7.9, 14.9, 3.5, 'What went wrong', [
        'Held-out error bottomed out near epoch 60 and rose after',
        '~260 while training loss kept falling — overfitting,',
        'measured directly rather than assumed.'], BRICK, fill='F3E7E4') +
    box(1.6, 12.2, 30.7, 3.3, [
        ('Best held-out Chamfer Distance 0.01568  versus  naive baseline 0.01557',
         pstyle(17, INK, True)),
        ('The model converged to predicting almost no motion — which is exactly what '
         'the naive baseline does. Understanding why is the rest of this presentation.',
         pstyle(13.5, GREY, italic=True))], fill=PAPER)
), notes='Be straight about this. Training works; the result does not beat the baseline. '
         'The next two slides explain why, which is the actual contribution.')

# ════════════════════════════════════════════════════════════════════
# 11 — results: comparison
# ════════════════════════════════════════════════════════════════════
chd = pstyle(12, 'FFFFFF', True, align='center')
cl = pstyle(12.5, INK)
cv = pstyle(12.5, INK, align='center')
cb = pstyle(12.5, TEAL, True, align='center')
cw = pstyle(12.5, BRICK, True, align='center')
tb = [rect(1.6, 4.5, 20.0, 1.05, TEAL)]
hdrs = [('Method', 2.0, 8.4, chd), ('k = 1', 10.6, 3.4, chd),
        ('k = 2', 14.2, 3.4, chd), ('k = 3', 17.8, 3.4, chd)]
for hd, cx, cwid, st in hdrs:
    tb.append(box(cx, 4.5, cwid, 1.05, [(hd, st if hd != 'Method' else
                                         pstyle(12, 'FFFFFF', True))]))
data = [('Nothing moves (naive)', '0.0093', '0.0165', '0.0222', False),
        ('Constant velocity', '0.0055', '0.0138', '0.0187', True),
        ('Ours — diffusion', '0.0093', '0.0165', '0.0225', False)]
yy = 5.65
for nm, a, bb, c, best in data:
    style_v = cb if best else (cw if nm.startswith('Ours') else cv)
    tb.append(box(2.0, yy, 8.4, 1.15, [(nm, cl)]))
    for cx, val in zip([10.6, 14.2, 17.8], [a, bb, c]):
        tb.append(box(cx, yy, 3.4, 1.15, [(val, style_v)]))
    tb.append(line(1.6, yy + 1.15, 21.6, yy + 1.15))
    yy += 1.2
slide('ResultsCompare', (
    title_bar('A two-line formula beats our model', 'results  2 / 4') +
    ''.join(tb) +
    box(1.6, 9.6, 20.0, 1.0, [('Chamfer Distance, lower is better · full held-out set, '
                               '10 videos', pstyle(11, GREY, italic=True))]) +
    card(22.4, 4.5, 9.9, 6.1, 'Read this honestly', [
        'Our model sits exactly on the naive',
        'baseline at every horizon.',
        '',
        'Constant velocity beats both — by',
        '41% at one frame ahead.',
        '',
        'It loses on all 10 videos.'], BRICK, fill='F3E7E4') +
    box(1.6, 11.6, 30.7, 3.4, [
        ('Rollout stays stable (growth exponent 0.61) and two explanations were tested '
         'and eliminated:', pstyle(14, INK, True)),
        ('•  Error does not grow with distance from training frames  (ρ = +0.02, '
         'p = 0.61)', pstyle(13, INK)),
        ('•  Dropped preprocessing frames do not explain it  (333 origins checked, '
         '0 drops found)', pstyle(13, INK))], fill=PAPER)
), notes='Do not soften this. We report the comparison as measured. Eliminating two '
         'explanations by measurement is what let us find the real cause.')

# ════════════════════════════════════════════════════════════════════
# 12 — the key finding
# ════════════════════════════════════════════════════════════════════
big = pstyle(23, TEAL, True, font='Cambria')
cosh = pstyle(12, GREY, True, align='center')
cosv = pstyle(20, TEAL, True, align='center', font='Cambria')
cosn = pstyle(20, BRICK, True, align='center', font='Cambria')
cells = []
for i, (k, v, good) in enumerate([('t+1', '+0.57', True), ('t+2', '+0.40', True),
                                  ('t+3', '+0.26', True), ('t+5', '−0.01', False),
                                  ('t+10', '−0.49', False)]):
    x0 = 1.6 + i * 6.2
    cells.append(box(x0, 8.9, 5.8, 3.1, [(k, cosh), (v, cosv if good else cosn)],
                     fill=PAPER if good else 'F7F2F1'))
slide('KeyFinding', (
    title_bar('The model knows which way tissue moves — not how far',
              'results  3 / 4   ·   the key finding') +
    box(1.6, 3.9, 30.7, 2.1, [
        ('Chamfer Distance alone cannot separate "learned nothing" from "learned the '
         'direction but the wrong distance". Measuring them separately can.', big)]) +
    box(1.6, 7.9, 30.7, 0.8, [('Agreement between predicted and true motion direction',
                               pstyle(12.5, GREY, True))]) +
    ''.join(cells) +
    card(1.6, 12.7, 14.9, 4.1, 'Why this matters', [
        'A model that learned nothing would read 0.00 everywhere.',
        '+0.57 with naive-level error can only mean the magnitude',
        'is wrong — a far more tractable problem.'], TEAL2) +
    card(17.4, 12.7, 14.9, 4.1, 'What follows from it', [
        '•  Predict the correction to constant velocity (Module 5)',
        '•  Calibrate the magnitude at inference (Module 6)',
        'Both implemented; evaluation is the next run.'], GOLD, fill='FBF4E6')
), notes='This is the slide to spend time on. It converts an unexplained failure into a '
         'specific, addressable one, and both fixes follow directly from it.')

# ════════════════════════════════════════════════════════════════════
# 13 — uncertainty
# ════════════════════════════════════════════════════════════════════
slide('Uncertainty', (
    title_bar('The model knows when it should not be trusted',
              'results  4 / 4   ·   our strongest result') +
    stat(1.6, 4.3, 9.9, 'r = 0.69', 'spread vs actual error') +
    stat(12.05, 4.3, 9.9, '5', 'futures sampled per input') +
    stat(22.5, 4.3, 9.8, '5×', 'uncertainty growth over 10 frames') +
    box(1.6, 8.1, 30.7, 2.6, [
        ('When the five sampled futures disagree, the prediction really is worse.',
         pstyle(18, INK, True)),
        ('Uncertainty rises smoothly with how far ahead we predict — '
         '0.00011 at one frame, 0.00055 at ten.', pstyle(13.5, GREY))], fill=PAPER) +
    card(1.6, 11.3, 14.9, 5.0, 'Why this is the result we are proudest of', [
        'A system that is confidently wrong is unusable in theatre.',
        'A system that signals "do not trust me here" is usable even',
        'when imperfect.',
        '',
        'No method in our comparison provides this.'], TEAL2) +
    card(17.4, 11.3, 14.9, 5.0, 'Stated carefully', [
        'This is correlation between predicted spread and realised',
        'error. Some of it may reflect that busy moments are both',
        'harder and more variable — we say so in the paper rather',
        'than claiming more than we measured.'], GREY, fill=PAPER)
), notes='This is independent of the accuracy ranking and is genuinely publishable. '
         'Mention the caveat before the panel raises it.')

# ════════════════════════════════════════════════════════════════════
# seeing the prediction — rendering, visualisation, forecast
# ════════════════════════════════════════════════════════════════════
slide('Visuals', (
    title_bar('Making the prediction visible', 'rendering and visualisation') +
    box(1.6, 3.9, 30.7, 1.1,
        [('A prediction moves 512 points. A picture needs all 58,000. Bridging that gap '
          'is what turns a metric into something a surgeon could look at.',
          pstyle(14, INK))]) +
    card(1.6, 5.3, 9.8, 4.2, 'From points to pixels', [
        'Predicted motion of the 512 tracked',
        'points is carried onto every Gaussian',
        'by distance-weighted interpolation,',
        'so a tracked point receives its own',
        'motion exactly.'], TEAL2) +
    card(12.05, 5.3, 9.8, 4.2, 'The renderer, re-verified', [
        'Module 2\u2019s renderer was re-implemented',
        'here, including degree-3 spherical',
        'harmonics. We checked it against the',
        'original rather than assuming they',
        'matched.'], TEAL2) +
    card(22.5, 5.3, 9.8, 4.2, 'Six panels per moment', [
        'Real frame, reconstruction ceiling,',
        'prediction, naive baseline, motion',
        'overlay and an amplified error map \u2014',
        'with image metrics over the full frame',
        'and over tissue alone.'], TEAL2) +
    card(1.6, 9.9, 14.9, 3.6, 'Predicted motion was invisible', [
        'At 1\u20132 pixels, predicted and real frames look identical.',
        'Motion is drawn as arrows coloured by agreement with truth,',
        'magnified automatically \u2014 and the factor is printed on every',
        'panel, never applied to any reported number.'], GOLD, fill='FBF4E6') +
    card(17.4, 9.9, 14.9, 3.6, 'Nothing on screen was a prediction', [
        'Every predicted frame used to be paired with a frame that',
        'already existed. The forecast now runs 12 frames past the end',
        'of the recording: the real-video panel stops and says so,',
        'while the predicted panel keeps going.'], GOLD, fill='FBF4E6') +
    stat(1.6, 13.8, 9.8, '30.04', 'reconstruction ceiling, dB') +
    stat(12.05, 13.8, 9.8, '26.79', 'our prediction, dB') +
    stat(22.5, 13.8, 9.8, '26.74', 'naive baseline, dB')
), notes='This is the demo material. The image-space gap between prediction and naive is '
         'small for the same reason the geometric one is: magnitude, not direction. '
         'Offer to play the forecast video in questions.')

# ════════════════════════════════════════════════════════════════════
# 14 — comparative research results
# ════════════════════════════════════════════════════════════════════
rh = pstyle(11.5, 'FFFFFF', True)
rv = pstyle(12, INK)
rvb = pstyle(12, TEAL, True)
rec = [rect(1.6, 5.0, 15.0, 1.0, TEAL)]
for cx, cwid, hd in [(1.95, 7.4, 'Method'), (9.6, 3.4, 'Input'), (13.2, 3.1, 'PSNR')]:
    rec.append(box(cx, 5.0, cwid, 1.0, [(hd, rh)]))
yy = 6.15
for nm, inp, val, ours in [('LerPlane (MICCAI 23)', 'stereo', '29.46', False),
                           ('EndoGaussian (24)', 'stereo', '30.25', False),
                           ('Deform3DGS (MICCAI 24)', 'stereo', '30.48', False),
                           ('Ours', 'monocular', '30.10', True)]:
    st = rvb if ours else rv
    rec.append(box(1.95, yy, 7.4, 1.1, [(nm, st)]))
    rec.append(box(9.6, yy, 3.4, 1.1, [(inp, st)]))
    rec.append(box(13.2, yy, 3.1, 1.1, [(val, st)]))
    rec.append(line(1.6, yy + 1.1, 16.6, yy + 1.1))
    yy += 1.15
slide('Comparative', (
    title_bar('How we compare with published work', 'comparative results') +
    box(1.6, 3.9, 15.0, 0.9, [('3D reconstruction  —  base paper family',
                               pstyle(13.5, TEAL2, True))]) +
    ''.join(rec) +
    box(1.6, 10.9, 15.0, 2.0, [
        ('Indicative, not controlled: published figures use stereo supervision on '
         'different sequences. We state this in the paper.',
         pstyle(11.5, GREY, italic=True))]) +
    card(1.6, 13.0, 15.0, 3.4, 'What the left table establishes', [
        'Our reconstruction sits within 0.4 dB of the stereo state of',
        'the art while using a single camera and no depth sensor.',
        '',
        'That geometry is the input Module 5 learns to predict.'],
         TEAL2, fill=MINT) +
    box(17.6, 3.9, 14.7, 0.9, [('Prediction  —  base paper family',
                                pstyle(13.5, BRICK, True))]) +
    card(17.6, 5.0, 14.7, 3.2, 'PointRNN  (Fan & Yang, 2019)', [
        'The closest published method: future point clouds from past',
        'ones, scored with Chamfer Distance. Implemented faithfully',
        'as our primary learned baseline.'], BRICK, fill='F7F2F1') +
    card(17.6, 8.5, 14.7, 3.0, 'Luo & Hu  (CVPR 2021)', [
        'First diffusion model over 3D point clouds. Our denoiser is',
        'a conditional, temporally-grounded instance of it.'], BRICK, fill='F7F2F1') +
    card(17.6, 11.8, 14.7, 3.0, 'Full comparison in progress', [
        '11 methods on one protocol: 5 classical, 5 learned, ours.',
        'Module 7 implemented; the run is the next step.'], GOLD, fill='FBF4E6')
), notes='Two base papers as the supervisor asked: Deform3DGS family for reconstruction, '
         'PointRNN and Luo & Hu for prediction. Be clear the reconstruction comparison is '
         'indicative because the datasets differ.')

# ════════════════════════════════════════════════════════════════════
# 15 — future improvements
# ════════════════════════════════════════════════════════════════════
nn = pstyle(26, 'FFFFFF', True, align='center', font='Cambria')
fh = pstyle(14, INK, True, space_after=0.08)
fb = pstyle(12.5, GREY)
items = [
    ('1', 'Run the 11-method comparison and the HPO studies',
     'Modules 7 and 8 are written and tested. Only the compute runs remain.'),
    ('2', 'Evaluate the two corrections already implemented',
     'Magnitude calibration needs no retraining; the residual target needs one retrain.'),
    ('3', 'Move to leave-one-video-out evaluation',
     'Removes the optimism of selecting and reporting on the same recordings.'),
    ('4', 'Fold the completed results into the manuscript',
     'The IEEE draft is written with the tables generated from result files.'),
    ('5', 'Submit, and extend the corpus beyond 10 recordings',
     'The overfitting we measured is consistent with a corpus this size.'),
]
fy = 4.4
fcontent = []
for num, head, sub in items:
    fcontent.append(rect(1.6, fy, 1.9, 1.9, TEAL2))
    fcontent.append(box(1.6, fy, 1.9, 1.9, [(num, nn)]))
    fcontent.append(box(3.9, fy, 28.4, 1.9, [(head, fh), (sub, fb)], fill=PAPER))
    fy += 2.12
slide('Future', title_bar('Remaining work until the final evaluation', 'what is left') + ''.join(fcontent),
      notes='Order matters: items 1 and 2 are the ones that could change the headline '
            'result, and both are already implemented.')

# ════════════════════════════════════════════════════════════════════
# 16 — summary
# ════════════════════════════════════════════════════════════════════
s1 = pstyle(30, 'FFFFFF', True, font='Cambria')
s2 = pstyle(15, MINT, space_after=0.26)
s3 = pstyle(13, 'A9BFBD', italic=True)
slide('Summary', (
    rect(0, 0, 0.55, H, TEAL2) +
    box(2.4, 2.4, 29, 1.8, [('Where Modules 5 and 6 stand', s1)]) +
    box(2.4, 5.0, 14.4, 8.0, [
        ('Delivered', pstyle(15, 'FFFFFF', True, space_after=0.2)),
        ('•  A complete prediction pipeline, no clinical labels', s2),
        ('•  A six-test evaluation framework with an integrity gate', s2),
        ('•  Calibrated uncertainty  (r = 0.69)', s2),
        ('•  Comparative study implemented  (11 methods)', s2),
        ('•  Conference paper drafted  (IEEE format)', s2)], valign='top') +
    box(17.8, 5.0, 13.6, 8.0, [
        ('Not yet achieved', pstyle(15, 'FFFFFF', True, space_after=0.2)),
        ('•  Beating the naive baseline on accuracy', s2),
        ('', s2),
        ('We localised the cause to displacement magnitude,', s3),
        ('not direction, and implemented two fixes that follow', s3),
        ('directly from that measurement. Evaluating them is', s3),
        ('the immediate next step.', s3)], valign='top') +
    line(2.4, 10.6, 31.4, 10.6, TEAL2, 0.04) +
    box(2.4, 11.0, 29, 2.0, [
        ('An honest negative result with an identified cause is worth more than '
         'an unexplained positive one.', pstyle(18, MINT, italic=True))], valign='top') +
    box(2.4, 13.6, 29, 1.2, [
        ('GASR  ·  S26-017-R-GASR  ·  Supervisor: Dr Akhtar Jamil  ·  FAST NUCES Islamabad',
         pstyle(12, '8FB3B0'))], valign='top')
), dark=True, notes='Close on the honest summary. If asked "why did you not beat the '
                    'baseline", the answer is on this slide and on the key-finding slide.')

# ════════════════════════════════════════════════════════════════════
# committee-mandated opening: goals, feedback, methodology, results
# ════════════════════════════════════════════════════════════════════
ob_h = pstyle(13, INK, True, space_after=0.05)
ob_b = pstyle(11.5, GREY)
done = pstyle(11.5, TEAL, True, align='center')
part = pstyle(11.5, GOLD, True, align='center')

def objective(x, y, w, tag, head, sub, status, scolor):
    return (box(x, y, w, 2.4, [(head, ob_h), (sub, ob_b)], fill=PAPER, valign='top') +
            box(x, y + 2.5, w, 0.8, [(status, scolor)], fill='FFFFFF'))

slide('Goals', (
    title_bar('What the project is for', 'goals and objectives') +
    box(1.6, 3.8, 30.7, 1.4,
        [('Reconstruct a surgical scene in 3D from one laparoscope camera, then '
          'anticipate how the tissue deforms next \u2014 with no depth sensor and no '
          'clinical annotation.', pstyle(14.5, INK, True))], fill=MINT) +
    objective(1.6, 5.8, 7.3, 'O1', 'O1  \u00b7  Reconstruct',
              '4D scene geometry from monocular video.', 'COMPLETE  \u00b7  FYP-1', done) +
    objective(9.6, 5.8, 7.3, 'O2', 'O2  \u00b7  Learn deformation',
              'Self-supervised model of tissue motion.', 'COMPLETE  \u00b7  MODULE 5', done) +
    objective(17.6, 5.8, 7.3, 'O3', 'O3  \u00b7  Predict ahead',
              'Future geometry at 1\u20133 frames, with confidence.',
              'IMPLEMENTED  \u00b7  MODULE 6', part) +
    objective(25.6, 5.8, 6.7, 'O4', 'O4  \u00b7  Evaluate',
              'Against baselines and the literature.', 'IN PROGRESS  \u00b7  M7\u2013M8', part) +
    box(1.6, 10.4, 30.7, 1.0, [('Research question', pstyle(13, TEAL2, True))]) +
    box(1.6, 11.4, 30.7, 4.6,
        [('Can future 3D tissue geometry be predicted from monocular surgical video '
          'alone \u2014 and can the system say how much to trust each prediction?',
          pstyle(19, INK, True, italic=True))], fill=PAPER)
), notes='One minute. Goal, four statuses, the question. Do not drift into method.')

# ── faculty feedback from the FYP-1 final evaluation ────────────────
fb_p = pstyle(12.5, BRICK, True, space_after=0.04)
fb_s = pstyle(12.5, TEAL, True, space_after=0.04)
fb_b = pstyle(11, INK)
fb_e = pstyle(10.5, GREY, italic=True)

def fb_row(y, fb, did, detail, evid):
    return (box(1.6, y, 12.6, 2.75, [(fb, fb_p)], fill='F7F2F1', valign='middle') +
            arrow(14.4, y + 1.38, 15.1, y + 1.38, GREY) +
            box(15.3, y, 17.0, 2.75, [(did, fb_s), (detail, fb_b), (evid, fb_e)],
                fill='F0F6F5', valign='top'))

slide('Feedback', (
    title_bar('Faculty feedback from FYP-1, and what we did about it',
              'addressing feedback') +
    fb_row(3.9, 'Implement a base paper for comparative analysis',
           'Two base papers implemented',
           'PointRNN (Fan & Yang) for point-cloud forecasting; Luo & Hu (CVPR 2021) '
           'for point diffusion.', 'module7_baselines.py') +
    fb_row(6.85, 'Select a few SOTA papers and implement them',
           'Eleven methods on one protocol',
           'Five classical and five learned predictors, plus ours, all scored on '
           'identical windows.', 'module7_baselines.py  ·  781 lines') +
    fb_row(9.8, 'Do a detailed analysis — hyperparameter tuning, optimisation',
           'Four tuning studies',
           'Inference cost against accuracy, context length, optimiser sensitivity, '
           'and diffusion ablations.', 'module8_hpo.py  ·  593 lines') +
    fb_row(12.75, 'Start writing a paper in IEEE or Springer format',
           'IEEE conference manuscript drafted',
           'Six pages, 22 references, tables generated directly from the result files.',
           'paper/main.tex  ·  paper/main.pdf') +
    box(1.6, 16.0, 30.7, 1.2,
        [('All four are implemented and in the repository. Two still need their '
          'compute runs, which is item 1 on the remaining-work slide.',
          pstyle(12.5, GOLD, True, italic=True))])
), notes='One minute. This is a required section. Say what was asked, what we built, '
         'and name the file. Be honest that M7 and M8 are written but not yet run.')

# ── methodology, stated so it can be judged ─────────────────────────
mh = pstyle(12.5, TEAL, True, space_after=0.05)
mb = pstyle(11.5, INK)

slide('Methodology', (
    title_bar('The methodology, finalised', 'progress  1 / 3') +
    box(1.6, 3.8, 30.7, 1.4,
        [('Given the last 4 frames of 512 tracked tissue points, predict their '
          'displacement k frames ahead, k \u2208 {1, 2, 3}. No clinical labels, anywhere.',
          pstyle(14.5, INK, True))], fill=MINT) +
    card(1.6, 5.8, 9.8, 5.9, 'Model', [
        'Conditional denoising diffusion',
        'v-prediction, cosine schedule',
        'Temporal transformer with',
        'two-way cross-attention',
        '16.3 M parameters'], TEAL2) +
    card(12.05, 5.8, 9.8, 5.9, 'What it predicts', [
        'The correction to constant-velocity',
        'extrapolation \u2014 not the raw',
        'displacement.',
        '',
        'So a model that learns nothing',
        'still beats naive by 41%.'], TEAL2) +
    card(22.5, 5.8, 9.8, 5.9, 'How it is queried', [
        'DDIM sampler, 20 steps',
        'Direct (k \u2264 3)',
        'Rollout (10 frames)',
        '5 samples \u2014 their spread is',
        'the uncertainty estimate'], TEAL2) +
    box(1.6, 12.4, 30.7, 3.4, [
        ('10 videos  \u00b7  7,371 windows  \u00b7  self-supervised throughout',
         pstyle(15, INK, True)),
        ('Targets come from the reconstruction, never from an annotator. The method is '
         'written up in full in Section III of the IEEE manuscript, every '
         'hyperparameter stated.', pstyle(12.5, GREY))], fill=PAPER)
), notes='40 seconds. Problem statement, the model, the residual target, the data line. '
         'Let the cards carry the detail -- do not read them out.')

# ── metrics and results ─────────────────────────────────────────────
slide('ResultsMetrics', (
    title_bar('Metrics chosen, and what they measured', 'progress  2 / 3') +
    box(1.6, 3.8, 15.0, 0.9, [('The three questions we measure', pstyle(13.5, TEAL2, True))]) +
    box(1.6, 4.8, 15.0, 4.6, [
        (t, pstyle(12, INK, space_after=0.2)) for t in [
            'Is the predicted shape in the right place?   \u2014  Chamfer Distance',
            'Is the motion pointing the right way?   \u2014  directional cosine',
            'Does the model know when it is unsure?   \u2014  spread vs. error',
            '',
            'We need the second one because a shape score alone cannot tell',
            '"learned nothing" from "right direction, wrong distance".']],
        fill=PAPER, valign='top') +
    box(17.6, 3.8, 14.7, 0.9, [('What the answers were', pstyle(13.5, BRICK, True))]) +
    card(17.6, 4.8, 14.7, 1.9, 'Accuracy  \u2014  not yet ahead', [
        'CD 0.01568 vs naive 0.01557. Constant',
        'velocity beats both, on all 10 videos.'], BRICK, fill='F3E7E4') +
    card(17.6, 6.9, 14.7, 1.9, 'Direction  \u2014  clearly learned', [
        'Cosine +0.57 at t+1. Nothing-learned',
        'would read 0.00. So: magnitude, not direction.'], TEAL2) +
    card(17.6, 9.0, 14.7, 1.8, 'Uncertainty  \u2014  calibrated', [
        'Spread vs realised error, r = 0.69.',
        'No baseline provides this at all.'], TEAL2) +
    box(1.6, 11.6, 30.7, 2.6, [
        ('Two explanations tested and eliminated before we concluded anything',
         pstyle(13.5, INK, True)),
        ('No drift (\u03c1 = +0.02, p = 0.61)   \u00b7   no dropped frames '
         '(333 origins checked, 0 found)', pstyle(12.5, INK))], fill=PAPER) +
    box(1.6, 14.6, 30.7, 1.8, [
        ('Six quality tests run as a suite. QT0 verifies the evaluation data is the '
         'data the model trained on, and gates everything downstream.',
         pstyle(12, GREY, italic=True))])
), notes='45 seconds. Lead with WHY the metrics -- that is what is being marked. '
         'Then the three result cards. Do not read the left panel out.')

# ── the live demonstration ──────────────────────────────────────────
dn = pstyle(22, 'FFFFFF', True, align='center', font='Cambria')
dh = pstyle(13, INK, True, space_after=0.05)
db = pstyle(11.5, GREY)

slide('DemoPlan', (
    title_bar('What you are about to see, running live', 'demonstration') +
    box(1.6, 3.8, 30.7, 1.2,
        [('Nothing in this demonstration is pre-rendered. The checkpoint is loaded and '
          'inference is run in front of you.', pstyle(14, INK, True))], fill=MINT) +
    ''.join(
        rect(1.6, 5.5 + i * 2.25, 1.9, 1.9, TEAL2) +
        box(1.6, 5.5 + i * 2.25, 1.9, 1.9, [(str(i + 1), dn)]) +
        box(3.9, 5.5 + i * 2.25, 28.4, 1.9, [(h, dh), (d, db)], fill=PAPER, valign='top')
        for i, (h, d) in enumerate([
            ('Load the trained checkpoint and verify the evaluation data',
             'QT0 runs first: the rebuilt trajectory is checked against Module 4’s '
             'saved windows before anything is predicted.'),
            ('Predict a held-out moment at t+1, t+2 and t+3',
             'DDIM sampling, 20 steps, against the naive and constant-velocity '
             'baselines on the same window.'),
            ('Render the prediction as actual video frames',
             'The predicted motion of 512 points is carried onto all 58,000 Gaussians '
             'and rasterised into an image.'),
            ('Compare side by side with the real frame',
             'Real · reconstruction ceiling · our prediction · naive baseline, '
             'with magnified motion arrows and an error map.'),
            ('Forecast past the end of the recording',
             'Twelve frames beyond the last recorded image — the only output that '
             'shows prediction rather than describing it.'),
        ]))
), notes='Show this while the notebook is starting. Say explicitly that it is live. '
         'If a cell is slow, keep talking over it -- do not watch the progress bar.')

# ── divider into the reference section ──────────────────────────────
slide('DividerRef', (
    box(0, 0, 1.1, H, [], fill=TEAL) +
    box(2.4, 6.5, 29, 2.2, [('Reference slides',
                             pstyle(34, 'FFFFFF', True, font='Cambria'))]) +
    box(2.4, 9.0, 29, 2.4,
        [('Everything beyond this point is held for questions: the architecture, each '
          'module in detail, the problems we hit and fixed, the training curve and the '
          'literature table.', pstyle(15, MINT))], valign='top')
), dark=True, notes='Do not present past this slide. Jump to whichever reference slide '
                    'a question needs.')

# ════════════════════════════════════════════════════════════════════
# deck order — one contiguous block per presenter
# ════════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════
# the base papers, compared qualitatively
# ════════════════════════════════════════════════════════════════════
bp_n = pstyle(13, TEAL, True, space_after=0.04)
bp_d = pstyle(11.5, INK)
bp_s = pstyle(12.5, BRICK, True, space_after=0.04)
bp_u = pstyle(11.5, INK)


def bp_row(y, name, does, stops, ours):
    h = 2.6
    return (box(1.6, y, 13.4, h, [(name, bp_n), (does, bp_d)],
                fill='F0F6F5', valign='middle') +
            arrow(15.2, y + h / 2, 15.9, y + h / 2, GREY) +
            box(16.1, y, 16.2, h, [(stops, bp_s), (ours, bp_u)],
                fill='F7F2F1', valign='middle'))


slide('BasePapers', (
    title_bar('What we build on, and where we go past it',
              'progress  3 / 3  \u00b7  the base papers') +
    box(1.6, 3.8, 13.4, 0.8, [('The work we build on', pstyle(12, TEAL2, True))]) +
    box(16.1, 3.8, 16.2, 0.8, [('Where it stops, and what we add',
                                pstyle(12, BRICK, True))]) +
    bp_row(4.7, 'EndoGaussian \u00b7 Deform3DGS  \u2014  surgical reconstruction',
           'Reconstruct a surgical scene in 3D at high fidelity, using a stereo camera.',
           'They only ever describe what has already happened.',
           'We reach comparable fidelity from a single camera \u2014 and then predict '
           'forward from it.') +
    bp_row(7.55, 'PointRNN  \u2014  point-cloud forecasting',
           'Predicts where a moving 3D point cloud will be next. Our closest published '
           'method, and our main learned baseline.',
           'Built for rigid outdoor scenes, and gives one answer with no confidence.',
           'Same task on deforming soft tissue, and ours says how much to trust '
           'each answer.') +
    bp_row(10.4, 'Luo & Hu (CVPR 2021)  \u2014  point-cloud diffusion',
           'The first diffusion model to generate 3D point clouds.',
           'It generates plausible shapes; it does not predict a future.',
           'Our denoiser is that idea made conditional and temporal, so it forecasts '
           'rather than invents.') +
    bp_row(13.25, 'TPG-VAE \u00b7 SurgVista  \u2014  surgical anticipation',
           'Do predict the surgical future, and do it well.',
           'In 2D pixels, and only when told in advance what the instrument will do.',
           'We predict 3D geometry, and we are not told the action.') +
    box(1.6, 16.1, 30.7, 1.2,
        [('No published method does all of it at once \u2014 predict the future, in 3D, '
          'on soft tissue, from a single camera, and say how confident it is.',
          pstyle(13, GOLD, True, italic=True))])
), notes='The comparison slide. Qualitative on purpose -- the numeric comparison needs '
         'Module 7 to run. Say what each paper gives us and what it cannot do.')

emit([
    # ══ A · the mandated 4-5 minutes, uninterrupted ══════════════════
    'Title',           # 1   10 s
    'Goals',           # 2   1 min   -- overview of goals / objectives
    'Feedback',        # 3   1 min   -- faculty feedback from FYP-1, addressed
    'Methodology',     # 4   \
    'ResultsMetrics',  # 5    > 2 min -- progress since FYP-1
    'BasePapers',      # 6   /
    'Future',          # 7   1 min   -- remaining work to the final evaluation
    # ══ B · demonstration and preliminary results ════════════════════
    'DemoPlan',        # 8   shown while the notebook loads
    'Visuals',         # 9
    'KeyFinding',      # 10
    'Uncertainty',     # 11
    'ResultsCompare',  # 12
    # ══ C · reference, for questions only ════════════════════════════
    'DividerRef',      # 13
    'Architecture',    # 14
    'Module5',         # 15
    'Problems5',       # 16
    'Module6',         # 17
    'Problems6',       # 18
    'ResultsTrain',    # 19
    'Literature',      # 20
    'Comparative',     # reconstruction PSNR table, held for questions
    'Pipeline',        # 21
    'Problem',         # 22
    'Summary',         # 23
])

# ════════════════════════════════════════════════════════════════════
FODP = f'''<?xml version="1.0" encoding="UTF-8"?>
<office:document xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
 xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0"
 xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
 xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0"
 xmlns:fo="urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0"
 xmlns:svg="urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0"
 xmlns:presentation="urn:oasis:names:tc:opendocument:xmlns:presentation:1.0"
 office:version="1.3" office:mimetype="application/vnd.oasis.opendocument.presentation">
<office:font-face-decls>
 <style:font-face style:name="Calibri" svg:font-family="Calibri"/>
 <style:font-face style:name="Cambria" svg:font-family="Cambria"/>
</office:font-face-decls>
<office:styles>
 <draw:marker draw:name="Arrow" svg:viewBox="0 0 20 30" svg:d="m10 0-10 30h20z"/>
</office:styles>
<office:automatic-styles>
 <style:page-layout style:name="PM1"><style:page-layout-properties
   fo:page-width="{W}cm" fo:page-height="{H}cm" style:print-orientation="landscape"
   fo:margin-top="0cm" fo:margin-bottom="0cm" fo:margin-left="0cm"
   fo:margin-right="0cm"/></style:page-layout>
 <style:style style:name="dpM" style:family="drawing-page"/>
 {"".join(styles)}
</office:automatic-styles>
<office:master-styles>
 <style:master-page style:name="Default" style:page-layout-name="PM1"
  draw:style-name="dpM"/>
</office:master-styles>
<office:body><office:presentation>{"".join(bodies)}</office:presentation></office:body>
</office:document>'''

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fyp2_mid_eval.fodp')
open(out, 'w', encoding='utf-8').write(FODP)
print(f'wrote {out}  ({len(bodies)} slides, {len(FODP) // 1024} KB)')
