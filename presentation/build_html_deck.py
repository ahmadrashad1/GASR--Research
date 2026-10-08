#!/usr/bin/env python3
"""
Builds the FYP-2 mid-evaluation deck as one self-contained HTML file.

Design system: "Clinical Echo" — the custom direction chosen from the style
previews. Near-black ground with a green cast (an endoscope feed at rest),
bone text, two semantic accents, and a motion-echo device on anything that
refers to the predicted future.

  coral  #D2563F   measured, ours, present, emphasis
  teal   #5FA8A0   predicted, future, confidence
  bone   #E9E6DE   neutral text

Slides keep a 1:1 mapping with the PowerPoint deck and the speaking script,
so every slide number quoted in the script still resolves.
"""
import base64, html, pathlib, random, re, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import arch_svg

HERE = pathlib.Path(__file__).parent
SKILL = pathlib.Path('/home/ahmadrashad/.claude/plugins/cache/frontend-slides'
                     '/frontend-slides/2.1.0/skills/frontend-slides')
VIEWPORT_BASE = (SKILL / 'viewport-base.css').read_text()
FONTS = (HERE / 'fonts' / 'embedded.css').read_text()
OUT = pathlib.Path('/home/ahmadrashad/Documents/FYP/GASR--Research/presentation'
                   '/FYP2_Mid_Eval_Deck.html')

e = lambda t: html.escape(str(t), quote=False)
slides = []


# ── component helpers ───────────────────────────────────────────────
def slide(cls, body, notes=''):
    n = len(slides) + 1
    note = f'\n<!-- SPEAKER NOTE (slide {n}): {notes} -->' if notes else ''
    if '<!--FLOW-->' in body:
        chrome, flow = body.split('<!--FLOW-->', 1)
        if 'class="lead' in chrome:
            cls += ' has-lead'
        body = chrome + f'<div class="content">{flow}</div>'
    slides.append(f'<section class="slide {cls}" data-n="{n}">{note}\n{body}\n</section>')


def head(kicker, title, right='Mid Evaluation · FYP-2', lead=None, echo=None):
    """Kicker row, rule, headline. `echo` names the word that gets the teal ghost."""
    t = title
    if echo:
        t = t.replace(echo, f'<em data-echo="{e(echo)}">{e(echo)}</em>')
    out = (f'<div class="kick r">{e(kicker)}</div>'
           f'<div class="kick-r r">{e(right)}</div><div class="hr-top"></div>'
           f'<h1 class="r" style="--d:.08s">{t}</h1>')
    if lead:
        out += f'<p class="lead r" style="--d:.16s">{lead}</p>'
    return out + '<!--FLOW-->'


def card(title, lines, tone='', wide=False):
    ls = ''.join(f'<li>{l}</li>' if l else '<li class="sp"></li>' for l in lines)
    return (f'<div class="card {tone}{" wide" if wide else ""}">'
            f'<h3>{title}</h3><ul>{ls}</ul></div>')


def stat(value, label, tone='teal'):
    return (f'<div class="stat {tone}"><b>{value}</b><i>{e(label)}</i></div>')


def rows(items, left_label=None, right_label=None):
    """Two-column problem→solution / paper→gap rows."""
    hdr = ''
    if left_label:
        hdr = (f'<div class="rows-hdr"><span class="lh">{e(left_label)}</span>'
               f'<span class="rh">{e(right_label)}</span></div>')
    body = ''.join(
        f'<div class="row"><div class="rl"><b>{l1}</b><span>{l2}</span></div>'
        f'<div class="rarrow"></div>'
        f'<div class="rr"><b>{r1}</b><span>{r2}</span></div></div>'
        for l1, l2, r1, r2 in items)
    return f'<div class="rows">{hdr}{body}</div>'


def steps(items):
    return '<div class="steps">' + ''.join(
        f'<div class="step"><span class="num">{i}</span>'
        f'<div><b>{h}</b><span>{d}</span></div></div>'
        for i, (h, d) in enumerate(items, 1)) + '</div>'


def table(cols, body_rows, highlight_last=False, note=None):
    th = ''.join(f'<th>{c}</th>' for c in cols)
    trs = ''
    for i, r in enumerate(body_rows):
        cls = ' class="hi"' if highlight_last and i == len(body_rows) - 1 else ''
        trs += '<tr%s>%s</tr>' % (cls, ''.join(f'<td>{c}</td>' for c in r))
    n = f'<p class="tnote">{note}</p>' if note else ''
    return f'<table class="tbl"><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>{n}'


def band(label, tone, nodes, caption=None):
    """One horizontal pipeline band for the architecture slide."""
    chips = ''
    for i, (t, s) in enumerate(nodes):
        chips += (f'<div class="node{" term" if i == len(nodes) - 1 else ""}">'
                  f'<b>{t}</b>{f"<span>{s}</span>" if s else ""}</div>')
        if i < len(nodes) - 1:
            chips += '<div class="arw"></div>'
    cap = f'<div class="bcap">{caption}</div>' if caption else ''
    return (f'<div class="band {tone}"><div class="blab">{e(label)}</div>'
            f'<div class="bflow">{chips}</div>{cap}</div>')


# ════════════════════════════════════════════════════════════════════
# A · the mandated five minutes
# ════════════════════════════════════════════════════════════════════
random.seed(11)
dots = ''.join(
    f'<i class="{"p" if i % 7 == 0 else ""}" style="left:{random.randint(1180,1850)}px;'
    f'top:{random.randint(240,980)}px"></i>' for i in range(70))

slide('title', f"""
<div class="field">{dots}</div><div class="scan"></div>
<div class="kick r">S26-017-R-GASR</div>
<div class="kick-r r">Mid Evaluation · FYP-2</div><div class="hr-top"></div>
<h1 class="big r" style="--d:.12s">From Reconstruction<br>to <em data-echo="Anticipation">Anticipation</em></h1>
<p class="lead wide r" style="--d:.3s">Predicting <b>future tissue geometry</b> from monocular
laparoscopic video — with an honest estimate of its own confidence.</p>
<div class="byline r" style="--d:.46s"><span>Ahmad Rashad &nbsp;·&nbsp; Shehroz &nbsp;·&nbsp; Humair Khan</span><br>
Supervisor — Dr Akhtar Jamil<br>Computer Science · FAST NUCES Islamabad</div>
<div class="gauge r" style="--d:.6s"><div class="glab">Reconstruct · Learn · Predict</div>
<div class="bar"><s>Module 5</s><u></u></div>
<div class="bar b2"><s>Module 6</s><u></u></div>
<div class="bar b3"><s>Modules 7–8</s><u></u></div></div>""",
      'Good morning. I am Ahmad Rashad, with Shehroz and Humair. Supervisor Dr Akhtar Jamil. '
      'A research FYP on 4D surgical reconstruction and deformation prediction.')

slide('goals', head('Goals and objectives', 'What the project is for') + f"""
<div class="banner r" style="--d:.16s">Reconstruct a surgical scene in 3D from one laparoscope
camera, then anticipate how the tissue deforms next — with no depth sensor and no clinical
annotation.</div>
<div class="objs r" style="--d:.26s">
  <div class="obj"><b>O1 · Reconstruct</b><span>4D scene geometry from monocular video.</span>
    <i class="done">Complete · FYP-1</i></div>
  <div class="obj"><b>O2 · Learn deformation</b><span>Self-supervised model of tissue motion.</span>
    <i class="done">Complete · Module 5</i></div>
  <div class="obj"><b>O3 · Predict ahead</b><span>Future geometry at 1–3 frames, with confidence.</span>
    <i class="part">Implemented · Module 6</i></div>
  <div class="obj"><b>O4 · Evaluate</b><span>Against baselines and the literature.</span>
    <i class="part">In progress · M7–M8</i></div></div>
<div class="qblock r" style="--d:.38s"><span class="qlab">Research question</span>
<p>Can future 3D tissue geometry be predicted from monocular surgical video alone — and can the
system say <em>how much to trust</em> each prediction?</p></div>""",
      'One minute. Goal, four statuses, the question. Do not drift into method.')

slide('fb', head('Addressing feedback', 'Faculty feedback from FYP-1, and what we did about it') +
      rows([
          ('Implement a base paper for comparative analysis', '',
           'Two base papers implemented',
           'PointRNN for point-cloud forecasting; Luo &amp; Hu (CVPR 2021) for point diffusion. '
           '<code>module7_baselines.py</code>'),
          ('Select a few SOTA papers and implement them', '',
           'Eleven methods on one protocol',
           'Five classical and five learned predictors, plus ours, all scored on identical '
           'windows. <code>781 lines</code>'),
          ('Do a detailed analysis — hyperparameter tuning, optimisation', '',
           'Four tuning studies',
           'Inference cost against accuracy, context length, optimiser sensitivity, diffusion '
           'ablations. <code>module8_hpo.py</code>'),
          ('Start writing a paper in IEEE or Springer format', '',
           'IEEE conference manuscript drafted',
           'Six pages, 22 references, tables generated from the result files. '
           '<code>paper/main.pdf</code>'),
      ]) + """
<div class="footnote r" style="--d:.5s">All four are implemented and in the repository. Two still
need their compute runs — item 1 on the remaining-work slide.</div>""",
      'One minute. Required section. Say the feedback, say what was built. Be honest that two '
      'still need compute runs.')

slide('meth', head('Progress 1 / 3', 'The methodology, finalised') + """
<div class="banner r" style="--d:.16s">Given the last 4 frames of 512 tracked tissue points,
predict their displacement <em>k</em> frames ahead, k ∈ {1, 2, 3}. No clinical labels, anywhere.</div>
<div class="grid3 r" style="--d:.26s">""" + card('Model', [
    'Conditional denoising diffusion', 'v-prediction, cosine schedule',
    'Temporal transformer with two-way cross-attention', '16.3 M parameters'], 'teal')
      + card('What it predicts', [
          'The <em>correction</em> to constant-velocity extrapolation — not the raw displacement.',
          '', 'So a model that learns nothing still beats naive by 41%.'], 'coral')
      + card('How it is queried', [
          'DDIM sampler, 20 steps', 'Direct (k ≤ 3)', 'Rollout (10 frames)',
          '5 samples — their spread is the uncertainty estimate'], 'teal') + """
</div>
<div class="footnote wide r" style="--d:.4s"><b>10 videos · 7,371 windows · self-supervised
throughout.</b> Targets come from the reconstruction, never from an annotator. The method is
written up in full in Section III of the IEEE manuscript, every hyperparameter stated.</div>""",
      '40 seconds. Problem statement, the model, the residual target, the data line.')

slide('metrics', head('Progress 2 / 3', 'Metrics chosen, and what they measured') + """
<div class="split r" style="--d:.2s">
  <div class="qs"><span class="sublab">The three questions we measure</span>
    <ol class="qlist">
      <li><b>Is the predicted shape in the right place?</b><span>Chamfer Distance</span></li>
      <li><b>Is the motion pointing the right way?</b><span>Directional cosine</span></li>
      <li><b>Does the model know when it is unsure?</b><span>Spread vs. realised error</span></li>
    </ol>
    <p class="why">The second exists because a shape score alone cannot tell
    <em>“learned nothing”</em> from <em>“right direction, wrong distance”</em>. They score the same.</p>
  </div>
  <div class="ans"><span class="sublab coral-l">What the answers were</span>
""" + card('Accuracy — not yet ahead',
           ['CD 0.01568 vs naive 0.01557. Constant velocity beats both, on all 10 videos.'], 'coral')
      + card('Direction — clearly learned',
             ['Cosine +0.57 at t+1. Nothing-learned would read 0.00. '
              'So: magnitude, not direction.'], 'teal')
      + card('Uncertainty — calibrated',
             ['Spread vs realised error, r = 0.69. No baseline provides this at all.'], 'teal') + """
  </div></div>
<div class="footnote wide r" style="--d:.42s"><b>Two explanations tested and eliminated before we
concluded anything.</b> No drift (ρ = +0.02, p = 0.61) · no dropped frames (333 origins checked,
0 found). Six quality tests run as a suite; QT0 gates everything downstream.</div>""",
      '45 seconds. Lead with WHY the metrics. Then the three answers. Do not recite values.')

slide('papers', head('Progress 3 / 3 · the base papers', 'What we build on, and where we go past it')
      + rows([
          ('EndoGaussian · Deform3DGS <span class="tag">reconstruction</span>',
           'Reconstruct a surgical scene in 3D at high fidelity, using a stereo camera.',
           'They only ever describe what has already happened.',
           'We reach comparable fidelity from a single camera — and then predict forward from it.'),
          ('PointRNN <span class="tag">point-cloud forecasting</span>',
           'Predicts where a moving 3D point cloud will be next. Our closest published method, '
           'and our main learned baseline.',
           'Built for rigid outdoor scenes, and gives one answer with no confidence.',
           'Same task on deforming soft tissue — and ours says how much to trust each answer.'),
          ('Luo &amp; Hu, CVPR 2021 <span class="tag">point diffusion</span>',
           'The first diffusion model to generate 3D point clouds.',
           'It generates plausible shapes; it does not predict a future.',
           'Our denoiser is that idea made conditional and temporal, so it forecasts rather '
           'than invents.'),
          ('TPG-VAE · SurgVista <span class="tag">surgical anticipation</span>',
           'Do predict the surgical future, and do it well.',
           'In 2D pixels, and only when told in advance what the instrument will do.',
           'We predict 3D geometry, and we are not told the action.'),
      ], 'The work we build on', 'Where it stops, and what we add') + """
<div class="verdict r" style="--d:.52s">No published method does all of it at once — predict the
<em>future</em>, in 3D, on soft tissue, from a single camera, and say how confident it is.</div>""",
      'The comparison slide. Qualitative on purpose — the numeric comparison needs Module 7 to run.')

slide('future', head('What is left', 'Remaining work until the final evaluation') + steps([
    ('Run the 11-method comparison and the HPO studies',
     'Modules 7 and 8 are written and tested. Only the compute runs remain.'),
    ('Evaluate the two corrections already implemented',
     'Magnitude calibration needs no retraining; the residual target needs one retrain.'),
    ('Move to leave-one-video-out evaluation',
     'Removes the optimism of selecting and reporting on the same recordings.'),
    ('Fold the completed results into the manuscript',
     'The IEEE draft is written with its tables generated from result files.'),
    ('Submit, and extend the corpus beyond 10 recordings',
     'The overfitting we measured is consistent with a corpus this size.'),
]), 'Give each item a status, not a hope. Then go straight into the demo.')

# ════════════════════════════════════════════════════════════════════
# B · demonstration
# ════════════════════════════════════════════════════════════════════
slide('demo', head('Demonstration', 'What you are about to see, running live') + """
<div class="banner live r" style="--d:.16s">Nothing in this demonstration is pre-rendered.
The checkpoint is loaded and inference is run in front of you.</div>""" + steps([
    ('Load the trained checkpoint and verify the evaluation data',
     'QT0 runs first: the rebuilt trajectory is checked against Module 4’s saved windows '
     'before anything is predicted.'),
    ('Predict a held-out moment at t+1, t+2 and t+3',
     'DDIM sampling, 20 steps, against the naive and constant-velocity baselines on the same window.'),
    ('Render the prediction as actual video frames',
     'The predicted motion of 512 points is carried onto all 58,000 Gaussians and rasterised.'),
    ('Compare side by side with the real frame',
     'Real · reconstruction ceiling · our prediction · naive baseline, with magnified motion arrows.'),
    ('Forecast past the end of the recording',
     'Twelve frames beyond the last recorded image — the only output that shows prediction '
     'rather than describing it.'),
]), 'Show this while the notebook starts. Say explicitly that it is live. Keep talking over slow cells.')

slide('visuals tight', head('Rendering and visualisation', 'Making the prediction visible',
                      lead='A prediction moves 512 points. A picture needs all 58,000. Bridging '
                           'that gap is what turns a metric into something a surgeon could look at.')
      + '<div class="grid3 r" style="--d:.26s">'
      + card('From points to pixels',
             ['Predicted motion of the 512 tracked points is carried onto every Gaussian by '
              'distance-weighted interpolation, so a tracked point receives its own motion '
              'exactly.'], 'teal')
      + card('The renderer, re-verified',
             ['Module 2’s renderer was re-implemented here, including degree-3 spherical '
              'harmonics. We checked it against the original rather than assuming they matched.'], 'teal')
      + card('Six panels per moment',
             ['Real frame, reconstruction ceiling, prediction, naive baseline, motion overlay '
              'and an amplified error map — with metrics over the full frame and tissue alone.'], 'teal')
      + '</div><div class="grid2 r" style="--d:.38s">'
      + card('Predicted motion was invisible',
             ['At 1–2 pixels, predicted and real frames look identical. Motion is drawn as arrows '
              'coloured by agreement with truth and magnified automatically — the factor is printed '
              'on every panel, never applied to any reported number.'], 'coral')
      + card('Nothing on screen was a prediction',
             ['Every predicted frame used to be paired with a frame that already existed. The '
              'forecast now runs 12 frames past the end of the recording: the real-video panel '
              'stops and says so, while the predicted panel keeps going.'], 'coral')
      + '</div><div class="stats3 r" style="--d:.5s">'
      + stat('30.04', 'reconstruction ceiling, dB', 'bone')
      + stat('26.79', 'our prediction, dB', 'coral')
      + stat('26.74', 'naive baseline, dB', 'dim') + '</div>',
      'The demo material. The image-space gap is small for the same reason the geometric one is: '
      'magnitude, not direction.')

slide('key', head('Results 3 / 4 · the key finding',
                  'The model knows which way tissue moves — not how far') + """
<p class="statement r" style="--d:.16s">Chamfer Distance alone cannot separate
<em>“learned nothing”</em> from <em>“learned the direction but the wrong distance”</em>.
Measuring them separately can.</p>
<span class="sublab r" style="--d:.24s">Agreement between predicted and true motion direction</span>
<div class="cos r" style="--d:.3s">
  <div class="c pos"><i>t+1</i><b>+0.57</b></div>
  <div class="c pos"><i>t+2</i><b>+0.40</b></div>
  <div class="c pos"><i>t+3</i><b>+0.26</b></div>
  <div class="c neg"><i>t+5</i><b>−0.01</b></div>
  <div class="c neg"><i>t+10</i><b>−0.49</b></div></div>
<div class="grid2 r" style="--d:.44s">""" + card('Why this matters', [
    'A model that learned nothing would read 0.00 everywhere. +0.57 with naive-level error can '
    'only mean the magnitude is wrong — a far more tractable problem.'], 'teal')
      + card('What follows from it', [
          '· Predict the correction to constant velocity <span class="mz">(Module 5)</span>',
          '· Calibrate the magnitude at inference <span class="mz">(Module 6)</span>',
          'Both implemented; evaluation is the next run.'], 'coral') + '</div>',
      'Spend time here. It converts an unexplained failure into a specific, addressable one.')

slide('unc', head('Results 4 / 4 · our strongest result',
                  'The model knows when it should not be trusted')
      + '<div class="stats3 r" style="--d:.18s">'
      + stat('r = 0.69', 'spread vs actual error', 'teal')
      + stat('5', 'futures sampled per input', 'bone')
      + stat('5×', 'uncertainty growth over 10 frames', 'bone') + """</div>
<p class="statement r" style="--d:.3s">When the five sampled futures disagree,
<em>the prediction really is worse</em>. Uncertainty rises smoothly with how far ahead we
predict — 0.00011 at one frame, 0.00055 at ten.</p>
<div class="grid2 r" style="--d:.42s">""" + card('Why this is the result we are proudest of', [
    'A system that is confidently wrong is unusable in theatre. A system that signals '
    '“do not trust me here” is usable even when imperfect.', '',
    'No method in our comparison provides this.'], 'teal')
      + card('Stated carefully', [
          'This is correlation between predicted spread and realised error. Some of it may '
          'reflect that busy moments are both harder and more variable — we say so in the paper '
          'rather than claiming more than we measured.'], 'dim') + '</div>',
      'Independent of the accuracy ranking, and genuinely publishable. Mention the caveat before '
      'the panel raises it.')

slide('cmp', head('Results 2 / 4', 'A two-line formula beats our model') + """
<div class="split wide-l r" style="--d:.2s"><div>""" + table(
    ['Method', 'k = 1', 'k = 2', 'k = 3'],
    [['Nothing moves <span class="mz">(naive)</span>', '0.0093', '0.0165', '0.0222'],
     ['<span class="cv">Constant velocity</span>', '<b class="cv">0.0055</b>',
      '<b class="cv">0.0138</b>', '<b class="cv">0.0187</b>'],
     ['<span class="ours">Ours — diffusion</span>', '<b class="ours">0.0093</b>',
      '<b class="ours">0.0165</b>', '<b class="ours">0.0225</b>']],
    note='Chamfer Distance, lower is better · full held-out set, 10 videos')
      + '</div>' + card('Read this honestly', [
          'Our model sits exactly on the naive baseline at every horizon.', '',
          'Constant velocity beats both — by 41% at one frame ahead.', '',
          'It loses on all 10 videos.'], 'coral') + """</div>
<div class="footnote wide r" style="--d:.4s"><b>Rollout stays stable (growth exponent 0.61)</b>
and two explanations were tested and eliminated: error does not grow with distance from training
frames (ρ = +0.02, p = 0.61); dropped preprocessing frames do not explain it (333 origins
checked, 0 found).</div>""",
      'Do not soften this. Eliminating two explanations by measurement is what let us find the cause.')

# ════════════════════════════════════════════════════════════════════
# C · reference
# ════════════════════════════════════════════════════════════════════
slide('divider', """
<div class="field">""" + dots + """</div><div class="scan"></div>
<div class="dvrule"></div>
<h1 class="big r">Reference slides</h1>
<p class="lead wide r" style="--d:.16s">Everything beyond this point is held for questions:
the architecture, each module in detail, the problems we hit and fixed, the training curve and
the literature table.</p>""", 'Do not present past this slide. Jump to what a question needs.')

slide('arch ref', head('Architecture', 'Architecture — Modules 5 and 6')
      + '<div class="archwrap r" style="--d:.18s">' + arch_svg.build() + '</div>',
      'Module 5 trains and emits a checkpoint; Module 6 consumes it and never trains. '
      'The horizon arrow into both branches is the fix that made the objective well posed.')

slide('ref', head('Module 5', 'Module 5 — learning how tissue moves',
                  lead='A conditional diffusion model that learns the deformation of 512 tracked '
                       'tissue points, trained entirely without clinical labels.')
      + '<div class="grid3 r" style="--d:.26s">'
      + card('Input', ['Last 4 frames of 512 tracked points',
                       'How far ahead to predict (1, 2 or 3)',
                       'A patch of random noise to shape'], 'teal')
      + card('Model', ['16.3 M parameters', 'Temporal transformer per point',
                       'Two-way cross-attention', 'Denoises over 20 steps'], 'teal')
      + card('Output', ['Movement of each point',
                        'Sample repeatedly → several plausible futures'], 'teal')
      + '</div><div class="grid2 r" style="--d:.38s">'
      + card('Why diffusion and not direct regression', [
          'Soft tissue under contact is genuinely uncertain. A single estimate hides that. '
          'Sampling repeatedly gives a spread of outcomes — which Module 6 shows is a usable '
          'confidence signal.'], 'coral')
      + card('Why the horizon is an input', [
          'The same 4 frames appear in training with targets 1, 2 and 3 frames ahead — three '
          'different right answers. Without being told which, the task is unlearnable no matter '
          'the capacity.'], 'coral') + """</div>
<div class="footnote wide r" style="--d:.5s">Built from 7,371 windows across 10 reconstructed
videos — 5,910 train, 1,461 validation — with no human annotation at any stage.</div>""")

slide('ref', head('Problems → solutions', 'Module 5 — what went wrong, and what we did') + rows([
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
]))

slide('ref', head('Module 6', 'Module 6 — using the model, and measuring it honestly',
                  lead='Module 6 trains nothing. It loads the trained model, asks it three '
                       'different questions, and grades every answer against estimators that '
                       'need no learning at all.')
      + '<div class="grid3 r" style="--d:.26s">'
      + card('Ask once', ['Where will tissue be 1, 2 or 3 frames from now?', '',
                          'Most reliable mode.'], 'teal')
      + card('Ask repeatedly', ['Feed each answer back in to reach 10 frames ahead.', '',
                                'Error compounds — measured.'], 'teal')
      + card('Ask five times', ['Five possible futures from one input. How much they disagree '
                                'is the confidence estimate.'], 'teal')
      + '</div><div class="grid2 r" style="--d:.38s">'
      + card('Graded against two simple rules', [
          '· “Nothing moves” — the naive baseline',
          '· “Everything continues straight” — constant velocity', '',
          'If a 16 M-parameter model cannot beat these, we say so.'], 'coral')
      + card('Six quality tests', [
          'QT0 verifies the evaluation data is the data the model trained on, before anything '
          'is predicted. QT1–QT5 cover accuracy, rollout, diversity, smoothness and cross-scene '
          'generalisation.'], 'coral') + """</div>
<div class="verdict small r" style="--d:.5s">Module 6 is not there to make the model look good.
It is there to establish, with evidence, exactly where the model stands.</div>""")

slide('ref', head('Problems → solutions', 'Module 6 — what went wrong, and what we did') + rows([
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
     'v = (3·x(t) − 4·x(t−1) + x(t−2)) / 2, verified exact on quadratic motion.'),
    ('What we called a Kalman filter was not one',
     'It was an alpha–beta filter with fixed gains and no covariance.',
     'Implement a real two-state Kalman filter',
     'Full 2×2 covariance; recovers a known velocity to 2.6e−06.'),
]))

slide('ref', head('Results 1 / 4', 'Training converged cleanly — and then overfitted')
      + '<div class="stats4 r" style="--d:.2s">'
      + stat('16.3 M', 'parameters', 'bone') + stat('500', 'epochs completed', 'bone')
      + stat('0.012', 'training loss, from 0.36', 'teal') + stat('~60', 'best epoch', 'coral')
      + '</div><div class="grid2 r" style="--d:.34s">'
      + card('What went right', [
          'Smooth convergence, no instability, no divergence. Multi-session training resumes '
          'exactly where it stopped, surviving Colab disconnection.'], 'teal')
      + card('What went wrong', [
          'Held-out error bottomed out near epoch 60 and rose after ~260 while training loss '
          'kept falling — overfitting, measured directly rather than assumed.'], 'coral') + """</div>
<div class="footnote wide r" style="--d:.46s"><b>Best held-out Chamfer Distance 0.01568 versus
naive baseline 0.01557.</b> The model converged to predicting almost no motion — which is exactly
what the naive baseline does. Understanding why is the rest of this presentation.</div>""")

slide('ref', head('Literature review', 'Four adjacent fields. None occupies the cell we need')
      + table(['Line of work', 'Representative', 'Predicts future?', 'Output space', 'Uncertainty'],
              [['Surgical 4D reconstruction', 'EndoNeRF, LerPlane, EndoGaussian, Deform3DGS',
                '<span class="no">No — retrospective</span>', '3D geometry', '<span class="no">No</span>'],
               ['Surgical video anticipation', 'TPG-VAE, SurgVista', 'Yes',
                '<span class="no">2D pixels only</span>', '<span class="no">No</span>'],
               ['Point cloud forecasting', 'PointRNN, MoNet, PCPNet', 'Yes',
                '<span class="no">3D, rigid outdoor LiDAR</span>', '<span class="no">No</span>'],
               ['Medical point diffusion', 'Xie et al. 2025 (liver)',
                '<span class="no">No — present state</span>', '3D points', '<span class="no">No</span>'],
               ['<b>This work</b>', '—', '<b>Yes</b>', '<b>3D tissue geometry</b>',
                '<b>Calibrated</b>']], highlight_last=True) + """
<div class="footnote wide r" style="--d:.4s">The contribution is the combination in the last row,
not any single column: future 3D tissue geometry, from monocular video, with no annotation, and a
calibrated confidence estimate.</div>""")

slide('ref', head('Where we are', 'FYP-1 built the reconstruction. FYP-2 adds the prediction')
      + '<div class="pipe r" style="--d:.2s">' + ''.join(
          f'<div class="pm {c}"><b>{m}</b><span>{d}</span><i>{s}</i></div>'
          + ('<div class="parw"></div>' if i < 5 else '')
          for i, (m, d, s, c) in enumerate([
              ('M1', 'Frames + masks', 'FYP-1 complete', 'done'),
              ('M2', '4D Gaussian reconstruction', 'FYP-1 complete', 'done'),
              ('M3', 'Trajectory extraction', 'FYP-1 complete', 'done'),
              ('M4', 'Self-supervised windows', 'FYP-1 complete', 'done'),
              ('M5', 'Diffusion predictor', 'FYP-2 focus', 'now'),
              ('M6', 'Inference + evaluation', 'FYP-2 focus', 'now')]))
      + '</div><div class="stats4 r" style="--d:.36s">'
      + stat('30.10 dB', 'reconstruction PSNR', 'bone') + stat('0.904', 'reconstruction SSIM', 'bone')
      + stat('57,943', 'Gaussians tracked', 'bone') + stat('7,371', 'windows built (train + val)', 'teal')
      + '</div>')

slide('ref', head('Motivation', 'The laparoscope shows the present, never the next moment') + """
<div class="split r" style="--d:.2s"><div class="prose">
<p>A surgeon operates through one camera. It gives a flat image with no depth, no 3D structure,
and no indication of how tissue will respond to the instrument about to touch it.</p>
<p>Reconstruction research has solved the first problem well. It has not addressed the second at all.</p>
<p class="em">Every published surgical 4D reconstruction system is retrospective: it models
deformation that has already been recorded.</p></div><div>"""
      + card('What exists today', ['Reconstruct the scene in 3D, for frames already captured'], 'dim')
      + card('What is missing', ['Anticipate where tissue will be a moment from now, with an '
                                 'honest indication of confidence'], 'coral') + """</div></div>
<div class="stats4 r" style="--d:.4s">""" + stat('0', 'depth sensors available', 'dim')
      + stat('1–2 px', 'tissue motion per frame', 'coral')
      + stat('0', 'deformation labels in clinical video', 'dim')
      + stat('10', 'videos reconstructed', 'bone') + '</div>')

slide('ref', head('Comparative results', 'How we compare with published work') + """
<div class="split r" style="--d:.2s"><div>
<span class="sublab">3D reconstruction — base paper family</span>""" + table(
    ['Method', 'Input', 'PSNR'],
    [['LerPlane (MICCAI 23)', 'stereo', '29.46'],
     ['EndoGaussian (24)', 'stereo', '30.25'],
     ['Deform3DGS (MICCAI 24)', 'stereo', '30.48'],
     ['<b>Ours</b>', '<b>monocular</b>', '<b>30.10</b>']], highlight_last=True,
    note='Indicative, not controlled: published figures use stereo supervision on different '
         'sequences. We state this in the paper.')
      + card('What the left table establishes', [
          'Our reconstruction sits within 0.4 dB of the stereo state of the art while using a '
          'single camera and no depth sensor. That geometry is the input Module 5 learns to '
          'predict.'], 'teal') + """</div><div>
<span class="sublab coral-l">Prediction — base paper family</span>"""
      + card('PointRNN (Fan &amp; Yang, 2019)', [
          'The closest published method: future point clouds from past ones, scored with Chamfer '
          'Distance. Implemented faithfully as our primary learned baseline.'], 'coral')
      + card('Luo &amp; Hu (CVPR 2021)', [
          'First diffusion model over 3D point clouds. Our denoiser is a conditional, '
          'temporally-grounded instance of it.'], 'coral')
      + card('Full comparison in progress', [
          '11 methods on one protocol: 5 classical, 5 learned, ours. Module 7 implemented; '
          'the run is the next step.'], 'dim') + '</div></div>')

slide('close', f"""
<div class="field">{dots}</div><div class="scan"></div>
<div class="kick r">S26-017-R-GASR</div>
<div class="kick-r r">Mid Evaluation · FYP-2</div><div class="hr-top"></div>
<h1 class="r" style="--d:.08s">Where Modules 5 and 6 stand</h1><!--FLOW-->
<div class="grid2 r" style="--d:.24s">"""
      + card('Delivered', ['· A complete prediction pipeline, no clinical labels',
                           '· A six-test evaluation framework with an integrity gate',
                           '· Calibrated uncertainty (r = 0.69)',
                           '· Comparative study implemented (11 methods)',
                           '· Conference paper drafted (IEEE format)'], 'teal')
      + card('Not yet achieved', ['· Beating the naive baseline on accuracy', '',
                                  'We localised the cause to displacement magnitude, not '
                                  'direction, and implemented two fixes that follow directly '
                                  'from that measurement. Evaluating them is the immediate '
                                  'next step.'], 'coral') + """</div>
<div class="verdict r" style="--d:.42s">An honest negative result with an identified cause is worth
more than an <em>unexplained positive one</em>.</div>
<div class="byline close-by r" style="--d:.54s">GASR · S26-017-R-GASR · Supervisor: Dr Akhtar Jamil
· FAST NUCES Islamabad</div>""",
      'Close on the honest summary. If asked why we did not beat the baseline, the answer is here '
      'and on the key-finding slide.')

# ════════════════════════════════════════════════════════════════════
CSS = r"""
/* === DESIGN TOKENS ====================================== */
:root{
  --stage-bg:#04090A; --slide-bg:#07100F;
  --bone:#E9E6DE; --dim:#6E7E7C; --dim2:#9BABA8;
  --coral:#D2563F; --teal:#5FA8A0;
  --panel:rgba(233,230,222,.035); --line:rgba(233,230,222,.13);
  --serif:'Fraunces',Georgia,'Times New Roman',serif;
  --sans:'IBM Plex Sans','Segoe UI',Helvetica,Arial,sans-serif;
  --mono:'IBM Plex Mono',Consolas,'Courier New',monospace;
}
/* === SLIDE GROUND ======================================= */
.slide{
  background:radial-gradient(130% 95% at 12% 8%,#0D1A18 0%,#07100F 58%,#050B0A 100%);
  font-family:var(--sans); color:var(--bone);
  box-sizing:border-box; padding:312px 150px 112px;
}
.slide.has-lead{padding-top:424px}
.slide.tight .card{padding:20px 24px 22px}
.slide.tight .card li{font-size:20px;line-height:1.4;margin-bottom:7px}
.slide.tight .grid3{margin-top:26px}
.slide.tight .grid2{margin-top:20px}
.slide.tight .stats3{margin-top:20px}
.slide.tight .stat{padding:15px 20px}
.slide.tight .stat b{font-size:42px}
.slide.tight .stat i{margin-top:9px;font-size:14px}
.slide.tight .lead{top:292px;font-size:25px}
.slide.ref{padding-top:296px} .slide.ref.has-lead{padding-top:408px}
.content{position:relative;width:100%;height:100%}
.content > *:first-child{margin-top:0}
.field{position:absolute;inset:0;opacity:.45;pointer-events:none}
.field i{position:absolute;width:3px;height:3px;border-radius:50%;background:var(--dim);opacity:.4}
.field i.p{background:var(--teal);opacity:.85;box-shadow:0 0 10px rgba(95,168,160,.75)}
.scan{position:absolute;left:0;right:0;height:1px;pointer-events:none;
  background:linear-gradient(90deg,transparent,rgba(95,168,160,.45),transparent)}
.slide.active .scan{animation:sc 9s linear infinite}
@keyframes sc{0%{top:-2%}100%{top:102%}}

/* === CHROME ============================================= */
.kick,.kick-r{position:absolute;top:92px;font-family:var(--mono);font-size:19px;
  letter-spacing:.3em;text-transform:uppercase}
.kick{left:150px;color:var(--coral)} .kick-r{right:150px;color:var(--dim);letter-spacing:.2em}
.hr-top{position:absolute;left:150px;right:150px;top:142px;height:1px;
  background:linear-gradient(90deg,var(--coral),rgba(210,86,63,0) 62%)}
.pager{position:absolute;right:150px;bottom:64px;font-family:var(--mono);font-size:17px;
  letter-spacing:.2em;color:var(--dim)}
.foot{position:absolute;left:150px;bottom:64px;font-family:var(--mono);font-size:16px;
  letter-spacing:.16em;color:rgba(110,126,124,.65);text-transform:uppercase}

/* === TYPE =============================================== */
h1{position:absolute;left:150px;right:150px;top:196px;margin:0;font-family:var(--serif);
  font-weight:300;font-size:70px;line-height:1.08;letter-spacing:-.02em;color:var(--bone)}
h1.big{font-size:118px;line-height:1.0;top:250px;right:400px}
h1 em{position:relative;display:inline-block;font-style:italic;font-weight:600;color:var(--coral)}
h1 em::after{content:attr(data-echo);position:absolute;left:0;top:0;color:var(--teal);
  opacity:.34;filter:blur(1.6px);transform:translate(14px,-6px)}
.slide.active h1 em::after{animation:echo 3.8s ease-in-out infinite}
@keyframes echo{0%,100%{transform:translate(13px,-5px);opacity:.26}
  50%{transform:translate(28px,-11px);opacity:.46}}
.lead{position:absolute;left:150px;right:150px;top:300px;margin:0;font-size:27px;line-height:1.5;
  color:var(--dim2);max-width:1380px}
.lead.wide{top:664px;right:760px;font-size:31px}
.lead b{color:var(--bone);font-weight:600}
.statement{margin:0 0 30px;font-size:28px;line-height:1.46;color:var(--dim2)}
.statement em{font-style:normal;color:var(--bone)}
.sublab{display:block;font-family:var(--mono);font-size:16px;letter-spacing:.24em;
  text-transform:uppercase;color:var(--teal);margin-bottom:22px}
.sublab.coral-l{color:var(--coral)}

.ref h1,.arch h1{font-size:58px;top:190px}

/* === PIECES ============================================= */
.banner{margin-top:0;padding:26px 32px;background:rgba(95,168,160,.085);
  border-left:3px solid var(--teal);font-size:25px;line-height:1.45;color:var(--bone)}
.banner em{font-style:italic;color:var(--teal)}
.banner.live{border-left-color:var(--coral);background:rgba(210,86,63,.09)}

.card{background:var(--panel);border-top:2px solid var(--line);padding:26px 28px 30px}
.card h3{margin:0 0 14px;font-family:var(--mono);font-size:17px;letter-spacing:.17em;
  text-transform:uppercase;color:var(--bone);font-weight:400}
.card ul{margin:0;padding:0;list-style:none}
.card li{font-size:21px;line-height:1.46;color:var(--dim2);margin-bottom:9px}
.card li.sp{height:12px;margin:0}
.card li em{font-style:normal;color:var(--bone);font-weight:600}
.card.teal{border-top-color:var(--teal)} .card.teal h3{color:var(--teal)}
.card.coral{border-top-color:var(--coral)} .card.coral h3{color:var(--coral)}
.card.dim{border-top-color:var(--dim)} .card.dim h3{color:var(--dim2)}

.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:26px;margin-top:34px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:26px;margin-top:26px}
.split{display:grid;grid-template-columns:1fr 1fr;gap:46px;margin-top:36px}
.split.wide-l{grid-template-columns:1.52fr 1fr}
.split .card{margin-bottom:22px}

.stats3{display:grid;grid-template-columns:repeat(3,1fr);gap:26px;margin-top:28px}
.stats4{display:grid;grid-template-columns:repeat(4,1fr);gap:24px;margin-top:38px}
.stat{background:var(--panel);padding:24px 26px;border-top:2px solid var(--line);text-align:center}
.stat b{display:block;font-family:var(--serif);font-weight:300;font-size:56px;line-height:1;
  color:var(--bone)}
.stat i{display:block;font-style:normal;font-family:var(--mono);font-size:15px;letter-spacing:.14em;
  text-transform:uppercase;color:var(--dim);margin-top:14px}
.stat.teal b{color:var(--teal)}.stat.teal{border-top-color:var(--teal)}
.stat.coral b{color:var(--coral)}.stat.coral{border-top-color:var(--coral)}
.stat.dim b{color:var(--dim2)}

.footnote{margin-top:30px;font-size:21px;line-height:1.5;color:var(--dim)}
.footnote b{color:var(--bone);font-weight:600}
.footnote.wide{padding:22px 0 0;border-top:1px solid var(--line)}

.verdict{margin-top:34px;padding:22px 30px;background:rgba(210,86,63,.08);
  border-left:3px solid var(--coral);font-family:var(--serif);font-size:27px;font-style:italic;
  font-weight:300;line-height:1.4;color:var(--bone)}
.verdict em{font-style:italic;color:var(--coral);font-weight:600}
.verdict.small{font-size:23px;margin-top:26px}

/* rows: left claim → right answer */
.rows{margin-top:36px}
.rows-hdr{display:grid;grid-template-columns:1fr 54px 1.2fr;margin-bottom:14px}
.rows-hdr span{font-family:var(--mono);font-size:15px;letter-spacing:.22em;text-transform:uppercase}
.rows-hdr .lh{color:var(--teal)} .rows-hdr .rh{color:var(--coral);grid-column:3}
.row{display:grid;grid-template-columns:1fr 54px 1.2fr;align-items:center;margin-bottom:14px}
.rl,.rr{padding:18px 24px;background:var(--panel)}
.rl{border-left:2px solid rgba(95,168,160,.5)}
.rr{border-left:2px solid rgba(210,86,63,.55)}
.rl b,.rr b{display:block;font-size:21px;font-weight:600;margin-bottom:5px}
.rl b{color:var(--teal)} .rr b{color:var(--coral)}
.rl span,.rr span{display:block;font-size:19px;line-height:1.42;color:var(--dim2)}
.rl b .tag{display:inline-block;font-family:var(--mono);font-size:13px;letter-spacing:.14em;
  text-transform:uppercase;color:var(--dim);margin-left:10px;font-weight:400}
.rarrow{position:relative;height:2px;background:rgba(233,230,222,.18)}
.rarrow::after{content:'';position:absolute;right:0;top:-4px;border-left:9px solid rgba(233,230,222,.4);
  border-top:5px solid transparent;border-bottom:5px solid transparent}
code{font-family:var(--mono);font-size:16px;color:var(--dim);letter-spacing:.04em}

/* numbered steps */
.steps{margin-top:34px}
.step{display:grid;grid-template-columns:78px 1fr;gap:26px;align-items:center;
  padding:17px 0;border-bottom:1px solid var(--line)}
.step:last-child{border-bottom:0}
.step .num{display:flex;align-items:center;justify-content:center;width:62px;height:62px;
  font-family:var(--serif);font-size:30px;color:var(--teal);border:1px solid rgba(95,168,160,.4)}
.step b{display:block;font-size:24px;font-weight:600;color:var(--bone);margin-bottom:5px}
.step span{display:block;font-size:19px;line-height:1.42;color:var(--dim)}

/* objectives */
.objs{display:grid;grid-template-columns:repeat(4,1fr);gap:24px;margin-top:36px}
.obj{background:var(--panel);border-top:2px solid var(--line);padding:24px 24px 20px}
.obj b{display:block;font-size:23px;color:var(--bone);margin-bottom:10px}
.obj span{display:block;font-size:19px;line-height:1.42;color:var(--dim2);min-height:80px}
.obj i{display:block;font-style:normal;font-family:var(--mono);font-size:14px;letter-spacing:.14em;
  text-transform:uppercase;padding-top:14px;border-top:1px solid var(--line)}
.obj i.done{color:var(--teal)} .obj i.part{color:var(--coral)}
.qblock{margin-top:44px;padding-top:26px;border-top:1px solid var(--line)}
.qlab{font-family:var(--mono);font-size:16px;letter-spacing:.24em;text-transform:uppercase;
  color:var(--coral)}
.qblock p{margin:18px 0 0;font-family:var(--serif);font-weight:300;font-size:40px;line-height:1.26;
  color:var(--bone)}
.qblock em{font-style:italic;font-weight:600;color:var(--teal)}

/* the three measured questions (slide 5) */
.qlist{margin:0;padding:0;counter-reset:q;list-style:none}
.qlist li{position:relative;padding:0 0 26px 58px;counter-increment:q}
.qlist li::before{content:counter(q);position:absolute;left:0;top:-2px;
  font-family:var(--serif);font-weight:300;font-size:32px;color:var(--teal);line-height:1}
.qlist b{display:block;font-size:24px;font-weight:600;color:var(--bone);margin-bottom:7px;
  line-height:1.3}
.qlist span{display:block;font-family:var(--mono);font-size:15px;letter-spacing:.16em;
  text-transform:uppercase;color:var(--dim)}
.why{margin:18px 0 0;padding-top:24px;border-top:1px solid var(--line);font-size:21px;
  line-height:1.52;color:var(--dim2)}
.why em{font-style:italic;color:var(--bone)}

/* cosine tiles */
.cos{display:grid;grid-template-columns:repeat(5,1fr);gap:22px;margin-top:8px}
.cos .c{background:var(--panel);border-top:2px solid var(--line);padding:22px;text-align:center}
.cos .c i{display:block;font-style:normal;font-family:var(--mono);font-size:16px;letter-spacing:.2em;
  color:var(--dim);margin-bottom:12px}
.cos .c b{font-family:var(--serif);font-weight:300;font-size:50px;line-height:1}
.cos .c.pos b{color:var(--teal)} .cos .c.pos{border-top-color:var(--teal)}
.cos .c.neg b{color:var(--coral)} .cos .c.neg{border-top-color:rgba(210,86,63,.5)}

/* tables */
.tbl{width:100%;border-collapse:collapse;margin-top:6px;font-size:20px}
.tbl th{font-family:var(--mono);font-size:15px;letter-spacing:.17em;text-transform:uppercase;
  color:var(--dim);text-align:left;padding:14px 18px;border-bottom:1px solid var(--line);font-weight:400}
.tbl td{padding:19px 18px;border-bottom:1px solid rgba(233,230,222,.07);color:var(--dim2)}
.tbl tr.hi td{background:rgba(95,168,160,.08);color:var(--bone)}
.tbl .ours{color:var(--coral);font-weight:600} .tbl .cv{color:var(--teal);font-weight:600}
.tbl .no{color:rgba(110,126,124,.8)} .tbl b{color:var(--bone)}
.tnote{margin:14px 0 0;font-size:17px;font-style:italic;color:var(--dim)}
.mz{color:var(--dim);font-weight:400}

/* architecture */
.archwrap{margin-top:14px;width:100%;display:flex;justify-content:center}
.archsvg{display:block;width:100%;height:auto;max-height:706px}
/* the architecture figure is the one slide that needs the extra vertical room */
.slide.arch{padding-top:246px}
.slide.arch h1{top:162px}
.arch{margin-top:34px}
.band{position:relative;padding:30px 30px 26px;margin-bottom:0;background:rgba(95,168,160,.055)}
.band.m6{background:rgba(210,86,63,.06)}
.blab{font-family:var(--mono);font-size:15px;letter-spacing:.22em;text-transform:uppercase;
  color:var(--teal);margin-bottom:18px}
.band.m6 .blab{color:var(--coral)}
.bflow{display:flex;align-items:stretch;gap:0}
.node{flex:1;background:rgba(7,16,15,.72);border:1px solid var(--line);padding:24px 16px;
  text-align:center;display:flex;flex-direction:column;justify-content:center}
.node b{display:block;font-size:19px;color:var(--bone);line-height:1.25}
.node span{display:block;font-family:var(--mono);font-size:14px;color:var(--dim);margin-top:6px}
.node.term{background:rgba(95,168,160,.2);border-color:var(--teal)}
.band.m6 .node.term{background:rgba(210,86,63,.22);border-color:var(--coral)}
.arw{width:34px;flex:none;position:relative}
.arw::after{content:'';position:absolute;left:6px;right:6px;top:50%;height:1px;
  background:rgba(233,230,222,.3)}
.arw::before{content:'';position:absolute;right:6px;top:calc(50% - 4px);
  border-left:8px solid rgba(233,230,222,.45);border-top:4px solid transparent;
  border-bottom:4px solid transparent;z-index:1}
.bcap{margin-top:16px;font-family:var(--mono);font-size:15px;letter-spacing:.1em;color:var(--dim);
  text-align:center}
.bcap em{font-style:italic;color:var(--teal)}
.ckpt{height:86px;display:flex;align-items:center;justify-content:flex-end;padding-right:120px}
.ckpt span{font-family:var(--mono);font-size:15px;letter-spacing:.2em;text-transform:uppercase;
  color:var(--dim);position:relative;padding-left:40px}
.ckpt span::before{content:'';position:absolute;left:0;top:50%;width:30px;height:1px;
  background:var(--teal)}

/* pipeline strip */
.pipe{display:flex;align-items:stretch;margin-top:34px}
.pm{flex:1;background:var(--panel);border-top:2px solid var(--line);padding:22px 18px 18px;
  text-align:center;display:flex;flex-direction:column}
.pm b{display:block;font-family:var(--serif);font-size:30px;font-weight:300;color:var(--bone)}
.pm span{display:block;font-size:18px;line-height:1.35;color:var(--dim2);margin:10px 0 14px;flex:1}
.pm i{display:block;font-style:normal;font-family:var(--mono);font-size:13px;letter-spacing:.13em;
  text-transform:uppercase;color:var(--dim)}
.pm.now{border-top-color:var(--coral);background:rgba(210,86,63,.07)}
.pm.now b{color:var(--coral)} .pm.now i{color:var(--coral)}
.pm.done{border-top-color:rgba(95,168,160,.45)}
.parw{width:28px;flex:none;position:relative}
.parw::after{content:'';position:absolute;left:5px;right:5px;top:50%;height:1px;
  background:rgba(233,230,222,.2)}

/* prose / byline / title-slide gauge */
.prose p{margin:0 0 22px;font-size:23px;line-height:1.52;color:var(--dim2)}
.prose p.em{font-family:var(--serif);font-style:italic;font-size:24px;color:var(--bone)}
.byline{position:absolute;left:150px;top:868px;font-family:var(--mono);font-size:18px;
  line-height:2.1;letter-spacing:.055em;color:var(--dim)}
.byline span{color:var(--bone)}
.byline.close-by{position:relative;left:0;top:auto;bottom:auto;margin-top:46px;font-size:16px;line-height:1.6}
.gauge{position:absolute;right:150px;bottom:148px;width:430px;text-align:right}
.glab{font-family:var(--mono);font-size:15px;letter-spacing:.26em;text-transform:uppercase;
  color:var(--coral);margin-bottom:46px}
.bar{height:3px;background:rgba(233,230,222,.13);position:relative;margin-bottom:46px}
.bar u{position:absolute;left:0;top:0;height:100%;background:var(--teal);display:block;
  transform:scaleX(0);transform-origin:left}
.slide.active .bar u{animation:fill 1.4s cubic-bezier(.76,0,.24,1) forwards}
.bar.b2 u{background:var(--coral);animation-delay:.18s}
.bar.b3 u{background:rgba(233,230,222,.4);animation-delay:.36s}
@keyframes fill{to{transform:scaleX(1)}}
.bar s{position:absolute;right:0;top:-25px;font-family:var(--mono);font-size:14px;
  letter-spacing:.2em;text-transform:uppercase;text-decoration:none;color:var(--dim)}
.dvrule{position:absolute;left:0;top:0;bottom:0;width:10px;background:var(--teal)}
.divider h1{top:420px}
.close h1{top:206px}
.divider .lead{top:600px;right:420px}

/* === REVEAL ============================================= */
.r{opacity:0}
.slide.active .r{animation:up .85s cubic-bezier(.22,1,.36,1) forwards;animation-delay:var(--d,0s)}
@keyframes up{from{opacity:0;transform:translateY(22px)}to{opacity:1;transform:none}}

/* === DECK CHROME (outside the stage) ==================== */
.deck-controls{display:flex;align-items:center;gap:16px;background:rgba(7,16,15,.82);
  border:1px solid rgba(233,230,222,.14);padding:9px 16px;border-radius:2px;
  font-family:var(--mono);font-size:12px;letter-spacing:.14em;color:#8FA09D;
  opacity:0;transition:opacity .3s}
.deck-viewport:hover .deck-controls{opacity:1}
.deck-controls button{background:none;border:0;color:#8FA09D;font:inherit;cursor:pointer;padding:2px 6px}
.deck-controls button:hover{color:#E9E6DE}
#bar{position:fixed;left:0;top:0;height:2px;background:var(--teal);z-index:1001;transition:width .3s}
"""

JS = r"""
/* === FIXED-STAGE SCALING ================================ */
function fitStage(){
  var s = Math.min(innerWidth/1920, innerHeight/1080);
  var st = document.querySelector('.deck-stage');
  st.style.transform='translate('+((innerWidth-1920*s)/2)+'px,'+((innerHeight-1080*s)/2)+'px) scale('+s+')';
}
addEventListener('resize', fitStage); fitStage();

/* === NAVIGATION ========================================= */
var slides=[].slice.call(document.querySelectorAll('.slide')), i=0;
var bar=document.getElementById('bar'), cnt=document.getElementById('cnt');
function show(n){
  i=Math.max(0,Math.min(slides.length-1,n));
  slides.forEach(function(s,k){ s.classList.toggle('active', k===i); });
  bar.style.width=((i+1)/slides.length*100)+'%';
  cnt.textContent=(i+1)+' / '+slides.length;
  location.hash=i+1;
}
addEventListener('keydown',function(ev){
  var k=ev.key;
  if(k==='ArrowRight'||k==='ArrowDown'||k===' '||k==='PageDown'){ev.preventDefault();show(i+1);}
  else if(k==='ArrowLeft'||k==='ArrowUp'||k==='PageUp'){ev.preventDefault();show(i-1);}
  else if(k==='Home'){show(0);} else if(k==='End'){show(slides.length-1);}
  else if(k==='f'||k==='F'){document.documentElement.requestFullscreen&&
    (document.fullscreenElement?document.exitFullscreen():document.documentElement.requestFullscreen());}
});
var wheelLock=0;
addEventListener('wheel',function(ev){
  var now=Date.now(); if(now-wheelLock<420) return; wheelLock=now;
  show(i + (ev.deltaY>0?1:-1));
},{passive:true});
document.querySelector('.deck-viewport').addEventListener('click',function(ev){
  if(ev.target.closest('.deck-controls')) return;
  show(i + (ev.clientX > innerWidth*0.35 ? 1 : -1));
});
/* touch */
var tx=null;
addEventListener('touchstart',function(e){tx=e.changedTouches[0].clientX;},{passive:true});
addEventListener('touchend',function(e){
  if(tx===null) return; var dx=e.changedTouches[0].clientX-tx;
  if(Math.abs(dx)>50) show(i + (dx<0?1:-1)); tx=null;
},{passive:true});

show(parseInt((location.hash||'#1').slice(1),10)-1 || 0);
"""

pager = ''.join('')
body = []
for idx, s in enumerate(slides, 1):
    chrome = ('' if 'class="slide title' in s or 'class="slide divider' in s
              or 'class="slide close' in s
              else f'<div class="foot">GASR · S26-017-R-GASR</div>'
                   f'<div class="pager">{idx:02d}</div>')
    body.append(s.replace('</section>', chrome + '</section>'))

HTML = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>From Reconstruction to Anticipation — FYP-2 Mid Evaluation</title>
<style>
/* === EMBEDDED FONTS (offline-safe: no network needed) === */
{FONTS}
/* === MANDATORY FIXED-STAGE BASE ========================= */
{VIEWPORT_BASE}
{CSS}
</style></head>
<body>
<div id="bar"></div>
<div class="deck-viewport"><div class="deck-stage">
{chr(10).join(body)}
</div>
<nav class="deck-controls"><button id="prev">&larr;</button><span id="cnt">1 / {len(slides)}</span>
<button id="next">&rarr;</button><span>· arrows / space · F fullscreen</span></nav>
</div>
<script>
{JS}
document.getElementById('prev').onclick=function(e){{e.stopPropagation();show(i-1);}};
document.getElementById('next').onclick=function(e){{e.stopPropagation();show(i+1);}};
</script>
</body></html>"""

OUT.write_text(HTML, encoding='utf-8')
print(f'wrote {OUT}')
print(f'  {len(slides)} slides, {len(HTML)//1024} KB (fonts embedded, fully offline)')
