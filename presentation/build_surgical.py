#!/usr/bin/env python3
"""
FYP-2 mid-evaluation deck — "Operating Theatre" theme.

Operating theatres are green because green is the complement of blood red: it
steadies the eye and makes tissue read clearly. The same logic drives this
deck. A deep surgical-green ground pushes the maroon laparoscopic imagery
forward, and the three accents are taken from the footage itself — drape
green, tissue red, and the gold of the motion arrows in our own figures.

Twelve slides, in the FYP committee's required order. Everything is embedded:
fonts, figures and photographs. The file opens with no network at all.
"""
import base64, html, pathlib, sys

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
import arch_svg

SKILL = pathlib.Path('/home/ahmadrashad/.claude/plugins/cache/frontend-slides'
                     '/frontend-slides/2.1.0/skills/frontend-slides')
VIEWPORT_BASE = (SKILL / 'viewport-base.css').read_text()
FONTS = (HERE / 'fonts' / 'surgical.css').read_text()
PRES = pathlib.Path('/home/ahmadrashad/Documents/FYP/GASR--Research/presentation')
IMG = PRES / 'images' / 'crop'
OUT = PRES / 'FYP2_Mid_Eval_Deck.html'

e = lambda t: html.escape(str(t), quote=False)
slides = []


def data_uri(name):
    mime = 'image/jpeg' if name.endswith(('.jpg', '.jpeg')) else 'image/png'
    return f'data:{mime};base64,' + base64.b64encode((IMG / name).read_bytes()).decode()


def fig(name, cls='', cap=None):
    c = f'<figcaption>{cap}</figcaption>' if cap else ''
    return (f'<figure class="fig {cls}"><img src="{data_uri(name)}" alt="">{c}</figure>')


# ── components ──────────────────────────────────────────────────────
def slide(cls, body, notes=''):
    n = len(slides) + 1
    note = f'\n<!-- SPEAKER NOTE (slide {n}): {notes} -->' if notes else ''
    if '<!--FLOW-->' in body:
        chrome, flow = body.split('<!--FLOW-->', 1)
        if 'class="lead' in chrome:
            cls += ' has-lead'
        body = chrome + f'<div class="content">{flow}</div>'
    slides.append(f'<section class="slide {cls}" data-n="{n}">{note}\n{body}\n</section>')


def head(kicker, title, right='Mid Evaluation · FYP-2', lead=None):
    out = (f'<div class="kick r">{e(kicker)}</div>'
           f'<div class="kick-r r">{e(right)}</div><div class="hr-top"></div>'
           f'<h1 class="r" style="--d:.07s">{title}</h1>')
    if lead:
        out += f'<p class="lead r" style="--d:.14s">{lead}</p>'
    return out + '<!--FLOW-->'


def card(title, lines, tone='', dense=False):
    ls = ''.join(f'<li>{l}</li>' for l in lines)
    return f'<div class="card {tone}{" dense" if dense else ""}"><h3>{title}</h3><ul>{ls}</ul></div>'


def stat(v, label, tone='green'):
    return f'<div class="stat {tone}"><b>{v}</b><i>{e(label)}</i></div>'


def rows(items, lh=None, rh=None):
    hdr = (f'<div class="rows-hdr"><span class="lh">{e(lh)}</span>'
           f'<span class="rh">{e(rh)}</span></div>') if lh else ''
    return '<div class="rows">' + hdr + ''.join(
        f'<div class="row"><div class="rl"><b>{a}</b>{f"<span>{b}</span>" if b else ""}</div>'
        f'<div class="rarrow"></div>'
        f'<div class="rr"><b>{c}</b><span>{d}</span></div></div>'
        for a, b, c, d in items) + '</div>'


def steps(items):
    return '<div class="steps">' + ''.join(
        f'<div class="step"><span class="num">{i}</span><div><b>{h}</b>'
        f'{f"<span>{d}</span>" if d else ""}</div></div>'
        for i, (h, d) in enumerate(items, 1)) + '</div>'


# ════════════════════════════════════════════════════════════════════
# 1 · title
# ════════════════════════════════════════════════════════════════════
slide('title', f"""
<div class="kick r">S26-017-R-GASR</div>
<div class="kick-r r">Mid Evaluation · FYP-2</div><div class="hr-top"></div>
<h1 class="big r" style="--d:.1s">From Reconstruction<br>to <em>Anticipation</em></h1>
<p class="lead wide r" style="--d:.26s">Predicting <b>future tissue geometry</b> from monocular
laparoscopic video — with an honest estimate of its own confidence.</p>
<div class="byline r" style="--d:.4s"><span>Ahmad Rashad &nbsp;·&nbsp; Shehroz &nbsp;·&nbsp;
Humair Khan</span><br>Supervisor — Dr Akhtar Jamil · Computer Science, FAST NUCES Islamabad</div>
<div class="hero"><img src="{data_uri('hero.jpg')}" alt=""><div class="hero-fade"></div></div>""",
      'Good morning. Ahmad Rashad, with Shehroz and Humair. Supervisor Dr Akhtar Jamil. '
      'A research FYP on 4D surgical reconstruction and deformation prediction.')

# ════════════════════════════════════════════════════════════════════
# 2 · goals
# ════════════════════════════════════════════════════════════════════
slide('goals', head('Goals and objectives', 'What the project is for',
                    lead='Reconstruct a surgical scene in 3D from one laparoscope camera — then '
                         'anticipate how the tissue deforms next.') + """
<div class="objs r" style="--d:.2s">
  <div class="obj"><b>Reconstruct</b><span>4D geometry from monocular video</span>
    <i class="ok">Complete · FYP-1</i></div>
  <div class="obj"><b>Learn deformation</b><span>Self-supervised, no labels</span>
    <i class="ok">Complete · Module 5</i></div>
  <div class="obj"><b>Predict ahead</b><span>1–3 frames, with confidence</span>
    <i class="part">Implemented · Module 6</i></div>
  <div class="obj"><b>Evaluate</b><span>Against baselines and literature</span>
    <i class="part">In progress · M7–M8</i></div></div>
<div class="qblock r" style="--d:.3s"><span class="qlab">Research question</span>
<p>Can future tissue geometry be predicted from ordinary surgical video — and can the system say
<em>how much to trust</em> each prediction?</p></div>""",
      'One minute. Goal, four statuses, the question. Do not drift into method.')

# ════════════════════════════════════════════════════════════════════
# 3 · faculty feedback
# ════════════════════════════════════════════════════════════════════
slide('fb', head('Addressing feedback', 'Faculty feedback from FYP-1, and what we did')
      + rows([
          ('Implement a base paper for comparative analysis', '',
           'Two base papers implemented',
           'PointRNN and Luo &amp; Hu <code>module7_baselines.py</code>'),
          ('Select a few SOTA papers and implement them', '',
           'Eleven methods, one protocol',
           'Five classical, five learned, plus ours <code>781 lines</code>'),
          ('Detailed analysis — tuning, optimisation', '',
           'Four tuning studies',
           'Cost/accuracy, context length, optimiser, ablations <code>module8_hpo.py</code>'),
          ('Start a paper in IEEE or Springer format', '',
           'IEEE manuscript drafted',
           'Six pages, 22 references <code>paper/main.pdf</code>'),
      ]) + """
<div class="note-bar r" style="--d:.42s">All four are implemented and in the repository. Two still
need their compute runs — item 1 on the remaining-work slide.</div>""",
      'One minute. Required section. Say the feedback, say what was built. Be honest that two '
      'still need compute runs.')

# ════════════════════════════════════════════════════════════════════
# 4 · architecture  (the figure, kept)
# ════════════════════════════════════════════════════════════════════
slide('arch', head('Progress 1 / 3 · methodology', 'Architecture — Modules 5 and 6',
                   lead='Given the last 4 frames of 512 tracked points, predict their displacement '
                        '1–3 frames ahead — trained on 10 videos, no clinical labels.')
      + '<div class="archwrap r" style="--d:.18s">' + arch_svg.build() + '</div>',
      'The figure is the methodology. Trace it bottom to top, left plate first. The orange block '
      'between the columns feeds both branches — that is the fix that made the task learnable.')

# ════════════════════════════════════════════════════════════════════
# 5 · what we measured
# ════════════════════════════════════════════════════════════════════
slide('measured', head('Progress 2 / 3', 'Metrics chosen, and what they measured')
      + fig('qt_top.png', 'wide')
      + '<div class="three r" style="--d:.3s">'
      + card('Accuracy — not yet ahead',
             ['Constant velocity beats us at every horizon, on all ten videos.'], 'tissue', dense=True)
      + card('Direction — clearly learned',
             ['Cosine +0.57 at t+1. A model that learned nothing reads 0.00.'], 'green', dense=True)
      + card('Uncertainty — calibrated',
             ['Spread across five futures tracks realised error, r = 0.69.'], 'green', dense=True)
      + '</div>',
      '45 seconds. Lead with why the metrics, then the three answers. Do not recite values.')

# ════════════════════════════════════════════════════════════════════
# 6 · base papers
# ════════════════════════════════════════════════════════════════════
slide('papers', head('Progress 3 / 3 · the base papers', 'What we build on, and where we go past it')
      + rows([
          ('EndoGaussian · Deform3DGS', 'surgical reconstruction',
           'Only ever describe what already happened',
           'We match them from one camera, then predict forward.'),
          ('PointRNN', 'point-cloud forecasting · our main learned baseline',
           'Rigid outdoor scenes, one answer, no confidence',
           'Deforming soft tissue — and ours says how much to trust it.'),
          ('Luo &amp; Hu, CVPR 2021', 'the first point-cloud diffusion model',
           'Generates shapes; does not forecast',
           'Ours is that idea made conditional on time.'),
          ('TPG-VAE · SurgVista', 'surgical anticipation',
           'Flat pixels, and told the instrument’s action',
           'We predict 3D geometry, and we are not told the action.'),
      ], 'The work we build on', 'Where it stops  →  what we add') + """
<div class="verdict r" style="--d:.46s">No published method does all of it at once — predict the
<em>future</em>, in 3D, on soft tissue, from a single camera, and say how confident it is.</div>""",
      'Qualitative on purpose: the numeric prediction comparison needs Module 7 to run.')

# ════════════════════════════════════════════════════════════════════
# 7 · remaining work
# ════════════════════════════════════════════════════════════════════
slide('future', head('What is left', 'Remaining work until the final evaluation') + steps([
    ('Run the 11-method comparison and the HPO studies',
     'Both written and tested. What remains is compute, not code.'),
    ('Evaluate the two corrections already implemented',
     'Calibration needs no retraining; the residual target needs one retrain.'),
    ('Move to leave-one-video-out evaluation', ''),
    ('Fold the finished results into the manuscript', ''),
    ('Submit, and extend the corpus beyond 10 recordings', ''),
]), 'Give each item a status, not a hope. Then go straight into the demo.')

# ════════════════════════════════════════════════════════════════════
# 8 · demonstration
# ════════════════════════════════════════════════════════════════════
slide('demo', head('Demonstration', 'What you are about to see, running live') + """
<div class="split-img r" style="--d:.2s"><div>""" + steps([
    ('Load the checkpoint, verify the data', 'QT0 gates everything downstream.'),
    ('Predict a held-out moment at t+1 … t+3', 'Against both baselines, same window.'),
    ('Render the prediction as a video frame', '512 points carried onto 58,000 Gaussians.'),
    ('Compare side by side with the real frame', 'With magnified motion arrows.'),
    ('Forecast past the end of the recording', 'Twelve frames beyond the last image.'),
]) + '</div>' + fig('sidebyside.jpg') + """</div>
<div class="note-bar live r" style="--d:.4s">Nothing here is pre-rendered. The checkpoint is
loaded and inference runs in front of you.</div>""",
      'Show this while the notebook starts. Say explicitly that it is live.')

# ════════════════════════════════════════════════════════════════════
# 9 · making the prediction visible
# ════════════════════════════════════════════════════════════════════
slide('viz', head('Rendering', 'Making the prediction visible')
      + fig('panels.jpg', 'wide',
            'real frame &nbsp;·&nbsp; 4DGS reconstruction &nbsp;·&nbsp; our prediction '
            '&nbsp;·&nbsp; naive baseline &nbsp;·&nbsp; predicted motion ×20 &nbsp;·&nbsp; '
            'error ×4')
      + '<div class="row-split r" style="--d:.3s">'
      + '<p class="aside">Tissue moves one to two pixels a frame, so the arrows are magnified — '
        'the factor is printed on every panel and never applied to a reported number. The forecast '
        'runs twelve frames past the end of the recording.</p>'
      + '<div class="stats3">' + stat('30.04', 'reconstruction ceiling, dB', 'steel')
      + stat('26.79', 'our prediction, dB', 'gold')
      + stat('26.74', 'naive baseline, dB', 'steel') + '</div></div>',
      'The demo material. The image-space gap is small for the same reason the geometric one is: '
      'magnitude, not direction.')

# ════════════════════════════════════════════════════════════════════
# 10 · the key finding
# ════════════════════════════════════════════════════════════════════
slide('key', head('The key finding', 'The model knows which way tissue moves — not how far')
      + fig('denoise.jpg', 'wide',
            'twenty denoising steps · green arrows move the true way, red oppose, gold is truth')
      + '<div class="cos r" style="--d:.3s">'
      + ''.join(f'<div class="c {k}"><i>{t}</i><b>{v}</b></div>' for t, v, k in
                [('t+1', '+0.57', 'pos'), ('t+2', '+0.40', 'pos'), ('t+3', '+0.26', 'pos'),
                 ('t+5', '−0.01', 'neg'), ('t+10', '−0.49', 'neg')])
      + '</div>'
      + '<div class="note-bar r" style="--d:.4s">A model that learned nothing would read 0.00 '
        'everywhere. The direction is right; the <em>distance</em> is wrong — which is a far more '
        'tractable problem, and both fixes are implemented.</div>',
      'Spend time here. It converts an unexplained failure into a specific, addressable one.')

# ════════════════════════════════════════════════════════════════════
# 11 · uncertainty
# ════════════════════════════════════════════════════════════════════
slide('unc', head('Our strongest result', 'The model knows when it should not be trusted') + """
<div class="split-img r" style="--d:.2s"><div>
<div class="stats-col">""" + stat('r = 0.69', 'spread vs realised error')
      + stat('5', 'futures sampled per input', 'steel')
      + stat('5×', 'uncertainty growth over 10 frames', 'steel') + """</div>
<p class="aside">When the five sampled futures disagree, the prediction really is worse — so the
system can flag its own weak moments. A model that is confidently wrong is unusable in theatre;
one that says <em>“do not trust me here”</em> is usable even when imperfect.</p>
<p class="aside small">Stated carefully: this is correlation between predicted spread and realised
error. Some of it may reflect that busy moments are both harder and more variable.</p>
</div>""" + fig('unc.png') + '</div>',
      'Independent of the accuracy ranking, and genuinely publishable. Give the caveat before the '
      'panel raises it.')

# ════════════════════════════════════════════════════════════════════
# 12 · the comparison
# ════════════════════════════════════════════════════════════════════
slide('cmp', head('Reported honestly', 'A two-line formula beats our model') + """
<div class="split-img wide-l r" style="--d:.2s"><div>
<table class="tbl"><thead><tr><th>Method</th><th>k = 1</th><th>k = 2</th><th>k = 3</th></tr></thead>
<tbody>
<tr><td>Nothing moves <span class="mz">(naive)</span></td><td>0.0093</td><td>0.0165</td><td>0.0222</td></tr>
<tr><td class="cv">Constant velocity</td><td class="cv">0.0055</td><td class="cv">0.0138</td><td class="cv">0.0187</td></tr>
<tr><td class="ours">Ours — diffusion</td><td class="ours">0.0093</td><td class="ours">0.0165</td><td class="ours">0.0225</td></tr>
</tbody></table>
<p class="tnote">Chamfer Distance, lower is better · full held-out set, 10 videos</p>
<p class="aside">We lose on all ten videos, so it is not a fluke we can explain away. Before
concluding anything we tested and eliminated the two obvious alternatives.</p>
</div>""" + fig('timeline8.png', '', 'no drift (ρ = +0.02, p = 0.61) · no dropped frames '
                                     '(333 origins checked, 0 found)') + '</div>',
      'Do not soften this. Eliminating two explanations by measurement is what let us find the '
      'cause.')

# ════════════════════════════════════════════════════════════════════
CSS = r"""
/* === OPERATING THEATRE ================================== */
/* Theatres are green because green is the complement of blood red. The ground
   is drape green; the accents are taken from the footage itself. */
:root{
  --stage-bg:#061513; --slide-bg:#0A1F1C;
  --bone:#EDF3F1; --steel:#A3B8B3; --dim:#6E8782;
  --green:#4FA88F;            /* drape green  — our system, verified   */
  --tissue:#C45A52;           /* tissue red   — the gap, where we lose */
  --gold:#D9A441;             /* motion arrows — prediction            */
  --panel:rgba(237,243,241,.045); --line:rgba(237,243,241,.13);
  --sans:'IBM Plex Sans','Segoe UI',Helvetica,Arial,sans-serif;
  --mono:'IBM Plex Mono',Consolas,monospace;
}
.slide{
  background:radial-gradient(120% 90% at 10% 0%,#12352F 0%,#0A1F1C 55%,#061513 100%);
  font-family:var(--sans); color:var(--bone);
  box-sizing:border-box; padding:276px 132px 104px;
}
.slide.has-lead{padding-top:356px}
.slide.arch{padding-top:246px} .slide.arch.has-lead{padding-top:286px}
.slide.arch h1{top:150px} .slide.arch .lead{top:230px;font-size:21px}
.content{position:relative;width:100%;height:100%;display:flex;flex-direction:column;justify-content:center}
.content > *:first-child{margin-top:0}

/* === CHROME ============================================= */
.kick,.kick-r{position:absolute;top:84px;font-family:var(--mono);font-size:18px;
  letter-spacing:.26em;text-transform:uppercase}
.kick{left:132px;color:var(--green)} .kick-r{right:132px;color:var(--dim);letter-spacing:.18em}
.hr-top{position:absolute;left:132px;right:132px;top:130px;height:1px;
  background:linear-gradient(90deg,var(--green),rgba(79,168,143,0) 58%)}
.pager{position:absolute;right:132px;bottom:56px;font-family:var(--mono);font-size:16px;
  letter-spacing:.2em;color:var(--dim)}
.foot{position:absolute;left:132px;bottom:56px;font-family:var(--mono);font-size:15px;
  letter-spacing:.15em;color:rgba(110,135,130,.7);text-transform:uppercase}

/* === TYPE =============================================== */
h1{position:absolute;left:132px;right:132px;top:176px;margin:0;font-weight:600;font-size:62px;
  line-height:1.08;letter-spacing:-.022em;color:var(--bone)}
h1.big{font-size:104px;line-height:1.02;top:236px;right:420px;font-weight:600}
h1 em{font-style:normal;color:var(--gold)}
.lead{position:absolute;left:132px;right:132px;top:272px;margin:0;font-size:25px;line-height:1.45;
  color:var(--steel);max-width:1400px}
.lead.wide{top:556px;right:700px;font-size:29px;color:var(--steel)}
.lead b{color:var(--bone);font-weight:600}
.aside{margin:0 0 18px;font-size:21px;line-height:1.5;color:var(--steel)}
.aside em{font-style:italic;color:var(--bone)}
.aside.small{font-size:18px;color:var(--dim)}

/* === FIGURES ============================================ */
.fig{margin:0;position:relative}
.fig img{display:block;width:100%;height:auto;border:1px solid var(--line);border-radius:3px}
.fig.wide{margin-top:4px}
.fig figcaption{margin-top:12px;font-family:var(--mono);font-size:14px;letter-spacing:.07em;
  color:var(--dim);text-align:center}
.hero{position:absolute;left:0;right:0;bottom:0;height:188px;overflow:hidden}
.hero img{width:100%;height:100%;object-fit:cover;object-position:center 50%;opacity:.92}
.hero-fade{position:absolute;inset:0;
  background:linear-gradient(180deg,#0A1F1C 0%,rgba(10,31,28,.62) 40%,rgba(10,31,28,.2) 100%)}

/* === BLOCKS ============================================= */
.card{background:var(--panel);border-top:2px solid var(--line);padding:20px 22px 22px}
.card h3{margin:0 0 10px;font-family:var(--mono);font-size:15px;letter-spacing:.14em;
  text-transform:uppercase;color:var(--bone);font-weight:500}
.card ul{margin:0;padding:0;list-style:none}
.card li{font-size:19px;line-height:1.42;color:var(--steel)}
.card.dense{padding:16px 20px 18px} .card.dense li{font-size:18px}
.card.green{border-top-color:var(--green)} .card.green h3{color:var(--green)}
.card.tissue{border-top-color:var(--tissue)} .card.tissue h3{color:var(--tissue)}
.three{display:grid;grid-template-columns:repeat(3,1fr);gap:22px;margin-top:26px}
.row-split{display:grid;grid-template-columns:1.25fr 1fr;gap:44px;margin-top:22px;align-items:start}
.split-img{display:grid;grid-template-columns:1fr 1fr;gap:46px;margin-top:0;align-items:start}
.split-img.wide-l{grid-template-columns:1.1fr 1fr}

.stats3{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
.stats-col{display:grid;grid-template-columns:1fr;gap:14px;margin-bottom:24px}
.stat{background:var(--panel);border-left:2px solid var(--line);padding:14px 18px}
.stat b{display:block;font-size:34px;font-weight:600;line-height:1.1;color:var(--bone)}
.stat i{display:block;font-style:normal;font-family:var(--mono);font-size:13px;letter-spacing:.11em;
  text-transform:uppercase;color:var(--dim);margin-top:7px}
.stat.green{border-left-color:var(--green)} .stat.green b{color:var(--green)}
.stat.gold{border-left-color:var(--gold)} .stat.gold b{color:var(--gold)}
.stat.steel b{color:var(--steel)}

.note-bar{margin-top:24px;padding:16px 22px;background:rgba(79,168,143,.09);
  border-left:3px solid var(--green);font-size:20px;line-height:1.45;color:var(--bone)}
.note-bar em{font-style:italic;color:var(--gold)}
.note-bar.live{border-left-color:var(--gold);background:rgba(217,164,65,.1)}
.verdict{margin-top:24px;padding:18px 24px;background:rgba(217,164,65,.1);
  border-left:3px solid var(--gold);font-size:22px;line-height:1.4;color:var(--bone)}
.verdict em{font-style:italic;color:var(--gold);font-weight:600}

/* rows */
.rows{margin-top:22px}
.rows-hdr{display:grid;grid-template-columns:1fr 46px 1.25fr;margin-bottom:10px}
.rows-hdr span{font-family:var(--mono);font-size:14px;letter-spacing:.2em;text-transform:uppercase}
.rows-hdr .lh{color:var(--green)} .rows-hdr .rh{color:var(--tissue);grid-column:3}
.row{display:grid;grid-template-columns:1fr 46px 1.25fr;align-items:center;margin-bottom:12px}
.rl,.rr{padding:15px 20px;background:var(--panel)}
.rl{border-left:2px solid rgba(79,168,143,.55)} .rr{border-left:2px solid rgba(196,90,82,.6)}
.rl b,.rr b{display:block;font-size:20px;font-weight:600;margin-bottom:3px}
.rl b{color:var(--green)} .rr b{color:var(--tissue)}
.rl span,.rr span{display:block;font-size:18px;line-height:1.4;color:var(--steel)}
.rarrow{position:relative;height:2px;background:rgba(237,243,241,.17)}
.rarrow::after{content:'';position:absolute;right:0;top:-4px;border-left:8px solid rgba(237,243,241,.4);
  border-top:4px solid transparent;border-bottom:4px solid transparent}
code{font-family:var(--mono);font-size:15px;color:var(--dim)}

/* steps */
.steps{margin-top:22px}
.step{display:grid;grid-template-columns:62px 1fr;gap:20px;align-items:center;padding:13px 0;
  border-bottom:1px solid var(--line)}
.step:last-child{border-bottom:0}
.step .num{display:flex;align-items:center;justify-content:center;width:48px;height:48px;
  font-size:22px;font-weight:600;color:var(--green);border:1px solid rgba(79,168,143,.45)}
.step b{display:block;font-size:21px;font-weight:600;color:var(--bone);margin-bottom:3px}
.step span{display:block;font-size:18px;line-height:1.38;color:var(--dim)}

/* objectives + question */
.objs{display:grid;grid-template-columns:repeat(4,1fr);gap:20px;margin-top:26px}
.obj{background:var(--panel);border-top:2px solid var(--line);padding:20px 20px 16px}
.obj b{display:block;font-size:23px;font-weight:600;color:var(--bone);margin-bottom:8px}
.obj span{display:block;font-size:18px;line-height:1.4;color:var(--steel);min-height:50px}
.obj i{display:block;font-style:normal;font-family:var(--mono);font-size:13px;letter-spacing:.12em;
  text-transform:uppercase;padding-top:12px;border-top:1px solid var(--line)}
.obj i.ok{color:var(--green)} .obj i.part{color:var(--gold)}
.qblock{margin-top:34px;padding-top:22px;border-top:1px solid var(--line)}
.qlab{font-family:var(--mono);font-size:15px;letter-spacing:.22em;text-transform:uppercase;
  color:var(--tissue)}
.qblock p{margin:14px 0 0;font-size:36px;font-weight:500;line-height:1.26;color:var(--bone);
  letter-spacing:-.015em}
.qblock em{font-style:normal;color:var(--green)}

/* cosine tiles */
.cos{display:grid;grid-template-columns:repeat(5,1fr);gap:18px;margin-top:22px}
.cos .c{background:var(--panel);border-top:2px solid var(--line);padding:16px;text-align:center}
.cos .c i{display:block;font-style:normal;font-family:var(--mono);font-size:14px;letter-spacing:.18em;
  color:var(--dim);margin-bottom:8px}
.cos .c b{font-size:38px;font-weight:600;line-height:1}
.cos .c.pos b{color:var(--green)} .cos .c.pos{border-top-color:var(--green)}
.cos .c.neg b{color:var(--tissue)} .cos .c.neg{border-top-color:rgba(196,90,82,.5)}

/* table */
.tbl{width:100%;border-collapse:collapse;font-size:19px}
.tbl th{font-family:var(--mono);font-size:14px;letter-spacing:.15em;text-transform:uppercase;
  color:var(--dim);text-align:left;padding:12px 16px;border-bottom:1px solid var(--line);font-weight:400}
.tbl td{padding:14px 16px;border-bottom:1px solid rgba(237,243,241,.07);color:var(--steel)}
.tbl .ours{color:var(--tissue);font-weight:600} .tbl .cv{color:var(--green);font-weight:600}
.mz{color:var(--dim)}
.tnote{margin:12px 0 20px;font-size:16px;font-style:italic;color:var(--dim)}

/* architecture plate */
.archwrap{margin-top:14px;width:100%;display:flex;justify-content:center}
.archsvg{display:block;width:100%;height:auto;max-height:668px}

/* title slide */
.byline{position:absolute;left:132px;top:716px;font-family:var(--mono);font-size:17px;
  line-height:1.95;letter-spacing:.04em;color:var(--dim);z-index:2}
.byline span{color:var(--bone)}

/* === REVEAL ============================================= */
.r{opacity:0}
.slide.active .r{animation:up .8s cubic-bezier(.22,1,.36,1) forwards;animation-delay:var(--d,0s)}
@keyframes up{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:none}}

/* === DECK CHROME ======================================== */
.deck-controls{display:flex;align-items:center;gap:14px;background:rgba(6,21,19,.85);
  border:1px solid rgba(237,243,241,.15);padding:8px 15px;font-family:var(--mono);font-size:12px;
  letter-spacing:.12em;color:#8FA8A2;opacity:0;transition:opacity .3s}
.deck-viewport:hover .deck-controls{opacity:1}
.deck-controls button{background:none;border:0;color:#8FA8A2;font:inherit;cursor:pointer;padding:2px 6px}
.deck-controls button:hover{color:#EDF3F1}
#bar{position:fixed;left:0;top:0;height:2px;background:var(--green);z-index:1001;transition:width .3s}

/* === PRINT / PDF EXPORT ================================= */
/* one 16:9 slide per page, so "print to PDF" produces a shareable deck */
@page{size:1920px 1080px;margin:0}
@media print{
  #bar,.deck-controls{display:none!important}
  .slide{background:#0A1F1C!important;-webkit-print-color-adjust:exact;print-color-adjust:exact}
  .r{opacity:1!important;animation:none!important;transform:none!important}
}
"""

JS = r"""
function fitStage(){
  var s = Math.min(innerWidth/1920, innerHeight/1080);
  var st = document.querySelector('.deck-stage');
  st.style.transform='translate('+((innerWidth-1920*s)/2)+'px,'+((innerHeight-1080*s)/2)+'px) scale('+s+')';
}
addEventListener('resize', fitStage); fitStage();

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
  else if(k==='f'||k==='F'){document.fullscreenElement?document.exitFullscreen()
    :document.documentElement.requestFullscreen();}
});
var wl=0;
addEventListener('wheel',function(ev){var n=Date.now();if(n-wl<420)return;wl=n;
  show(i+(ev.deltaY>0?1:-1));},{passive:true});
document.querySelector('.deck-viewport').addEventListener('click',function(ev){
  if(ev.target.closest('.deck-controls'))return;
  show(i+(ev.clientX>innerWidth*0.35?1:-1));});
var tx=null;
addEventListener('touchstart',function(e){tx=e.changedTouches[0].clientX;},{passive:true});
addEventListener('touchend',function(e){if(tx===null)return;
  var dx=e.changedTouches[0].clientX-tx;if(Math.abs(dx)>50)show(i+(dx<0?1:-1));tx=null;},{passive:true});
show(parseInt((location.hash||'#1').slice(1),10)-1 || 0);
"""

import re
body = []
for idx, s in enumerate(slides, 1):
    s = re.sub(r'data-n="[0-9]+"', 'data-n="%d"' % idx, s)
    chrome = ('' if 'class="slide title' in s
              else f'<div class="foot">GASR · S26-017-R-GASR</div>'
                   f'<div class="pager">{idx:02d}</div>')
    body.append(s.replace('</section>', chrome + '</section>'))

HTML = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>From Reconstruction to Anticipation — FYP-2 Mid Evaluation</title>
<style>
/* === EMBEDDED FONTS (offline-safe) ====================== */
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
print(f'  {len(slides)} slides, {len(HTML)//1024} KB (fonts + figures embedded, fully offline)')
