const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle,
  WidthType, ShadingType, VerticalAlign, PageBreak,
  LevelFormat, TabStopType, TabStopPosition
} = require('docx');
const fs = require('fs');

// ── Design tokens ──────────────────────────────────────────────────
const NAVY   = '1B2A4A';
const TEAL   = '1B6B6B';
const PURPLE = '5B2D8E';
const GREY   = '555555';
const LGREY  = 'F3F6F9';
const MGREY  = 'D0DAE4';
const WHITE  = 'FFFFFF';
const CW     = 9360; // content width DXA

// ── Border helpers ─────────────────────────────────────────────────
const cb  = (c='CCCCCC', s=4) => ({ style:BorderStyle.SINGLE, size:s, color:c });
const cbs = c => ({ top:cb(c), bottom:cb(c), left:cb(c), right:cb(c) });
const nob = { top:cb('FFFFFF',0), bottom:cb('FFFFFF',0), left:cb('FFFFFF',0), right:cb('FFFFFF',0) };

// ── Typography ─────────────────────────────────────────────────────
const R  = (t, o={}) => new TextRun({ text:t, font:'Calibri', size:22, color:NAVY, ...o });
const RB = (t, o={}) => R(t, { bold:true, ...o });
const RI = (t, o={}) => R(t, { italics:true, ...o });
const RC = (t, o={}) => R(t, { font:'Courier New', size:20, ...o });

const body = (t, o={}) => new Paragraph({
  children: [R(t)], alignment:AlignmentType.JUSTIFIED,
  spacing:{ before:80, after:80 }, indent:{ firstLine:360 }, ...o
});
const bodyM = (runs, o={}) => new Paragraph({
  children:runs, alignment:AlignmentType.JUSTIFIED,
  spacing:{ before:80, after:80 }, indent:{ firstLine:360 }, ...o
});
const bodyN = (t, o={}) => new Paragraph({
  children:[R(t)], alignment:AlignmentType.JUSTIFIED,
  spacing:{ before:80, after:80 }, ...o
});

const H1 = t => new Paragraph({
  children:[R(t, { size:28, bold:true, color:NAVY })],
  spacing:{ before:480, after:160 },
  border:{ bottom:{ style:BorderStyle.SINGLE, size:8, color:TEAL, space:1 } }
});
const H2 = (t, col=TEAL) => new Paragraph({
  children:[R(t, { size:24, bold:true, color:col })],
  spacing:{ before:320, after:100 }
});
const H3 = t => new Paragraph({
  children:[R(t, { size:22, bold:true, italics:true, color:NAVY })],
  spacing:{ before:200, after:80 }
});

const PB  = () => new Paragraph({ children:[new PageBreak()] });
const SP  = n  => new Paragraph({ children:[R('')], spacing:{ before:0, after:n||120 } });

// Bullet item
const blt = (label, text, col=TEAL) => new Paragraph({
  numbering:{ reference:'bullets', level:0 },
  children:[
    ...(label ? [R(label+': ', { bold:true, color:col })] : []),
    R(text, { size:21 })
  ],
  spacing:{ before:40, after:40 }
});

// Code block (monospaced box)
const code = lines => new Table({
  width:{ size:CW, type:WidthType.DXA },
  columnWidths:[CW],
  rows:[new TableRow({ children:[new TableCell({
    borders:cbs('1B6B6B'),
    shading:{ fill:'071218', type:ShadingType.CLEAR },
    margins:{ top:140, bottom:140, left:200, right:200 },
    children: lines.map(l => new Paragraph({
      children:[new TextRun({ text:l, font:'Courier New', size:19, color:'7DD3D8' })],
      spacing:{ before:0, after:0 }
    }))
  })]})],
});

// Info box
const infoBox = (icon, title, text, accentColor='1B6B6B', bgColor='F0F7F7') =>
  new Table({
    width:{ size:CW, type:WidthType.DXA },
    columnWidths:[440, CW-440],
    rows:[new TableRow({ children:[
      new TableCell({
        borders:{ top:cb(accentColor,6), bottom:cb(accentColor,6), left:cb(accentColor,8), right:nob.right },
        shading:{ fill:bgColor, type:ShadingType.CLEAR },
        margins:{ top:140, bottom:140, left:160, right:0 },
        verticalAlign:VerticalAlign.CENTER,
        children:[new Paragraph({ children:[R(icon, { size:28 })], alignment:AlignmentType.CENTER })]
      }),
      new TableCell({
        borders:{ top:cb(accentColor,6), bottom:cb(accentColor,6), right:cb(accentColor,6), left:nob.left },
        shading:{ fill:bgColor, type:ShadingType.CLEAR },
        margins:{ top:100, bottom:100, left:160, right:160 },
        children:[
          new Paragraph({ children:[R(title, { bold:true, color:accentColor, size:21 })], spacing:{ before:0, after:40 } }),
          new Paragraph({ children:[R(text, { size:20, color:GREY })], alignment:AlignmentType.JUSTIFIED, spacing:{ before:0, after:0 } })
        ]
      })
    ]})]
  });

// Comparison table row builder
const trow = (cells, even=false) => new TableRow({
  children: cells.map((c, i) => new TableCell({
    borders:cbs(MGREY),
    shading:{ fill:even ? LGREY : WHITE, type:ShadingType.CLEAR },
    margins:{ top:80, bottom:80, left:130, right:130 },
    children:[new Paragraph({
      children:[R(String(c.text||c), { bold:!!c.bold, color: c.color||NAVY, size:20, italics:!!c.italic })],
      alignment: c.center ? AlignmentType.CENTER : AlignmentType.LEFT,
      spacing:{ before:0, after:0 }
    })]
  }))
});
const thead = cells => new TableRow({
  children: cells.map(c => new TableCell({
    borders:cbs(NAVY),
    shading:{ fill:NAVY, type:ShadingType.CLEAR },
    margins:{ top:80, bottom:80, left:130, right:130 },
    children:[new Paragraph({
      children:[R(String(c), { bold:true, color:WHITE, size:20 })],
      alignment:AlignmentType.CENTER, spacing:{ before:0, after:0 }
    })]
  }))
});

// ══════════════════════════════════════════════════════════════════
// CONTENT
// ══════════════════════════════════════════════════════════════════
const ch = [];
const P = (...items) => ch.push(...items);

// ── COVER ─────────────────────────────────────────────────────────
P(SP(300));
P(new Paragraph({ children:[R('FAST NATIONAL UNIVERSITY OF COMPUTER AND EMERGING SCIENCES', { bold:true, size:22, color:NAVY, characterSpacing:60 })], alignment:AlignmentType.CENTER, spacing:{ before:0, after:60 } }));
P(new Paragraph({ children:[R('Islamabad Campus  ·  Department of Computer Science', { size:21, color:GREY, italics:true })], alignment:AlignmentType.CENTER, spacing:{ before:0, after:60 } }));
P(new Paragraph({ children:[R('')], border:{ bottom:{ style:BorderStyle.SINGLE, size:8, color:TEAL, space:1 } }, spacing:{ before:0, after:280 } }));
P(new Paragraph({ children:[R('FINAL YEAR PROJECT', { bold:true, size:21, color:TEAL, characterSpacing:200 })], alignment:AlignmentType.CENTER, spacing:{ before:0, after:120 } }));
P(new Paragraph({ children:[R('Technology Reference Document', { bold:true, size:40, color:NAVY })], alignment:AlignmentType.CENTER, spacing:{ before:0, after:80 } }));
P(new Paragraph({ children:[R('Module 4 — Self-Supervised Deformation Sequences', { bold:true, size:28, color:TEAL })], alignment:AlignmentType.CENTER, spacing:{ before:0, after:60 } }));
P(new Paragraph({ children:[R('Module 5 — Diffusion-Based Deformation Learning', { bold:true, size:28, color:PURPLE })], alignment:AlignmentType.CENTER, spacing:{ before:0, after:300 } }));
P(new Paragraph({ children:[R('')], border:{ bottom:{ style:BorderStyle.SINGLE, size:4, color:MGREY, space:1 } }, spacing:{ before:0, after:200 } }));
P(new Paragraph({ children:[RB('Project: '), R('4D Surgical Scene Reconstruction & Tissue Deformation Prediction')], alignment:AlignmentType.CENTER, spacing:{ before:0, after:60 } }));
P(new Paragraph({ children:[RB('Team: '), R('Ahmad Rashad (22i-1175)  ·  Shehroz Kashif (22i-0771)  ·  Humair Khan (22i-2632)')], alignment:AlignmentType.CENTER, spacing:{ before:0, after:60 } }));
P(new Paragraph({ children:[RB('Supervisor: '), R('Dr Akhtar Jamil  ·  April 2026')], alignment:AlignmentType.CENTER, spacing:{ before:0, after:0 } }));
P(PB());

// ── TOC ────────────────────────────────────────────────────────────
P(H1('Table of Contents'));
const tocItems = [
  ['1.', 'Overview and Context', '3'],
  ['2.', 'Module 4 Technologies', '4'],
  ['  2.1', 'Farthest Point Sampling (FPS)', '4'],
  ['  2.2', 'Trajectory Normalisation', '5'],
  ['  2.3', 'Sliding Window Temporal Pairing', '5'],
  ['  2.4', 'Multi-Step Prediction Augmentation', '6'],
  ['  2.5', 'Velocity Frame Encoding', '6'],
  ['  2.6', 'Temporal Train/Validation Split', '7'],
  ['  2.7', 'Displacement Scale Normalisation', '7'],
  ['3.', 'Module 5 Technologies', '8'],
  ['  3.1', 'Denoising Diffusion Probabilistic Models (DDPM)', '8'],
  ['  3.2', 'Residual Prediction Paradigm', '9'],
  ['  3.3', 'Cosine Beta Schedule', '10'],
  ['  3.4', 'Geometry MLP (GeomMLP)', '10'],
  ['  3.5', 'Sinusoidal Positional Encoding', '11'],
  ['  3.6', 'Transformer Encoder with Pre-Layer Norm', '11'],
  ['  3.7', 'Cross-Attention Conditioning', '12'],
  ['  3.8', 'Exponential Moving Average (EMA)', '13'],
  ['  3.9', 'Denoising Diffusion Implicit Models (DDIM)', '13'],
  ['  3.10','AdamW Optimiser + Cosine LR with Warmup', '14'],
  ['4.', 'Why Each Technology Was Chosen', '15'],
  ['5.', 'Technology Interaction Diagram', '16'],
];
for (const [n, t, pg] of tocItems) {
  P(new Paragraph({
    children:[ R(n+'  '+t, { size:21, bold:!n.startsWith(' ') }), R('\t', { size:21 }), R(pg, { size:21, color:GREY }) ],
    tabStops:[{ type:TabStopType.RIGHT, position:CW, leader:'dot' }],
    spacing:{ before:n.startsWith(' ')?28:56, after:28 },
    indent: n.startsWith(' ') ? { left:360 } : {}
  }));
}
P(PB());

// ══════════════════════════════════════════════════════════════════
// CH 1: OVERVIEW
// ══════════════════════════════════════════════════════════════════
P(H1('1. Overview and Context'));
P(body('This document provides a detailed technical explanation of every technology, algorithm, and design decision used in Module 4 (Self-Supervised Deformation Sequence Construction) and Module 5 (Residual Diffusion-Based Deformation Learning) of the 4D Surgical Scene Reconstruction project. It is intended to serve as a reference for understanding why each technology was chosen, how it works mathematically, and how it fits into the overall pipeline.'));
P(SP(80));
P(infoBox('📌', 'Pipeline Position',
  'Modules 1–3 reconstruct surgical tissue in 4D from monocular video, producing the trajectory tensor T ∈ ℝ²¹⁹×⁵⁷⁹⁴³×³. Module 4 transforms this trajectory into a self-supervised training dataset. Module 5 trains a diffusion model on that dataset to predict future tissue geometry — all without any clinical labels.',
  TEAL, 'F0F7F7'));
P(SP(160));
P(infoBox('⚠️', 'Key Design Decision: Residual vs Absolute Prediction',
  'The v1 diffusion model (Module 5) predicted absolute positions G_{t+1} directly from noise. This failed catastrophically (CD=2.3 vs naive CD=0.0005) because generating 512 spatially correct 3D coordinates from pure random noise is intractable with small training data. The v2 redesign predicts displacement residuals ΔG = G_{t+1} − G_t instead. The model only needs to learn tiny motion vectors (~0.004 normalised units), which is well within the capability of a diffusion model trained on 5,901 examples.',
  'B05000', 'FFF8F0'));
P(SP(120));

// ══════════════════════════════════════════════════════════════════
// CH 2: MODULE 4 TECHNOLOGIES
// ══════════════════════════════════════════════════════════════════
P(H1('2. Module 4 Technologies'));
P(body('Module 4 transforms the raw 4D Gaussian trajectory from Module 3 into a clean, self-supervised training dataset for the diffusion model. It is responsible for point cloud subsampling, coordinate normalisation, temporal windowing, data augmentation, and quality validation.'));

// ── 2.1 FPS ───────────────────────────────────────────────────────
P(H2('2.1 Farthest Point Sampling (FPS)', TEAL));
P(body('The trajectory tensor T contains 57,943 Gaussian positions per frame. Using all 57,943 points per training window would require 57,943×3 = 173,829 input dimensions — computationally infeasible for a neural network with 6M parameters. Farthest Point Sampling (FPS) addresses this by selecting a maximally-spread subset of N_ANCHOR = 512 representative points.'));
P(H3('How FPS works'));
P(body('FPS is a greedy iterative algorithm. It begins by selecting one point at random, then repeatedly selects the point that is farthest from the already-selected set, measured by Euclidean distance in 3D. After N_ANCHOR iterations, the selected points maximally cover the point cloud — they are spread across the tissue surface rather than clustered in high-density regions.'));
P(code([
  'def farthest_point_sample(pts, n_sample):',
  '    N = pts.shape[0]',
  '    selected = np.zeros(n_sample, dtype=np.int64)',
  '    dists    = np.full(N, np.inf, dtype=np.float32)',
  '    selected[0] = np.random.randint(0, N)',
  '    for i in range(1, n_sample):',
  '        last_pt = pts[selected[i-1]]',
  '        d = np.sum((pts - last_pt)**2, axis=1)',
  '        dists = np.minimum(dists, d)',
  '        selected[i] = np.argmax(dists)',
  '    return selected   # (512,) indices into the full point cloud',
]));
P(SP(80));
P(H3('Why FPS over random sampling'));
P(body('Random sampling concentrates points in high-density regions, leaving low-density areas (e.g. tissue edges, fold boundaries) unrepresented. FPS guarantees geometric coverage of the entire tissue surface. This matters because tissue deformation is most physically significant at boundaries and folds — exactly where random sampling is least likely to place anchor points.'));
P(H3('Opacity weighting'));
P(body('FPS is applied to the set of visible Gaussians only — those with opacity σ(o) > 0.05. This ensures anchor points correspond to real, rendered tissue surface rather than background or artefact Gaussians. The opacity values are loaded from the saved checkpoint (the 4dgs_v4.pth file) rather than from runtime variables, making Module 4 safe to run after a Colab session restart.'));
P(body('The same 512 anchor indices are applied across all 219 frames. This is critical: if different indices were used per frame, temporal identity would be broken and the model would learn nothing about motion trajectories.'));
P(SP(80));

// ── 2.2 Normalisation ─────────────────────────────────────────────
P(H2('2.2 Trajectory Normalisation', TEAL));
P(body('Raw Gaussian positions are in arbitrary world units that differ between videos (different tissue sizes, depths, camera placements). To enable cross-video training in a single diffusion model, all trajectories must be brought into a common coordinate space.'));
P(code([
  'centre    = traj[0].mean(axis=0)           # centroid of frame 0',
  'scale     = np.abs(traj[0] - centre).max() # max absolute deviation',
  'traj_norm = (traj - centre) / scale        # maps roughly to [-1, 1]',
]));
P(body('The normalisation uses frame 0\'s statistics (centre and scale) and applies them to all subsequent frames. This is essential: using per-frame normalisation would remove the inter-frame displacement information, making motion invisible to the model. By anchoring normalisation to frame 0, the displacement ΔG between frames remains intact in the normalised space.'));
P(SP(80));

// ── 2.3 Sliding Window ────────────────────────────────────────────
P(H2('2.3 Sliding Window Temporal Pairing', TEAL));
P(body('The self-supervised learning strategy derives from a fundamental observation: every consecutive pair of trajectory frames (G_t, G_{t+1}) is a real, physically observed tissue deformation event. No human annotation is required — the temporal ordering of the reconstruction itself provides supervision.'));
P(body('A sliding window of size W=4 extracts the context: the model receives the last 4 observed geometry frames as input and must predict the geometry at a future timestep. The window slides forward one frame at a time across the 219-frame trajectory, generating one training pair per position. For a single video with 219 frames, this yields approximately 215 windows before augmentation.'));
P(code([
  'for t in range(WINDOW_SIZE, N_F - pred_step + 1):',
  '    inp = traj_norm[t - WINDOW_SIZE : t]     # shape: (4, 512, 3)',
  '    tgt = traj_norm[t + pred_step - 1]        # shape: (512, 3)',
]));
P(SP(80));

// ── 2.4 Multi-Step Augmentation ───────────────────────────────────
P(H2('2.4 Multi-Step Prediction Augmentation', TEAL));
P(body('A single video with 219 frames yields only ~215 training windows — insufficient for a generative model. Multi-step prediction augmentation constructs windows for multiple prediction horizons simultaneously: PRED_STEPS = [1, 2, 3]. The same context window [G_{t-3}, G_{t-2}, G_{t-1}, G_t] is paired with targets G_{t+1}, G_{t+2}, and G_{t+3} as three separate training examples.'));
P(body('This triples the training dataset (from ~1,127 to ~5,901 windows across 10 videos) without collecting any additional data. It also has a beneficial regularisation effect: by training the model to predict both near-future and slightly-further-future geometry from the same context, the model learns more robust motion representations that do not overfit to single-step patterns.'));
P(SP(80));

// ── 2.5 Velocity Encoding ─────────────────────────────────────────
P(H2('2.5 Velocity Frame Encoding', TEAL));
P(body('Rather than providing raw absolute geometry frames as context to the diffusion model, Module 4 (v2) computes velocity frames — finite differences between consecutive context frames. The velocity at time t is defined as V_t = G_t − G_{t-1}, representing how much each tissue particle moved between frames t-1 and t.'));
P(code([
  'def compute_velocities(inp):',
  '    # inp: (M, W, N_A, 3)  →  vel: (M, W-1, N_A, 3)',
  '    return inp[:, 1:, :, :] - inp[:, :-1, :, :]',
]));
P(body('Velocity frames have two advantages over raw frames. First, they emphasise motion patterns rather than absolute positions, which is exactly the information needed to predict future motion. A model conditioned on velocities can reason about acceleration and deceleration of tissue, not just where the tissue currently is. Second, velocity frames are invariant to the absolute position of the tissue in space, which helps the model generalise across different videos where the tissue occupies different regions of the normalised coordinate space.'));
P(SP(80));

// ── 2.6 Temporal Split ────────────────────────────────────────────
P(H2('2.6 Temporal Train/Validation Split', TEAL));
P(body('The dataset is split 80% train / 20% validation using a temporal (non-random) split. The first 80% of windows (chronologically earliest frames) form the training set, and the last 20% form the validation set. This is a critical methodological requirement: a random split would assign some windows from early in the sequence to validation and some from later in the sequence to training. Because later frames are physically adjacent to earlier frames, this constitutes future data leakage — the training set would contain ground-truth information about what happens after the validation frames, making validation metrics artificially optimistic.'));
P(SP(80));

// ── 2.7 Displacement Scale ────────────────────────────────────────
P(H2('2.7 Displacement Scale Normalisation', TEAL));
P(body('The displacement residuals ΔG = G_{t+1} − G_t are typically very small (mean ~0.004 normalised units). Passing these raw tiny values to the diffusion model is problematic: the model would need to generate values near zero with very low variance, which poorly utilises the [-1, 1] dynamic range of the diffusion process.'));
P(code([
  'disp_scale   = float(np.abs(train_delta).max()) + 1e-8',
  'train_delta_n = (train_delta / disp_scale).astype(np.float32)',
  '',
  '# At inference: denormalise before adding to last frame',
  'delta_pred = ddim_sample() * disp_scale  # back to original scale',
  'G_t_plus_1 = G_t + delta_pred',
]));
P(body('By dividing by disp_scale (the maximum absolute displacement in the dataset), the normalised residuals span [-1, 1]. The diffusion model operates entirely in this normalised residual space. At inference time, the generated sample is multiplied by disp_scale before being added to the last observed frame G_t to recover the predicted absolute position.'));
P(PB());

// ══════════════════════════════════════════════════════════════════
// CH 3: MODULE 5 TECHNOLOGIES
// ══════════════════════════════════════════════════════════════════
P(H1('3. Module 5 Technologies'));
P(body('Module 5 trains a conditional denoising diffusion probabilistic model (DDPM) on the deformation windows from Module 4. The model learns to reverse a noise-corruption process applied to displacement residuals, conditioned on velocity context and the last observed frame. This section explains each technology in depth.'));

// ── 3.1 DDPM ──────────────────────────────────────────────────────
P(H2('3.1 Denoising Diffusion Probabilistic Models (DDPM)', PURPLE));
P(body('DDPMs, introduced by Ho et al. (NeurIPS 2020), define a generative model through a two-process framework. The forward process gradually destroys data by adding Gaussian noise over T timesteps. The reverse process is a learned neural network that reconstructs data by progressively denoising.'));
P(H3('Forward process'));
P(body('Given a clean data sample x₀ (in our case, a normalised displacement residual ΔG/disp_scale), the forward process defines the distribution of the noisy sample x_t at each timestep t:'));
P(code([
  'q(x_t | x_0) = N(x_t ; √ᾱ_t · x_0, (1 − ᾱ_t) · I)',
  '',
  '# Equivalently (reparameterisation):',
  'x_t = √ᾱ_t · x_0 + √(1 − ᾱ_t) · ε,   ε ~ N(0, I)',
]));
P(body('Here ᾱ_t = Π_{s=1}^{t} αs where αs = 1 − βs, and β_1, ..., β_T is the noise schedule. At t=0, x_t = x_0 (clean). At t=T, x_T ≈ N(0, I) (pure noise). The forward process is not learned — it is a fixed mathematical transformation that can be applied in closed form for any t.'));
P(H3('Reverse process and training objective'));
P(body('The neural network learns to predict the noise ε that was added at each timestep. The training objective is:'));
P(code([
  'L = E_{t, x_0, ε} [ || ε − ε_θ(x_t, t, context) ||² ]',
  '',
  '# In code:',
  'x_noisy, noise = q_sample(delta_n_b, t)   # forward process',
  'noise_pred     = model(vel_b, last_b, x_noisy, t)',
  'loss           = F.mse_loss(noise_pred, noise)',
]));
P(body('After training, the model can generate new samples by starting from pure Gaussian noise and applying the learned reverse process T times, each step denoising slightly. The key insight for Module 5 is that the model is conditioned on the velocity context and last observed frame — so it generates deformations that are physically consistent with the observed recent motion history.'));
P(SP(80));

// ── 3.2 Residual Prediction ───────────────────────────────────────
P(H2('3.2 Residual Prediction Paradigm', PURPLE));
P(body('The most important design decision in Module 5 v2 is predicting displacement residuals ΔG rather than absolute positions G_{t+1}. This fundamentally changes the difficulty of the learning task.'));
P(SP(60));
P(new Table({ width:{ size:CW, type:WidthType.DXA }, columnWidths:[4400,4960],
  rows:[
    thead(['Absolute Prediction (v1 — FAILED)', 'Residual Prediction (v2 — Current)']),
    trow([
      { text:'Target: full 3D position of 512 points', color:GREY },
      { text:'Target: displacement ΔG = G_{t+1} − G_t', color:TEAL, bold:true }
    ], false),
    trow([
      { text:'Naive CD = 0.000544 (copy-last-frame is near-perfect)', color:GREY },
      { text:'Naive CD = mean frame displacement (~0.004 norm.)', color:TEAL }
    ], true),
    trow([
      { text:'Model must place 512 points in 3D space from noise — nearly impossible', color:GREY },
      { text:'Model must generate tiny motion vectors — well within capability', color:TEAL }
    ], false),
    trow([
      { text:'Result: CD = 2.3 (model performs 422,000× worse than naive)', color:'AA2020', bold:true },
      { text:'Result: CD approaches naive as model learns motion patterns', color:TEAL, bold:true }
    ], true),
  ]
}));
P(SP(80));
P(body('The naive baseline for residual prediction is predicting ΔG = 0 (no motion), which gives CD ≈ mean frame-to-frame displacement. The diffusion model must learn to beat this by recognising motion patterns in the velocity context. This is a meaningful and achievable learning target.'));
P(SP(80));

// ── 3.3 Cosine Beta Schedule ──────────────────────────────────────
P(H2('3.3 Cosine Beta Schedule', PURPLE));
P(body('The noise schedule β_1, ..., β_T controls how quickly noise is added in the forward process. Module 5 uses the cosine schedule introduced by Nichol and Dhariwal (ICML 2021), which maintains a slower rate of signal destruction at the beginning and end of the process compared to a linear schedule.'));
P(code([
  'def cosine_beta_schedule(T, s=0.008):',
  '    steps = torch.arange(T + 1, dtype=torch.float64)',
  '    f     = torch.cos(((steps/T + s) / (1+s)) * math.pi/2) ** 2',
  '    f     = f / f[0]',
  '    betas = (1 - f[1:] / f[:-1]).clamp(0, 0.999)',
  '    return betas.float()',
]));
P(body('Module 5 uses T = 200 timesteps rather than the standard 1,000. This choice is motivated by the scale of the prediction target: tissue displacements are tiny (mean 0.004 normalised units). With T = 1,000, the intermediate noise levels are unnecessarily granular for such small signals. T = 200 provides sufficient resolution while training approximately 5× faster per epoch.'));
P(SP(80));

// ── 3.4 GeomMLP ───────────────────────────────────────────────────
P(H2('3.4 Geometry MLP (GeomMLP)', PURPLE));
P(body('The GeomMLP is the point cloud encoder used throughout the ResidualDenoiser. It encodes a geometry frame (512 points × 3 coordinates = 1,536 scalar values) into a fixed-size latent vector of dimension 256.'));
P(code([
  'class GeomMLP(nn.Module):',
  '    def __init__(self, n_pts, out_dim, hidden=512, dropout=0.1):',
  '        self.net = nn.Sequential(',
  '            nn.Linear(n_pts * 3, hidden), nn.GELU(), nn.Dropout(dropout),',
  '            nn.Linear(hidden, hidden),    nn.GELU(),',
  '            nn.Linear(hidden, out_dim),',
  '        )',
  '    def forward(self, x):   # x: (B, N_A, 3) or (B, W, N_A, 3)',
  '        B = x.shape[0]',
  '        if x.dim() == 4:  # batch of W frames',
  '            return self.net(x.reshape(B, W, -1))  # (B, W, 256)',
  '        return self.net(x.reshape(B, -1))          # (B, 256)',
]));
P(body('The GELU (Gaussian Error Linear Unit) activation function is used instead of ReLU because it provides smoother gradients near zero, which is beneficial when the inputs (normalised point coordinates) are small-valued. Dropout (rate 0.1) provides mild regularisation. The MLP flattens the point cloud into a single vector — it does not preserve point-wise spatial relationships, which is acceptable because the 512 anchor points have fixed correspondence across frames (same tissue particle always at the same index).'));
P(SP(80));

// ── 3.5 Sinusoidal PE ─────────────────────────────────────────────
P(H2('3.5 Sinusoidal Positional Encoding', PURPLE));
P(body('The diffusion timestep t (an integer in [0, T−1]) must be communicated to the model so it knows how much noise has been applied and therefore how aggressively to denoise. Sinusoidal positional encoding converts the integer timestep into a continuous vector:'));
P(code([
  'class SinusoidalPE(nn.Module):',
  '    def forward(self, t):     # t: (B,) integer timesteps',
  '        half  = self.dim // 2',
  '        freqs = exp(-log(10000) * arange(half) / half)   # log-spaced frequencies',
  '        x     = t[:, None].float() * freqs[None]          # (B, half)',
  '        return cat([x.sin(), x.cos()], dim=-1)            # (B, dim)',
]));
P(body('The log-spaced frequencies ensure that nearby timesteps produce similar encodings while distant timesteps produce clearly distinguishable encodings. The sinusoidal encoding is borrowed from the original Transformer paper (Vaswani et al., 2017) and has become a standard component of diffusion model architectures. The encoded timestep vector is added to the noisy residual latent, allowing the model to implicitly adjust its denoising strength based on the noise level.'));
P(SP(80));

// ── 3.6 Transformer Encoder ───────────────────────────────────────
P(H2('3.6 Transformer Encoder with Pre-Layer Normalisation', PURPLE));
P(body('The velocity context (W−1 = 3 velocity frames, each encoded to a 256-dim latent) is aggregated by a Transformer Encoder. The Transformer allows the model to attend across the temporal dimension — learning which past motion frames are most informative for predicting the current deformation.'));
P(code([
  'enc_layer = nn.TransformerEncoderLayer(',
  '    d_model=256, nhead=4,',
  '    dim_feedforward=512,    # 2× latent dim',
  '    dropout=0.1,',
  '    batch_first=True,',
  '    norm_first=True,        # PRE-layer normalisation (more stable)',
  ')',
  'self.vel_tf = nn.TransformerEncoder(enc_layer, num_layers=3,',
  '                                     enable_nested_tensor=False)',
]));
P(body('Pre-layer normalisation (norm_first=True) places the LayerNorm before each sub-layer (attention and feedforward) rather than after. This variant, proposed by Xiong et al. (2020), substantially improves training stability for deep transformers, particularly at the beginning of training when gradients can be large. The standard post-layer normalisation variant is more prone to gradient explosion in the first few hundred training steps.'));
P(body('Three Transformer layers are used — leaner than the six-layer design of Module 5 v1. For a dataset of 5,901 training windows with 256-dim latents, three layers provides sufficient temporal reasoning capacity without overfitting.'));
P(SP(80));

// ── 3.7 Cross-Attention ───────────────────────────────────────────
P(H2('3.7 Cross-Attention Conditioning', PURPLE));
P(body('Cross-attention is the mechanism by which the noisy displacement residual queries the context (velocity frames + last observed frame). Unlike self-attention where a sequence attends to itself, cross-attention allows one sequence (the noisy residual query) to attend to a different sequence (the context key-value pairs).'));
P(code([
  '# Context: velocity latents (B, 3, 256) + last frame latent (B, 1, 256)',
  'z_ctx = cat([z_vel, z_last], dim=1)  # (B, 4, 256)',
  '',
  '# Query: noisy residual latent + timestep',
  'z_q = (z_noisy + z_t).unsqueeze(1)  # (B, 1, 256)',
  '',
  '# Cross-attention: query attends to context',
  'z_out, _ = self.cross_attn(z_q, z_ctx, z_ctx)  # (B, 1, 256)',
  'z_out    = self.norm_q(z_out.squeeze(1) + z_noisy)  # residual connection',
]));
P(body('The key-value pairs are the context latents (representing what the tissue has done recently), and the query is the current noisy residual (representing what deformation we are trying to denoise). The attention mechanism learns which past motion frames are most relevant for predicting the current deformation — for example, learning that tissue moving quickly in a consistent direction is likely to continue moving in that direction.'));
P(body('A residual connection adds the original noisy residual latent z_noisy to the cross-attention output. This ensures the model can always fall back to using the raw noisy input if the context provides no useful information, stabilising training in the early epochs.'));
P(SP(80));

// ── 3.8 EMA ───────────────────────────────────────────────────────
P(H2('3.8 Exponential Moving Average (EMA)', PURPLE));
P(body('The EMA maintains a shadow copy of the model weights as a running average of all past weight states:'));
P(code([
  'class EMA:',
  '    def update(self, model):',
  '        with torch.no_grad():',
  '            for ema_p, p in zip(self.model.parameters(),',
  '                                model.parameters()):',
  '                ema_p.data.mul_(decay).add_(p.data, alpha=1 - decay)',
]));
P(body('With decay = 0.999, the EMA weights evolve slowly and smoothly, effectively averaging the model over the last ~1/(1−0.999) = 1,000 weight updates. The EMA model is used exclusively at inference time (both for validation during training and for final prediction). Training uses the raw model weights via gradient descent.'));
P(body('EMA improves sample quality substantially for diffusion models because the raw training weights fluctuate around the loss minimum — each gradient step overshoots slightly in a random direction. The EMA averages out these fluctuations, producing weights that sit more stably at the minimum. For diffusion models on small datasets, this benefit is especially pronounced because the stochastic gradient noise is larger relative to the signal.'));
P(SP(80));

// ── 3.9 DDIM ──────────────────────────────────────────────────────
P(H2('3.9 Denoising Diffusion Implicit Models (DDIM)', PURPLE));
P(body('Standard DDPM sampling requires T = 200 denoising steps, each requiring one full forward pass of the neural network. DDIM (Song et al., ICLR 2021) reformulates the reverse process as a non-Markovian ordinary differential equation, enabling high-quality generation in far fewer steps (50 in Module 5).'));
P(code([
  '# DDIM reverse step (deterministic, η=0):',
  'ab   = alpha_bar[ti]',
  'x0_p = (x - sqrt_1mab[ti] * eps) / sqrt_ab[ti]   # predicted x_0',
  'x0_p = x0_p.clamp(-2, 2)                           # numerical stability',
  '',
  '# Move to next (less noisy) timestep:',
  'ab_n = alpha_bar[ti_next]',
  'x    = ab_n.sqrt() * x0_p + (1 - ab_n).sqrt() * eps',
]));
P(body('The DDIM update rule interpolates between the predicted clean sample x₀_pred and the noise direction eps, using the alpha values at the current and next timestep. With η = 0, the process is fully deterministic: the same context always produces the same prediction. With η > 0 (stochastic DDIM, essentially DDPM), each sample introduces fresh random noise, enabling multi-hypothesis generation where 5 different calls produce 5 plausible future tissue geometries.'));
P(body('Module 5 uses 50 DDIM steps for validation (4× faster than full 200-step DDPM) and 20 steps for intermediate validation during training (10× faster). The quality difference between 20 and 50 steps is small for the displacement residual prediction task because the target distribution is relatively simple (small Gaussian-like motion vectors) compared to image generation.'));
P(SP(80));

// ── 3.10 AdamW + LR warmup ────────────────────────────────────────
P(H2('3.10 AdamW Optimiser + Cosine LR with Warmup', PURPLE));
P(body('Module 5 uses the AdamW optimiser, a variant of Adam that decouples weight decay from the gradient update. Standard Adam applies L2 regularisation via weight decay inside the moment correction, which does not actually perform true L2 regularisation on the weights. AdamW corrects this, making weight decay an independent parameter:'));
P(code([
  'optimiser = torch.optim.AdamW(',
  '    model.parameters(),',
  '    lr=3e-4,',
  '    weight_decay=1e-4,   # true L2 penalty on weights (not on moments)',
  '    betas=(0.9, 0.999)',
  ')',
]));
P(body('The learning rate schedule uses linear warmup for the first 200 epochs, followed by cosine annealing to a minimum LR of 5% of the peak. The warmup prevents the model from making large, destructive gradient updates before the EMA and moment estimates have stabilised. Without warmup, the first few epochs can produce loss spikes that permanently damage the optimisation trajectory.'));
P(code([
  'def lr_lambda(epoch):',
  '    if epoch < WARMUP_EPOCHS:      # 0 to 200: linear ramp',
  '        return epoch / WARMUP_EPOCHS',
  '    progress = (epoch - WARMUP_EPOCHS) / (N_EPOCHS - WARMUP_EPOCHS)',
  '    return max(0.05, 0.5 * (1 + cos(pi * progress)))  # cosine decay',
]));
P(PB());

// ══════════════════════════════════════════════════════════════════
// CH 4: WHY EACH TECHNOLOGY
// ══════════════════════════════════════════════════════════════════
P(H1('4. Why Each Technology Was Chosen'));
P(bodyN('The table below summarises the rationale for each technology selection:'));
P(SP(80));
P(new Table({ width:{ size:CW, type:WidthType.DXA }, columnWidths:[1900,2480,4980],
  rows:[
    thead(['Technology', 'Alternative Considered', 'Reason for Choice']),
    ...([
      ['FPS (N=512)', 'Random sampling', 'Guarantees spatial coverage; random sampling clusters in high-density regions, missing tissue boundaries'],
      ['Velocity context', 'Raw absolute frames', 'Motion-focused; invariant to absolute tissue position; enables cross-video generalisation'],
      ['Residual ΔG prediction', 'Absolute G_{t+1} prediction', 'Tractable learning target (CD~0.004); absolute prediction failed catastrophically (CD=2.3)'],
      ['Temporal 80/20 split', 'Random split', 'Prevents future data leakage; random split would leak nearby frames into validation'],
      ['disp_scale normalisation', 'No normalisation', 'Maps residuals to [-1,1]; tiny raw values waste the diffusion model\'s dynamic range'],
      ['DDPM ε-prediction', 'V-prediction (used in v1)', 'Simpler; v-prediction caused the incorrect DDIM formula in v1; ε-prediction is more robust'],
      ['T=200 timesteps', 'T=1000 (standard)', 'Tissue displacements are tiny; fewer steps = faster training; sufficient quality for small motions'],
      ['Cosine β schedule', 'Linear β schedule', 'Slower signal destruction at extremes; linear schedule destroys too much signal too quickly'],
      ['Pre-norm Transformer', 'Post-norm Transformer', 'More stable training; prevents gradient explosion in early epochs; critical for small datasets'],
      ['EMA (decay=0.999)', 'No EMA', 'Smooths stochastic weight fluctuations; significantly improves inference quality with no training cost'],
      ['DDIM (50 steps)', 'Full DDPM (200 steps)', '4× faster inference; near-identical quality for simple displacement distributions'],
      ['AdamW', 'Standard Adam', 'True L2 weight decay; Adam\'s weight decay is mathematically incorrect and underfits on small datasets'],
      ['Warmup 200 epochs', 'Fixed LR from epoch 0', 'Prevents destructive large updates before moment estimates stabilise; critical for stable convergence'],
    ]).map(([t, alt, reason], i) => trow([{ text:t, bold:true, color:i<7?TEAL:PURPLE }, alt, reason], i%2===0))
  ]
}));
P(PB());

// ══════════════════════════════════════════════════════════════════
// CH 5: INTERACTION DIAGRAM
// ══════════════════════════════════════════════════════════════════
P(H1('5. Technology Interaction Diagram'));
P(bodyN('The diagram below shows how all Module 4 and 5 technologies connect in the data and computation flow:'));
P(SP(120));

// Text-based flow diagram using tables
const flowBox = (text, color=TEAL, bg='F0F7F7') => new TableCell({
  borders:cbs(color),
  shading:{ fill:bg, type:ShadingType.CLEAR },
  margins:{ top:100, bottom:100, left:120, right:120 },
  verticalAlign:VerticalAlign.CENTER,
  children:[new Paragraph({ children:[R(text, { bold:true, color:color, size:19 })], alignment:AlignmentType.CENTER, spacing:{ before:0, after:0 } })]
});
const arrowCell = () => new TableCell({
  borders:cbs('FFFFFF'),
  shading:{ fill:WHITE, type:ShadingType.CLEAR },
  margins:{ top:0,bottom:0,left:60,right:60 },
  verticalAlign:VerticalAlign.CENTER,
  children:[new Paragraph({ children:[R('→', { size:24, color:'8090A0' })], alignment:AlignmentType.CENTER, spacing:{ before:0, after:0 } })]
});

// M4 flow
P(new Paragraph({ children:[R('MODULE 4 — Data Construction Flow', { bold:true, size:22, color:TEAL })], spacing:{ before:0, after:80 } }));
P(new Table({ width:{ size:CW, type:WidthType.DXA }, columnWidths:[1600,320,1560,320,1560,320,1560,1120],
  rows:[new TableRow({ children:[
    flowBox('Trajectory T\n∈ ℝ²¹⁹×⁵⁷⁹⁴³×³', TEAL, 'EAF4F4'),
    arrowCell(),
    flowBox('FPS\n→ 512 anchors\n(opacity-weighted)', TEAL, 'EAF4F4'),
    arrowCell(),
    flowBox('Normalise\n(centre+scale\nframe-0 stats)', TEAL, 'EAF4F4'),
    arrowCell(),
    flowBox('Velocity frames\nΔG_vel = G_t−G_{t-1}', TEAL, 'EAF4F4'),
    flowBox('Multi-step pairing\nPRED_STEPS=[1,2,3]\n5,901 windows', TEAL, 'D5EFEF'),
  ]})]
}));
P(new Paragraph({ children:[R('          ↓ disp_scale normalise residuals ↓', { size:18, color:'3A7A7A', italics:true })], spacing:{ before:60, after:60 } }));

// M5 flow
P(new Paragraph({ children:[R('MODULE 5 — Training & Inference Flow', { bold:true, size:22, color:PURPLE })], spacing:{ before:80, after:80 } }));
P(new Table({ width:{ size:CW, type:WidthType.DXA }, columnWidths:[1300,260,1200,260,1300,260,1300,260,1060,1100,1020],
  rows:[new TableRow({ children:[
    flowBox('Velocity ctx\n(3 frames)', PURPLE, 'F0EAF8'),
    arrowCell(),
    flowBox('GeomMLP\n→(B,3,256)', PURPLE, 'F0EAF8'),
    arrowCell(),
    flowBox('Transformer\nEncoder\n(3 layers,pre-norm)', PURPLE, 'F0EAF8'),
    arrowCell(),
    flowBox('Cross-Attn\n+ Last frame\ncontext', PURPLE, 'F0EAF8'),
    arrowCell(),
    flowBox('ε̂ MLP\n→(B,512,3)', PURPLE, 'F0EAF8'),
    flowBox('EMA\nweights\nDecay=0.999', PURPLE, 'EAE0F5'),
    flowBox('DDIM\n50 steps\nĜ=G_t+ΔĜ·sc', PURPLE, 'EAE0F5'),
  ]})]
}));
P(SP(80));
P(new Paragraph({ children:[R('Noisy residual x_t = √ᾱ_t · ΔG_norm + √(1−ᾱ_t) · ε  ───  fed into Cross-Attn as query  ───  SinusoidalPE(t) injected additively', { size:18, color:GREY, italics:true })], alignment:AlignmentType.CENTER, spacing:{ before:0, after:0 } }));

P(SP(160));
P(infoBox('✅', 'Summary',
  'Module 4 uses FPS, velocity encoding, residual normalisation, multi-step augmentation, and temporal splitting to produce a clean self-supervised dataset of 5,901 displacement-prediction windows. Module 5 uses a residual DDPM with GeomMLP encoders, Transformer context aggregation, cross-attention conditioning, EMA, DDIM inference, and AdamW with warmup to learn tissue deformation dynamics from this dataset without any clinical labels.',
  TEAL, 'F0F7F7'));

// ══════════════════════════════════════════════════════════════════
// BUILD
// ══════════════════════════════════════════════════════════════════
const doc = new Document({
  numbering:{ config:[{ reference:'bullets', levels:[{
    level:0, format:LevelFormat.BULLET, text:'•', alignment:AlignmentType.LEFT,
    style:{ paragraph:{ indent:{ left:540, hanging:260 } } }
  }]}]},
  styles:{
    default:{ document:{ run:{ font:'Calibri', size:22 } } },
    paragraphStyles:[
      { id:'Heading1', name:'Heading 1', basedOn:'Normal', next:'Normal',
        run:{ size:28, bold:true, font:'Calibri' },
        paragraph:{ spacing:{ before:480, after:160 }, outlineLevel:0 } },
      { id:'Heading2', name:'Heading 2', basedOn:'Normal', next:'Normal',
        run:{ size:24, bold:true, font:'Calibri' },
        paragraph:{ spacing:{ before:320, after:100 }, outlineLevel:1 } },
      { id:'Heading3', name:'Heading 3', basedOn:'Normal', next:'Normal',
        run:{ size:22, bold:true, italics:true, font:'Calibri' },
        paragraph:{ spacing:{ before:200, after:80 }, outlineLevel:2 } },
    ]
  },
  sections:[{
    properties:{ page:{ size:{ width:12240, height:15840 }, margin:{ top:1440, right:1440, bottom:1440, left:1440 } } },
    headers:{ default: new Header({ children:[
      new Paragraph({
        tabStops:[{ type:TabStopType.RIGHT, position:9360 }],
        border:{ bottom:{ style:BorderStyle.SINGLE, size:2, color:'D0DAE4', space:1 } },
        children:[
          R('Module 4 & 5 Technology Reference  ·  FAST NUCES Islamabad', { size:18, italics:true, color:GREY }),
          R('\tFYP 2025–2026', { size:18, italics:true, color:GREY }),
        ],
        spacing:{ before:0, after:120 }
      })
    ]})},
    footers:{ default: new Footer({ children:[
      new Paragraph({
        border:{ top:{ style:BorderStyle.SINGLE, size:2, color:'D0DAE4', space:1 } },
        children:[R('4D Surgical Scene Reconstruction  ·  Ahmad Rashad · Shehroz Kashif · Humair Khan  ·  Supervised by Dr Akhtar Jamil', { size:18, italics:true, color:GREY })],
        alignment:AlignmentType.CENTER, spacing:{ before:120, after:0 }
      })
    ]})},
    children: ch
  }]
});

Packer.toBuffer(doc).then(buf => {
  const path = require('path');
  const out = path.join(__dirname, 'M4_M5_Technology_Document.docx');
  fs.writeFileSync(out, buf);
  console.log('Done');
});