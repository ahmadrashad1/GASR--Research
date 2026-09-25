import os, sys, time, shutil, subprocess, tarfile

# ════════════════════════════════════════════════════════════════════
# GSPLAT SETUP — install the rasterizer and keep its compiled CUDA
# extension on Drive so later sessions do not rebuild it
#
# `pip install gsplat` takes seconds. The cost is what happens on first use: gsplat builds
# a CUDA extension with nvcc, which took 868 s (14.5 min) in this project's Module 2 run.
# Colab throws that build away with the VM, so every fresh session pays it again -- and
# Module 6 silently skips its rendered-prediction columns when gsplat is not importable.
#
# So this cell moves the build cache somewhere that survives: compile locally (fast disk),
# then archive the result to Drive. The next session restores the archive and the same
# extension loads in seconds.
#
# The cache is keyed by everything the compiled binary depends on -- gsplat version, torch
# version, CUDA version, Python version and GPU compute capability. A T4 archive will not
# be handed to an A100, and a torch upgrade invalidates the key rather than loading a
# binary built against different headers. If a restored cache fails to load for any reason,
# it is discarded and rebuilt rather than left half-working.
# ════════════════════════════════════════════════════════════════════
GSPLAT_VERSION = '1.5.3'      # the version this project's Module 2 results were produced with
FORCE_REBUILD  = False        # True ignores any cached build and recompiles from scratch
EXT_DIR        = '/content/torch_extensions'   # local scratch: never compile onto Drive FUSE

print('=' * 64)
print('  GSPLAT SETUP')
print('=' * 64)

# ── Where the cache lives. DRIVE_BASE comes from the notebook's Cell 0. ──
drive_base = globals().get('DRIVE_BASE')
if drive_base and os.path.isdir(drive_base):
    cache_dir = f'{drive_base}/env_cache'
    os.makedirs(cache_dir, exist_ok=True)
else:
    cache_dir = None
    print('  ⚠️  DRIVE_BASE not set or not mounted — running without a persistent cache.')
    print('      Run Cell 0 (GPU check & Drive mount) first to avoid recompiling every session.')

# torch_extensions must be redirected BEFORE gsplat is imported: the path is read when the
# extension is loaded, so setting it afterwards silently has no effect.
os.makedirs(EXT_DIR, exist_ok=True)
os.environ['TORCH_EXTENSIONS_DIR'] = EXT_DIR

import torch
torch_before = torch.__version__

# ── 1. Install ────────────────────────────────────────────────────────
try:
    import gsplat
    have = getattr(gsplat, '__version__', 'unknown')
except ImportError:
    have = None

if have == GSPLAT_VERSION:
    print(f'  gsplat {have} already installed.')
else:
    what = f'{have} -> {GSPLAT_VERSION}' if have else GSPLAT_VERSION
    print(f'  Installing gsplat {what} ...')
    r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                        f'gsplat=={GSPLAT_VERSION}'], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
        raise RuntimeError('pip install gsplat failed — see the output above.')
    # A fresh install may shadow an already-imported module object.
    for m in [m for m in list(sys.modules) if m == 'gsplat' or m.startswith('gsplat.')]:
        del sys.modules[m]
    import gsplat
    print(f'  Installed gsplat {gsplat.__version__}.')

if torch.__version__ != torch_before:
    print(f'  ⚠️  pip changed torch {torch_before} -> {torch.__version__}. Restart the runtime '
          f'and re-run this cell; a mixed torch state will fail at compile time.')

# ── 2. Environment the compiled binary is tied to ─────────────────────
if not torch.cuda.is_available():
    raise RuntimeError('No CUDA device. gsplat rasterisation is GPU-only — set '
                       'Runtime > Change runtime type > T4 GPU, then re-run.')
cap = torch.cuda.get_device_capability()
key = (f'gsplat{GSPLAT_VERSION}-pt{torch.__version__}-cu{torch.version.cuda}'
       f'-py{sys.version_info.major}.{sys.version_info.minor}-sm{cap[0]}{cap[1]}')
archive = f'{cache_dir}/{key}.tar.gz' if cache_dir else None
print(f'  GPU             : {torch.cuda.get_device_name(0)} (sm_{cap[0]}{cap[1]})')
print(f'  torch / CUDA    : {torch.__version__} / {torch.version.cuda}')
print(f'  Cache key       : {key}')

if shutil.which('nvcc') is None:
    print('  ⚠️  nvcc is not on PATH. If the build fails, install the CUDA toolkit: '
          'apt-get install -y cuda-nvcc-12-8')

# ── 3. Restore a previous build ───────────────────────────────────────
restored = False
if FORCE_REBUILD:
    shutil.rmtree(EXT_DIR, ignore_errors=True)
    os.makedirs(EXT_DIR, exist_ok=True)
    print('  FORCE_REBUILD=True — ignoring any cached build.')
elif archive and os.path.exists(archive):
    try:
        with tarfile.open(archive, 'r:gz') as tf:
            tf.extractall(EXT_DIR)
        restored = True
        print(f'  Restored cached build ({os.path.getsize(archive) / 1e6:.1f} MB) from Drive.')
    except Exception as e:
        print(f'  ⚠️  Cached build could not be unpacked ({e}) — rebuilding.')
        shutil.rmtree(EXT_DIR, ignore_errors=True)
        os.makedirs(EXT_DIR, exist_ok=True)
elif archive:
    print('  No cached build for this environment — compiling once (this is the slow part).')


# ── 4. Build / load, and prove the rasterizer actually runs ───────────
def warm_up():
    """Trigger the CUDA extension load and render one tiny frame. Rendering something is
    the only check that matters: importing gsplat succeeds long before the kernels exist."""
    from gsplat import rasterization
    n = 8
    means = torch.zeros(n, 3, device='cuda')
    means[:, 2] = 2.0                                     # in front of the camera
    quats = torch.zeros(n, 4, device='cuda'); quats[:, 0] = 1.0
    scales = torch.full((n, 3), 0.01, device='cuda')
    opac = torch.full((n,), 0.5, device='cuda')
    colors = torch.full((n, 3), 0.5, device='cuda')
    viewmats = torch.eye(4, device='cuda')[None]
    Ks = torch.tensor([[[100., 0, 50], [0, 100., 50], [0, 0, 1]]], device='cuda')
    out, _, _ = rasterization(means=means, quats=quats, scales=scales, opacities=opac,
                              colors=colors, viewmats=viewmats, Ks=Ks,
                              width=100, height=100, render_mode='RGB')
    return tuple(out.shape)


print('  Loading CUDA extension ...' + ('' if restored else '  (first build takes ~15 min)'))
t0 = time.time()
try:
    shape = warm_up()
except Exception as e:
    if not restored:
        raise
    # A restored cache that cannot load is worse than none: wipe it and build clean.
    print(f'  ⚠️  Restored build failed to load ({type(e).__name__}) — discarding and rebuilding.')
    shutil.rmtree(EXT_DIR, ignore_errors=True)
    os.makedirs(EXT_DIR, exist_ok=True)
    if archive and os.path.exists(archive):
        os.remove(archive)
    restored = False
    t0 = time.time()
    shape = warm_up()
elapsed = time.time() - t0
print(f'  ✅ Rasterizer ready in {elapsed:.1f}s — test render {shape}')

# ── 5. Save the build for next time ───────────────────────────────────
if archive and not restored:
    tmp = f'{EXT_DIR}/_cache.tar.gz'          # write locally, then copy: Drive FUSE dislikes
    with tarfile.open(tmp, 'w:gz') as tf:     # being written to incrementally
        for name in os.listdir(EXT_DIR):
            if name != '_cache.tar.gz':
                tf.add(os.path.join(EXT_DIR, name), arcname=name)
    shutil.move(tmp, archive)
    print(f'  Cached build to {archive} ({os.path.getsize(archive) / 1e6:.1f} MB)')
    print(f'  Later sessions will restore it in seconds instead of rebuilding.')
elif restored:
    print(f'  Build came from cache — {archive}')

print('\n  → Module 6 can now render predicted frames. Re-run the Module 6 cell: the '
      'side-by-side figure gains its 4DGS reconstruction, predicted-frame and error columns.')
print('=' * 64)
