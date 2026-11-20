# ════════════════════════════════════════════════════════════════════
# CLOSE-OUT AUDIT — what is already measured, and what the paper is still missing
#
# Read-only. Opens nothing but the result files each module writes, and reports which
# manuscript quantities each one unlocks. Run it to decide what still has to be run,
# rather than re-running an hour of evaluation to find out it was already done.
# ════════════════════════════════════════════════════════════════════
import os, json, textwrap, time

if 'DRIVE_BASE' not in globals():
    DRIVE_BASE = '/content/drive/MyDrive/EndoGaussian'
    if not os.path.isdir(DRIVE_BASE):
        from google.colab import drive
        drive.mount('/content/drive')

WANT = [
    ('module4_combined/module4_metadata.json', 'Module 4',
     'dataset table, window size, train/val counts'),
    ('module5/diffusion_ckpt_best.pth', 'Module 5',
     'the weights every other module loads'),
    ('module5/module5_quality_tests.json', 'Module 5 QT cell',
     'QT1-QT6 on the full validation set'),
    ('module6/module6_metrics.json', 'Module 6',
     'oursCD, oursMPE, accuracyVerdict, the 3-method comparison, rollout, uncertainty'),
    ('module7/module7_comparison.json', 'Module 7',
     'the 11-method table: nMethods, oursRank, bestMethod, bestMPE'),
    ('module8/module8_hpo.json', 'Module 8',
     'the ablation section (Studies A, B, C)'),
    ('module9_demo/demo_data.js', 'Module 9',
     'the clinical demonstrator'),
]

print('=' * 72)
print('  CLOSE-OUT AUDIT')
print('=' * 72)
print(f'  {DRIVE_BASE}\n')

missing = []
for rel, who, unlocks in WANT:
    p = f'{DRIVE_BASE}/{rel}'
    if os.path.exists(p):
        sz = os.path.getsize(p)
        age = (time.time() - os.path.getmtime(p)) / 3600.0
        when = f'{age:.1f} h ago' if age < 48 else f'{age / 24:.1f} days ago'
        unit = f'{sz / 1e6:.1f} MB' if sz >= 1e6 else f'{sz / 1e3:.0f} KB'
        print(f'  [ok]      {rel}')
        print(f'            {unit:>9}  written {when}')
    else:
        missing.append((rel, who, unlocks))
        print(f'  [MISSING] {rel}')
        print(f'            {who} has not written this — it unlocks: {unlocks}')
    print()

# Module 8 is the one file that can be PARTIALLY complete, because each study is saved
# as it finishes. Report which studies are actually in there rather than just that the
# file exists.
p8 = f'{DRIVE_BASE}/module8/module8_hpo.json'
if os.path.exists(p8):
    try:
        r8 = json.load(open(p8)).get('results', {}) or {}
        have = [n for n, k in (('A (optimiser/LR)', 'A_optim'),
                               ('B (context window)', 'B_context'),
                               ('C (backbone size)', 'C_backbone')) if r8.get(k)]
        lack = [n for n, k in (('A (optimiser/LR)', 'A_optim'),
                               ('B (context window)', 'B_context'),
                               ('C (backbone size)', 'C_backbone')) if not r8.get(k)]
        print(f'  Module 8 studies present : {", ".join(have) or "none"}')
        if lack:
            print(f'  Module 8 studies absent  : {", ".join(lack)}')
    except Exception as e:
        print(f'  Module 8 file is present but unreadable: {e}')
    print()

# Module 6's metrics are written BEFORE its visual/rendering section, so a run that
# died in the renderer still leaves a usable metrics file. Worth distinguishing.
p6 = f'{DRIVE_BASE}/module6/module6_metrics.json'
if os.path.exists(p6):
    try:
        with open(p6) as _fh:
            m6 = json.load(_fh)
        # The epoch these metrics were scored at is the one fact that decides whether
        # they describe the current checkpoint or a superseded one. Module 6 records it
        # at the top level as 'checkpoint_epoch' and names the file in 'checkpoint';
        # earlier revisions nested both inside a dict. Read either shape, and say so
        # outright when neither is present rather than printing nothing.
        _ck = m6.get('checkpoint')
        _ep = m6.get('checkpoint_epoch')
        if _ep is None and isinstance(_ck, dict):
            _ep = _ck.get('epoch')
        print(f'  Module 6 scored epoch    : {_ep if _ep is not None else "NOT RECORDED"}')
        if isinstance(_ck, str):
            print(f'  Module 6 checkpoint      : {_ck}')
        # Print EVERY key. A truncated list is indistinguishable from a complete one,
        # so it invites the conclusion that a section never ran when it did.
        _keys = sorted(m6)
        print(f'  Module 6 metrics keys    : {len(_keys)} total')
        for _line in textwrap.wrap(', '.join(_keys), 66):
            print(f'      {_line}')
        _late = [s for s in ('rollout', 'timeline', 'visual', 'quality_tests')
                 if s not in m6]
        for _sec in _late:
            print(f'      absent: "{_sec}" -- that section did not finish')
    except Exception as e:
        print(f'  Module 6 file is present but unreadable: {e}')
    print()

print('=' * 72)
if missing:
    print('  STILL TO RUN')
    for rel, who, unlocks in missing:
        print(f'    {who:<18} -> {unlocks}')
else:
    print('  Everything the manuscript reads is on Drive.')
print('=' * 72)
print('\n  Next: zip the result files down to the laptop, which is where the paper')
print('  is built. One archive, nothing large:')
print('''
import shutil, os
os.makedirs('/content/closeout', exist_ok=True)
for rel in ['module4_combined/module4_metadata.json',
            'module5/module5_quality_tests.json',
            'module6/module6_metrics.json',
            'module7/module7_comparison.json',
            'module8/module8_hpo.json']:
    src = f'{DRIVE_BASE}/{rel}'
    if os.path.exists(src):
        dst = f'/content/closeout/{rel}'
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
shutil.make_archive('/content/closeout_results', 'zip', '/content/closeout')
from google.colab import files; files.download('/content/closeout_results.zip')
''')
