"""Rebuild the trained network from the training script, verbatim.

The server must not carry its own copy of the model. A copy is a second place for
the architecture to live, and the failure it produces is the worst kind available
here: weights that load, a server that starts, and predictions that quietly differ
from every number in the paper.

So the classes are not re-written here. They are lifted out of
`module5_diffusion.py` as source text and executed against constants taken from the
checkpoint's own `arch_cfg`/`config`. There is exactly one definition of
ConditionalDenoiser in this repository, and this module is a reader of it.

Importing module5_diffusion.py directly is not an option: it builds a model, an
optimiser and a data pipeline at module scope and expects Module 4's workspace to
exist. Extracting the definitions sidesteps all of that without duplicating any of it.
"""
import ast
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# Everything the server needs in order to run the network forward. Deliberately a
# short list: no training, no EMA update, no loss.
NEEDED = (
    'GeomMLP',
    'sinusoidal_embedding',
    'ConditionalDenoiser',
    'cosine_beta_schedule',
    'pred_x0_from_v',
    'pred_eps_from_v',
    'const_velocity_pred',
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING_SCRIPT = os.path.join(REPO_ROOT, 'module5_diffusion.py')


def extract_source(names=NEEDED, path=TRAINING_SCRIPT):
    """The verbatim source text of each named top-level class or function.

    Raises rather than guessing. A rename in the training script must break the
    server loudly at startup, not silently fall back to something else.
    """
    with open(path) as fh:
        text = fh.read()
    lines = text.splitlines(keepends=True)
    found = {}
    for node in ast.parse(text).body:
        name = getattr(node, 'name', None)
        if name in names and isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            found[name] = ''.join(lines[node.lineno - 1:node.end_lineno])
    missing = [n for n in names if n not in found]
    if missing:
        raise RuntimeError(
            f'{os.path.basename(path)} no longer defines {missing}. The server builds '
            f'the network from that file on purpose; fix the name here rather than '
            f'pasting a copy of the model into serve/.')
    return found


def build_namespace(arch_cfg, config, dropout=0.1):
    """Execute the extracted definitions with the checkpoint's own hyperparameters.

    Dropout is irrelevant to the result -- the server only ever runs in eval() -- but
    the modules are constructed with it, so it has to be a number.
    """
    ns = {
        'torch': torch, 'nn': nn, 'F': F, 'math': math,
        'LATENT_DIM': arch_cfg['LATENT_DIM'],
        'HIDDEN_DIM': arch_cfg['HIDDEN_DIM'],
        'N_HEADS': arch_cfg['N_HEADS'],
        'N_TF_LAYERS': arch_cfg['N_TF_LAYERS'],
        'WINDOW_SIZE': arch_cfg['WINDOW_SIZE'],
        'MAX_PRED_STEP': arch_cfg['MAX_PRED_STEP'],
        'CTX_SCALE': config['CTX_SCALE'],
        'DROPOUT': dropout,
    }
    source = extract_source()
    for name in NEEDED:
        exec(source[name], ns)
    return ns
