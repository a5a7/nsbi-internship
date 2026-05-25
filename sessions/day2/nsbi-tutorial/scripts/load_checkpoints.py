#!/usr/bin/env python3
"""
load_checkpoints.py
───────────────────
Verify that both CARL and EveNet-Lite checkpoints load correctly.
Run from WSL with nsbi-venv activated:
    python3 /mnt/c/Users/Unnat/2025-lbnl/sessions/day2/nsbi-tutorial/scripts/load_checkpoints.py
"""

import sys
from pathlib import Path

HOME        = Path.home()
CHECKPOINTS = HOME / 'checkpoints'
NSBI_DIR    = Path('/mnt/c/Users/Unnat/2025-lbnl/sessions/day2/nsbi-tutorial')
EVENET_REPO = Path('/mnt/c/Users/Unnat/2025-lbnl/EveNet-Lite-main')

for p in (str(NSBI_DIR), str(EVENET_REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from models.carl import CARL
from evenet_lite import EvenetLiteClassifier

EVENET_FEATURE_NAMES = {
    'x':       ['pt', 'energy', 'eta', 'phi'],
    'globals': ['m4l', 'sum_pt', 'sum_energy'],
}

CKPTS = {
    'CARL   sbi/bkg': CHECKPOINTS / 'CARL_sbi_over_bkg_epoch=69-val_loss=0.69.ckpt',
    'CARL   sig/bkg': CHECKPOINTS / 'CARL_sig_over_bkg_epoch=57-train_loss=0.41.ckpt',
    'EveNet sbi/bkg': CHECKPOINTS / 'EveNet_sbi_over_bkg-val_loss-epoch0020-0.6926.pt',
    'EveNet sig/bkg': CHECKPOINTS / 'EveNet_sig_over_bkg-val_loss-epoch0007-0.4123.pt',
}

# ── CARL ──────────────────────────────────────────────────────────────────────
def load_carl(path):
    ckpt = torch.load(path, map_location='cpu')
    hp = ckpt['hyper_parameters']
    hp.pop('_instantiator', None)
    model = CARL(**hp)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model

# ── EveNet ────────────────────────────────────────────────────────────────────
def load_evenet(path):
    clf = EvenetLiteClassifier(
        class_labels=['bkg', 'num'],
        device='cpu',
        num_workers=0,
        global_input_dim=3,
        sequential_input_dim=4,
    )
    clf.load_checkpoint(str(path), feature_names=EVENET_FEATURE_NAMES, map_location='cpu')
    return clf

# ── Run ───────────────────────────────────────────────────────────────────────
print(f'\n{"─"*50}')
for name, path in CKPTS.items():
    print(f'\nLoading {name} ...')
    print(f'  path: {path}')
    if not path.exists():
        print(f'  ERROR: file not found')
        continue
    try:
        if 'CARL' in name:
            model = load_carl(path)
            n_params = sum(p.numel() for p in model.parameters())
            print(f'  OK — {n_params:,} parameters')
        else:
            clf = load_evenet(path)
            n_params = sum(p.numel() for p in clf.model.parameters())
            print(f'  OK — {n_params:,} parameters')
    except Exception as e:
        print(f'  ERROR: {e}')

print(f'\n{"─"*50}\nDone.')
