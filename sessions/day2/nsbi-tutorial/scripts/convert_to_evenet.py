#!/usr/bin/env python3
"""
convert_to_evenet.py
────────────────────
Converts NSBI slim CSVs into EveNet-Lite tensor format (.pt files).

Input files (slim CSVs in WSL home directory):
    ~/ggzz4l_sbi_slim.csv
    ~/ggzz4l_sig_slim.csv
    ~/ggzz4l_bkg_slim.csv

Output files:
    ~/ggzz4l_sbi_evenet.pt
    ~/ggzz4l_sig_evenet.pt
    ~/ggzz4l_bkg_evenet.pt

Each .pt file is a dict:
    x          [N, 4, 7]   per-lepton features (see feature order below)
    mask       [N, 4]      all-ones (no padding; always exactly 4 leptons)
    globals    [N, 10]     event-level features
    weights    [N]         raw event weights (wt column)
    msq_sig_sm [N]
    msq_int_sm [N]
    msq_sbi_sm [N]
    msq_bkg_sm [N]

━━━ FEATURE ORDER (must match DEFAULT_FEATURE_NAMES in classifier.py) ━━━━━━━

x[:, :, 0]  energy      l1-l4 energy column — raw GeV (normalizer applies log1p)
x[:, :, 1]  pt          l1-l4 pt column     — raw GeV (normalizer applies log1p)
x[:, :, 2]  eta         l1-l4 eta column    ← PET local_point_index[0]
x[:, :, 3]  phi         l1-l4 phi column    ← PET local_point_index[1]
x[:, :, 4]  isBTag      0 for all leptons
x[:, :, 5]  isLepton    1 for all leptons
x[:, :, 6]  charge      0 — charge not recoverable from sorted l1-l4 columns

globals[:, 0]  met      = 0      (no MET in ZZ→4l)
globals[:, 1]  met_phi  = 0
globals[:, 2]  nLepton  = 4      (constant)
globals[:, 3]  nbJet    = 0      (constant)
globals[:, 4]  nJet     = 0      (constant)
globals[:, 5]  HT       = Σ l_pt (variable)
globals[:, 6]  HT_lep   = HT     (all particles are leptons)
globals[:, 7]  M_all    = 4l invariant mass computed from p3-p6 Cartesian
                          (order-independent; more numerically stable)
globals[:, 8]  M_leps   = M_all  (all particles are leptons)
globals[:, 9]  M_bjets  = 0      (no b-jets)

━━━ NORMALISATION NOTE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EveNet-Lite's "log_normalize" applies log1p internally — pass raw values.
Constant features (met, nbJet, …) would get std≈0 under log_normalize.
Use EVENET_NORMALIZATION_RULES (printed at the end) to override those to "none".

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Usage (WSL, nsbi-venv activated):
    python3 /mnt/c/Users/Unnat/2025-lbnl/sessions/day2/nsbi-tutorial/scripts/convert_to_evenet.py
"""

import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path

HOME  = Path.home()
NAMES = ['sbi', 'sig', 'bkg']

LEPTON_PREFIXES  = ['l1', 'l2', 'l3', 'l4']    # sorted by pT — from slim CSV
MASS_PARTICLES   = ['p3', 'p4', 'p5', 'p6']     # Cartesian — for M_all only

REQUIRED_COLS = (
    [f'{l}_{q}' for l in LEPTON_PREFIXES for q in ('pt', 'eta', 'phi', 'energy')]
    + [f'{p}_{q}' for p in MASS_PARTICLES for q in ('px', 'py', 'pz', 'E')]
    + ['wt', 'msq_sig_sm', 'msq_int_sm', 'msq_sbi_sm', 'msq_bkg_sm']
)


def invariant_mass_cartesian(px, py, pz, E):
    """
    Invariant mass of a multi-particle system.
    px, py, pz, E: [N, n_particles] arrays.
    Returns [N] array.
    """
    M2 = (E.sum(axis=1)**2
          - px.sum(axis=1)**2
          - py.sum(axis=1)**2
          - pz.sum(axis=1)**2)
    return np.sqrt(np.maximum(M2, 0.0))


def convert(name: str) -> None:
    slim_path = HOME / f'ggzz4l_{name}_slim.csv'
    out_path  = HOME / f'ggzz4l_{name}_evenet.pt'

    print(f'\n[{name}]  Loading {slim_path} ...')
    if not slim_path.exists():
        print(f'  ERROR: file not found — {slim_path}', file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(slim_path)
    N  = len(df)
    print(f'  {N:,} events')

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print(f'  ERROR: missing columns: {missing}', file=sys.stderr)
        sys.exit(1)

    # ── per-lepton features from sorted l1-l4 columns: [N, 4] ────────────────
    pt_mat     = np.stack([df[f'{l}_pt'].to_numpy()     for l in LEPTON_PREFIXES], axis=1)
    eta_mat    = np.stack([df[f'{l}_eta'].to_numpy()    for l in LEPTON_PREFIXES], axis=1)
    phi_mat    = np.stack([df[f'{l}_phi'].to_numpy()    for l in LEPTON_PREFIXES], axis=1)
    energy_mat = np.stack([df[f'{l}_energy'].to_numpy() for l in LEPTON_PREFIXES], axis=1)

    isBTag_mat   = np.zeros((N, 4), dtype=np.float32)
    isLepton_mat = np.ones( (N, 4), dtype=np.float32)
    charge_mat   = np.zeros((N, 4), dtype=np.float32)  # charge unknown after pT sort

    # x: [N, 4, 7]  — feature axis order must match DEFAULT_FEATURE_NAMES
    x = np.stack(
        [
            energy_mat.astype(np.float32),   # 0 — energy
            pt_mat.astype(np.float32),        # 1 — pt
            eta_mat.astype(np.float32),       # 2 — eta  (local_point_index[0])
            phi_mat.astype(np.float32),       # 3 — phi  (local_point_index[1])
            isBTag_mat,                       # 4 — isBTag
            isLepton_mat,                     # 5 — isLepton
            charge_mat,                       # 6 — charge
        ],
        axis=2,
    )

    # mask: [N, 4] — all ones
    mask = np.ones((N, 4), dtype=np.float32)

    # ── M_all from p3-p6 Cartesian 4-vectors (order-independent) ─────────────
    px_mat = np.stack([df[f'{p}_px'].to_numpy() for p in MASS_PARTICLES], axis=1)
    py_mat = np.stack([df[f'{p}_py'].to_numpy() for p in MASS_PARTICLES], axis=1)
    pz_mat = np.stack([df[f'{p}_pz'].to_numpy() for p in MASS_PARTICLES], axis=1)
    E_mat  = np.stack([df[f'{p}_E'].to_numpy()  for p in MASS_PARTICLES], axis=1)
    M_all  = invariant_mass_cartesian(px_mat, py_mat, pz_mat, E_mat).astype(np.float32)

    # ── global features [N, 10] ───────────────────────────────────────────────
    HT = pt_mat.sum(axis=1).astype(np.float32)

    globals_ = np.stack(
        [
            np.zeros(N, dtype=np.float32),       # met
            np.zeros(N, dtype=np.float32),        # met_phi
            np.full(N, 4.0, dtype=np.float32),   # nLepton
            np.zeros(N, dtype=np.float32),        # nbJet
            np.zeros(N, dtype=np.float32),        # nJet
            HT,                                   # HT
            HT.copy(),                            # HT_lep (= HT, all are leptons)
            M_all,                                # M_all
            M_all.copy(),                         # M_leps (= M_all, all are leptons)
            np.zeros(N, dtype=np.float32),        # M_bjets
        ],
        axis=1,
    )

    # ── matrix elements + weights ─────────────────────────────────────────────
    wt         = df['wt'].to_numpy().astype(np.float32)
    msq_sig_sm = df['msq_sig_sm'].to_numpy().astype(np.float32)
    msq_int_sm = df['msq_int_sm'].to_numpy().astype(np.float32)
    msq_sbi_sm = df['msq_sbi_sm'].to_numpy().astype(np.float32)
    msq_bkg_sm = df['msq_bkg_sm'].to_numpy().astype(np.float32)

    # ── sanity checks ─────────────────────────────────────────────────────────
    assert x.shape        == (N, 4, 7), f'x shape wrong: {x.shape}'
    assert mask.shape     == (N, 4),    f'mask shape wrong: {mask.shape}'
    assert globals_.shape == (N, 10),   f'globals shape wrong: {globals_.shape}'
    assert not np.isnan(x).any(),        'NaN in x'
    assert not np.isinf(x).any(),        'Inf in x'
    assert not np.isnan(globals_).any(), 'NaN in globals'
    assert (pt_mat > 0).all(),           'Non-positive pT'
    assert (energy_mat > 0).all(),       'Non-positive energy'
    assert (M_all > 0).all(),            'Non-positive M_all'

    # ── save ──────────────────────────────────────────────────────────────────
    out = {
        'x':          torch.from_numpy(x),
        'mask':       torch.from_numpy(mask),
        'globals':    torch.from_numpy(globals_),
        'weights':    torch.from_numpy(wt),
        'msq_sig_sm': torch.from_numpy(msq_sig_sm),
        'msq_int_sm': torch.from_numpy(msq_int_sm),
        'msq_sbi_sm': torch.from_numpy(msq_sbi_sm),
        'msq_bkg_sm': torch.from_numpy(msq_bkg_sm),
    }
    torch.save(out, out_path)

    print(f'  Saved  → {out_path}')
    print(f'  x:          {tuple(out["x"].shape)}')
    print(f'  mask:       {tuple(out["mask"].shape)}')
    print(f'  globals:    {tuple(out["globals"].shape)}')
    print(f'  HT range:   [{HT.min():.1f}, {HT.max():.1f}] GeV')
    print(f'  M_all range:[{M_all.min():.1f}, {M_all.max():.1f}] GeV')


def main() -> None:
    for name in NAMES:
        convert(name)

    print('\n' + '=' * 70)
    print('All files written successfully.\n')
    print('Pass this to clf.fit(normalization_rules=...):\n')
    print("""\
EVENET_NORMALIZATION_RULES = {
    "x": {
        "energy":   "log_normalize",
        "pt":       "log_normalize",
        "eta":      "normalize",
        "phi":      "normalize_uniform",
        "isBTag":   "none",
        "isLepton": "none",
        "charge":   "none",
    },
    "globals": {
        "met":     "none",        # constant 0 — std≈0 breaks log_normalize
        "met_phi": "none",        # constant 0
        "nLepton": "none",        # constant 4
        "nbJet":   "none",        # constant 0
        "nJet":    "none",        # constant 0
        "HT":      "log_normalize",
        "HT_lep":  "log_normalize",
        "M_all":   "log_normalize",
        "M_leps":  "log_normalize",
        "M_bjets": "none",        # constant 0
    },
}
""")
    print('=' * 70)


if __name__ == '__main__':
    main()
