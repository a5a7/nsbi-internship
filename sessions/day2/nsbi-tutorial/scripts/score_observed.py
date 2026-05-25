#!/usr/bin/env python3
"""
score_observed.py
─────────────────
Score observed_0.csv with CARL and EveNet (sbi/sig models).
Calibration factors are derived from bkg_train (slim CSV, same as run_diagnostics.py).
Output: ~/observed_scores.pt  — used by run_measurement.py

Run (WSL, nsbi-venv activated):
  python3 /mnt/c/Users/Unnat/2025-lbnl/sessions/day2/nsbi-tutorial/scripts/score_observed.py
"""

import sys, gc, logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────────────
HOME        = Path.home()
CHECKPOINTS = HOME / 'checkpoints'
NSBI_DIR    = Path('/mnt/c/Users/Unnat/2025-lbnl/sessions/day2/nsbi-tutorial')
EVENET_REPO = Path('/mnt/c/Users/Unnat/2025-lbnl/EveNet-Lite-main')
OBS_CSVS    = [
    Path('/mnt/c/Users/Unnat/2025-lbnl/observed_0.csv'),
    Path('/mnt/c/Users/Unnat/2025-lbnl/observed_1.csv'),
    Path('/mnt/c/Users/Unnat/2025-lbnl/observed_2.csv'),
    Path('/mnt/c/Users/Unnat/2025-lbnl/observed_3.csv'),
    Path('/mnt/c/Users/Unnat/2025-lbnl/observed_4.csv'),
]
OUT_PT      = HOME / 'observed_scores.pt'

for p in (str(NSBI_DIR), str(EVENET_REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

from models.carl import CARL
from evenet_lite import EvenetLiteClassifier

# ── Constants ──────────────────────────────────────────────────────────────────
CARL_FEATURES = [
    'l1_pt','l1_eta','l1_phi','l1_energy',
    'l2_pt','l2_eta','l2_phi','l2_energy',
    'l3_pt','l3_eta','l3_phi','l3_energy',
    'l4_pt','l4_eta','l4_phi','l4_energy',
]
LEPTON_PREFIXES = ['l1','l2','l3','l4']

EVENET_FEATURE_NAMES = {
    'x':       ['pt', 'energy', 'eta', 'phi'],
    'globals': ['m4l', 'sum_pt', 'sum_energy'],
}

CARL_SBI_CKPT   = CHECKPOINTS / 'CARL_sbi_over_bkg_epoch=69-val_loss=0.69.ckpt'
CARL_SIG_CKPT   = CHECKPOINTS / 'CARL_sig_over_bkg_epoch=57-train_loss=0.41.ckpt'
EVENET_SBI_CKPT = CHECKPOINTS / 'EveNet_sbi_over_bkg-val_loss-epoch0020-0.6926.pt'
EVENET_SIG_CKPT = CHECKPOINTS / 'EveNet_sig_over_bkg-val_loss-epoch0007-0.4123.pt'

N_CALIB  = 50_000   # bkg events for calibration (80% train split, same as run_diagnostics)
VAL_FRAC = 0.2
SEED     = 42


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Model loaders
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_carl(path):
    ckpt = torch.load(path, map_location='cpu')
    hp = ckpt['hyper_parameters']
    hp.pop('_instantiator', None)
    m = CARL(**hp)
    m.load_state_dict(ckpt['state_dict'])
    m.eval()
    return m


def load_evenet(path):
    clf = EvenetLiteClassifier(
        class_labels=['bkg', 'num'], device='cpu', num_workers=0,
        global_input_dim=3, sequential_input_dim=4,
    )
    clf.load_checkpoint(str(path), feature_names=EVENET_FEATURE_NAMES, map_location='cpu')
    return clf


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Feature helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def df_to_evenet(df):
    N   = len(df)
    pt  = np.stack([df[f'{l}_pt'].to_numpy()     for l in LEPTON_PREFIXES], 1).astype(np.float32)
    en  = np.stack([df[f'{l}_energy'].to_numpy() for l in LEPTON_PREFIXES], 1).astype(np.float32)
    eta = np.stack([df[f'{l}_eta'].to_numpy()    for l in LEPTON_PREFIXES], 1).astype(np.float32)
    phi = np.stack([df[f'{l}_phi'].to_numpy()    for l in LEPTON_PREFIXES], 1).astype(np.float32)
    x   = np.stack([pt, en, eta, phi], axis=2)   # [N, 4, 4] — order matches EVENET_FEATURE_NAMES
    px  = pt * np.cos(phi); py = pt * np.sin(phi); pz = pt * np.sinh(eta)
    M2  = en.sum(1)**2 - px.sum(1)**2 - py.sum(1)**2 - pz.sum(1)**2
    m4l = np.sqrt(np.maximum(M2, 0)).astype(np.float32)
    return {
        'x':       torch.from_numpy(x),
        'x_mask':  torch.ones((N, 4), dtype=torch.float32),
        'globals': torch.from_numpy(np.stack([m4l, pt.sum(1), en.sum(1)], axis=1)),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Scoring
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def score_carl(model, X_scaled, batch=2048):
    Xt = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        out = [model(Xt[i:i+batch]).flatten() for i in range(0, len(Xt), batch)]
    return torch.cat(out).clamp(1e-7, 1 - 1e-7)


def score_evenet(clf, feats, label, batch_size=512):
    N = len(feats['x'])
    n_batches = (N + batch_size - 1) // batch_size
    all_logits = []
    print(f'  [{label}] {N:,} events: ', end='', flush=True)
    logging.disable(logging.INFO)
    try:
        with torch.no_grad():
            for b, start in enumerate(range(0, N, batch_size)):
                batch = {k: v[start:start + batch_size] for k, v in feats.items()}
                all_logits.append(clf.predict(batch, batch_size=len(batch['x'])))
                if (b + 1) % 10 == 0 or b == n_batches - 1:
                    print('=', end='', flush=True)
    finally:
        logging.disable(logging.NOTSET)
    print()
    return torch.softmax(torch.cat(all_logits), dim=-1)[:, 1].clamp(1e-7, 1 - 1e-7).cpu()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    # ── Load bkg slim CSV for calibration (same 80/20 split as run_diagnostics) ─
    print('Loading bkg slim CSV for calibration ...')
    bkg_all = pd.read_csv(HOME / 'ggzz4l_bkg_slim.csv', nrows=N_CALIB)
    idx       = np.random.default_rng(SEED).permutation(len(bkg_all))
    n_val     = int(len(bkg_all) * VAL_FRAC)
    bkg_train = bkg_all.iloc[idx[n_val:]].reset_index(drop=True)
    print(f'  {len(bkg_train):,} bkg_train events for calibration')

    wt_bt = torch.tensor(bkg_train['wt'].to_numpy(), dtype=torch.float32)

    # ── Load observed data (combine all files) ─────────────────────────────────
    print(f'\nLoading {len(OBS_CSVS)} observed file(s) ...')
    obs_frames = []
    for p in OBS_CSVS:
        df = pd.read_csv(p)
        print(f'  {p.name}: {len(df):,} events, N_obs={df["n"].sum():.2f}')
        obs_frames.append(df)
    obs = pd.concat(obs_frames, ignore_index=True)
    n_obs = torch.tensor(obs['n'].to_numpy(), dtype=torch.float32)
    N_obs = n_obs.sum().item()
    print(f'  Combined: {len(obs):,} events, N_obs = {N_obs:.2f}')

    # ── Load models ────────────────────────────────────────────────────────────
    print('\nLoading models ...')
    logging.disable(logging.INFO)
    carl_sbi   = load_carl(CARL_SBI_CKPT)
    carl_sig   = load_carl(CARL_SIG_CKPT)
    evenet_sbi = load_evenet(EVENET_SBI_CKPT)
    evenet_sig = load_evenet(EVENET_SIG_CKPT)
    logging.disable(logging.NOTSET)
    print('  All 4 checkpoints loaded.')

    # ── StandardScaler for CARL (fit on bkg_train only) ───────────────────────
    # Note: run_diagnostics.py fits on concat(bkg_train, sbi_train, sig_train);
    # here we use only bkg_train to avoid loading the other CSVs. The difference
    # is small — the scaler is used for inference, not training.
    scaler = StandardScaler()
    scaler.fit(bkg_train[CARL_FEATURES].to_numpy())

    def Xf(df): return scaler.transform(df[CARL_FEATURES].to_numpy())

    # ── CARL calibration on bkg_train ─────────────────────────────────────────
    print('\n── CARL calibration ──────────────────────────────────────────────')
    s_bt_carl_sbi = score_carl(carl_sbi, Xf(bkg_train))
    s_bt_carl_sig = score_carl(carl_sig, Xf(bkg_train))

    def compute_C(s, wt):
        return (wt.sum() / (wt * s / (1 - s)).sum()).item()

    C_carl_sbi = compute_C(s_bt_carl_sbi, wt_bt)
    C_carl_sig = compute_C(s_bt_carl_sig, wt_bt)
    print(f'  C_carl_sbi = {C_carl_sbi:.6f}')
    print(f'  C_carl_sig = {C_carl_sig:.6f}')

    # ── EveNet calibration on bkg_train ───────────────────────────────────────
    print('\n── EveNet calibration ────────────────────────────────────────────')
    ev_bt = df_to_evenet(bkg_train)
    s_bt_eve_sbi = score_evenet(evenet_sbi, ev_bt, 'bkg_train/sbi')
    s_bt_eve_sig = score_evenet(evenet_sig, ev_bt, 'bkg_train/sig')
    del ev_bt; gc.collect()

    C_eve_sbi = compute_C(s_bt_eve_sbi, wt_bt)
    C_eve_sig = compute_C(s_bt_eve_sig, wt_bt)
    print(f'  C_eve_sbi  = {C_eve_sbi:.6f}')
    print(f'  C_eve_sig  = {C_eve_sig:.6f}')

    # ── Score observed events ──────────────────────────────────────────────────
    print('\n── Scoring observed_0.csv ────────────────────────────────────────')
    s_obs_carl_sbi = score_carl(carl_sbi, Xf(obs))
    print(f'  CARL sbi: done')
    s_obs_carl_sig = score_carl(carl_sig, Xf(obs))
    print(f'  CARL sig: done')

    ev_obs = df_to_evenet(obs)
    s_obs_eve_sbi = score_evenet(evenet_sbi, ev_obs, 'obs/sbi')
    s_obs_eve_sig = score_evenet(evenet_sig, ev_obs, 'obs/sig')
    del ev_obs; gc.collect()

    # ── Apply calibration: r(x) = s(x)/(1-s(x)) * C ──────────────────────────
    def r_calib(s, C): return s / (1 - s) * C

    r_sbi_carl   = r_calib(s_obs_carl_sbi, C_carl_sbi)
    r_sig_carl   = r_calib(s_obs_carl_sig, C_carl_sig)
    r_sbi_evenet = r_calib(s_obs_eve_sbi,  C_eve_sbi)
    r_sig_evenet = r_calib(s_obs_eve_sig,  C_eve_sig)

    print(f'\nCalibrated r stats (on observed):')
    for name, r in [('r_sbi_carl', r_sbi_carl), ('r_sig_carl', r_sig_carl),
                    ('r_sbi_evenet', r_sbi_evenet), ('r_sig_evenet', r_sig_evenet)]:
        print(f'  {name}: mean={r.mean():.4f}  median={r.median():.4f}  '
              f'min={r.min():.4f}  max={r.max():.4f}')

    # ── Save ──────────────────────────────────────────────────────────────────
    out = {
        'r_sbi_over_bkg_carl':   r_sbi_carl,
        'r_sig_over_bkg_carl':   r_sig_carl,
        'r_sbi_over_bkg_evenet': r_sbi_evenet,
        'r_sig_over_bkg_evenet': r_sig_evenet,
        'n': n_obs,
        'n_files': len(OBS_CSVS),   # number of combined experiments (used for rate term)
        'C_carl_sbi':   torch.tensor(C_carl_sbi),
        'C_carl_sig':   torch.tensor(C_carl_sig),
        'C_eve_sbi':    torch.tensor(C_eve_sbi),
        'C_eve_sig':    torch.tensor(C_eve_sig),
    }
    torch.save(out, OUT_PT)
    print(f'\nSaved → {OUT_PT}')
    print(f'  N_obs = {N_obs:.2f}')


if __name__ == '__main__':
    main()
