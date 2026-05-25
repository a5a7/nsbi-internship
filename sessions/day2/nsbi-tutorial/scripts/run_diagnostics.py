#!/usr/bin/env python3
"""
run_diagnostics.py
──────────────────
Side-by-side diagnostic plots for CARL and EveNet-Lite:
  1. Reweighting histogram: D, N, (N/D)×D in m4l  [2×2 grid]
  2. Calibration curve: NSBI ŝ(x) vs MC s(x)      [2×2 grid]

Run (WSL, nsbi-venv activated):
  python3 /mnt/c/Users/Unnat/2025-lbnl/sessions/day2/nsbi-tutorial/scripts/run_diagnostics.py
"""

import sys, gc, logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
HOME        = Path.home()
CHECKPOINTS = HOME / 'checkpoints'
NSBI_DIR    = Path('/mnt/c/Users/Unnat/2025-lbnl/sessions/day2/nsbi-tutorial')
EVENET_REPO = Path('/mnt/c/Users/Unnat/2025-lbnl/EveNet-Lite-main')
PLOTS_DIR   = NSBI_DIR / 'plots'
PLOTS_DIR.mkdir(exist_ok=True)

for p in (str(NSBI_DIR), str(EVENET_REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

from models.carl import CARL
from evenet_lite import EvenetLiteClassifier

# ── Constants ─────────────────────────────────────────────────────────────────
CARL_FEATURES = [
    'l1_pt','l1_eta','l1_phi','l1_energy',
    'l2_pt','l2_eta','l2_phi','l2_energy',
    'l3_pt','l3_eta','l3_phi','l3_energy',
    'l4_pt','l4_eta','l4_phi','l4_energy',
]
LEPTON_PREFIXES = ['l1','l2','l3','l4']

EVENET_FEATURE_NAMES = {
    'x':       ['pt','energy','eta','phi'],
    'globals': ['m4l','sum_pt','sum_energy'],
}

CARL_SBI_CKPT   = CHECKPOINTS / 'CARL_sbi_over_bkg_epoch=69-val_loss=0.69.ckpt'
CARL_SIG_CKPT   = CHECKPOINTS / 'CARL_sig_over_bkg_epoch=57-train_loss=0.41.ckpt'
EVENET_SBI_CKPT = CHECKPOINTS / 'EveNet_sbi_over_bkg-val_loss-epoch0020-0.6926.pt'
EVENET_SIG_CKPT = CHECKPOINTS / 'EveNet_sig_over_bkg-val_loss-epoch0007-0.4123.pt'

N_MAX    = 50_000
SEED     = 42
VAL_FRAC = 0.2
M4L_BINS = np.linspace(180, 1000, 42)   # 41 bins, matching tutorial


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

def compute_m4l(df):
    pt  = np.stack([df[f'{l}_pt'].to_numpy()     for l in LEPTON_PREFIXES], 1).astype(np.float32)
    en  = np.stack([df[f'{l}_energy'].to_numpy() for l in LEPTON_PREFIXES], 1).astype(np.float32)
    eta = np.stack([df[f'{l}_eta'].to_numpy()    for l in LEPTON_PREFIXES], 1).astype(np.float32)
    phi = np.stack([df[f'{l}_phi'].to_numpy()    for l in LEPTON_PREFIXES], 1).astype(np.float32)
    px = pt * np.cos(phi); py = pt * np.sin(phi); pz = pt * np.sinh(eta)
    M2 = en.sum(1)**2 - px.sum(1)**2 - py.sum(1)**2 - pz.sum(1)**2
    return np.sqrt(np.maximum(M2, 0)).astype(np.float32)


def df_to_evenet(df):
    N = len(df)
    pt  = np.stack([df[f'{l}_pt'].to_numpy()     for l in LEPTON_PREFIXES], 1).astype(np.float32)
    en  = np.stack([df[f'{l}_energy'].to_numpy() for l in LEPTON_PREFIXES], 1).astype(np.float32)
    eta = np.stack([df[f'{l}_eta'].to_numpy()    for l in LEPTON_PREFIXES], 1).astype(np.float32)
    phi = np.stack([df[f'{l}_phi'].to_numpy()    for l in LEPTON_PREFIXES], 1).astype(np.float32)
    x = np.stack([pt, en, eta, phi], axis=2)  # [N, 4 leptons, 4 features]
    px = pt * np.cos(phi); py = pt * np.sin(phi); pz = pt * np.sinh(eta)
    M2 = en.sum(1)**2 - px.sum(1)**2 - py.sum(1)**2 - pz.sum(1)**2
    m4l    = np.sqrt(np.maximum(M2, 0)).astype(np.float32)
    sum_pt = pt.sum(1).astype(np.float32)
    sum_e  = en.sum(1).astype(np.float32)
    return {
        'x':       torch.from_numpy(x),
        'x_mask':  torch.ones((N, 4), dtype=torch.float32),
        'globals': torch.from_numpy(np.stack([m4l, sum_pt, sum_e], axis=1)),
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
# Plot helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_reweighting(ax, m4l_bkg, wt_bkg, r_bkg, m4l_num, wt_num, title):
    kw = dict(bins=M4L_BINS, histtype='step', density=True)
    ax.hist(m4l_bkg, weights=wt_bkg,                    label='$D$',                       **kw)
    ax.hist(m4l_num, weights=wt_num,                    label='$N$',                       **kw)
    ax.hist(m4l_bkg, weights=wt_bkg * r_bkg.numpy(),   label='$\\frac{N}{D}\\times D$',   linestyle='--', **kw)
    ax.set_xlabel('$m_{4\\ell}$ [GeV]')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)
    ax.set_title(title)


def plot_calib_curve(ax, s_bkg, s_num, wt_bkg, wt_num, title, n_bins=40):
    """
    Pool bkg + num events with class-balanced weights.
    Bin by raw classifier output s(x). In each bin compute the MC fraction
    s_mc = sum(w_num) / (sum(w_num) + sum(w_bkg)) and plot vs bin center.
    """
    s_all = np.concatenate([s_bkg.numpy(), s_num.numpy()])
    y_all = np.concatenate([np.zeros(len(s_bkg)), np.ones(len(s_num))])
    # class-balance: normalise each class to unit sum
    w_all = np.concatenate([wt_bkg / wt_bkg.sum(), wt_num / wt_num.sum()])

    bins    = np.linspace(0, 1, n_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    widths  = (bins[1:]  - bins[:-1]) / 2

    s_mc, s_mc_err = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (s_all >= lo) & (s_all < hi)
        wn = w_all[mask & (y_all == 1)]
        wd = w_all[mask & (y_all == 0)]
        Wn, Wd = wn.sum(), wd.sum()
        tot = Wn + Wd
        if tot > 0:
            p = Wn / tot
            # weighted variance
            err = np.sqrt((Wd / tot**2)**2 * (wn**2).sum() + (Wn / tot**2)**2 * (wd**2).sum())
        else:
            p, err = np.nan, 0.0
        s_mc.append(p)
        s_mc_err.append(err)

    s_mc = np.array(s_mc)
    s_mc_err = np.array(s_mc_err)
    ax.plot([0, 1], [0, 1], '--', color='grey')
    ax.errorbar(centers, s_mc, xerr=widths, yerr=s_mc_err, linestyle='none', marker='+')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('NSBI estimate, $\\hat{s}(x)$')
    ax.set_ylabel('MC estimate, $s(x)$')
    ax.set_title(title)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    # ── Data ──────────────────────────────────────────────────────────────────
    print('Loading CSVs ...')
    def load_csv(name):
        return pd.read_csv(HOME / f'ggzz4l_{name}_slim.csv', nrows=N_MAX)

    def split_df(df):
        idx = np.random.default_rng(SEED).permutation(len(df))
        n_val = int(len(df) * VAL_FRAC)
        return df.iloc[idx[n_val:]].reset_index(drop=True), df.iloc[idx[:n_val]].reset_index(drop=True)

    bkg_train, bkg_val = split_df(load_csv('bkg'))
    sbi_train, sbi_val = split_df(load_csv('sbi'))
    sig_train, sig_val = split_df(load_csv('sig'))
    print(f'  bkg {len(bkg_train)}/{len(bkg_val)} | sbi {len(sbi_train)}/{len(sbi_val)} | sig {len(sig_train)}/{len(sig_val)}')

    m4l_bkgv = compute_m4l(bkg_val)
    m4l_sbiv  = compute_m4l(sbi_val)
    m4l_sigv  = compute_m4l(sig_val)
    wt_bkgv   = bkg_val['wt'].to_numpy()
    wt_sbiv   = sbi_val['wt'].to_numpy()
    wt_sigv   = sig_val['wt'].to_numpy()
    wt_bt     = torch.tensor(bkg_train['wt'].to_numpy(), dtype=torch.float32)

    # ── Models ────────────────────────────────────────────────────────────────
    print('\nLoading models ...')
    logging.disable(logging.INFO)
    carl_sbi   = load_carl(CARL_SBI_CKPT)
    carl_sig   = load_carl(CARL_SIG_CKPT)
    evenet_sbi = load_evenet(EVENET_SBI_CKPT)
    evenet_sig = load_evenet(EVENET_SIG_CKPT)
    logging.disable(logging.NOTSET)

    # ── CARL scaler ───────────────────────────────────────────────────────────
    scaler = StandardScaler()
    scaler.fit(pd.concat([bkg_train, sbi_train, sig_train])[CARL_FEATURES].to_numpy())

    # ── CARL: score all splits ────────────────────────────────────────────────
    print('\n── CARL ──────────────────────────────────────────────────────────')
    def Xf(df): return scaler.transform(df[CARL_FEATURES].to_numpy())

    s_bt_carl_sbi = score_carl(carl_sbi, Xf(bkg_train))
    s_bt_carl_sig = score_carl(carl_sig, Xf(bkg_train))
    s_bv_carl_sbi = score_carl(carl_sbi, Xf(bkg_val))
    s_bv_carl_sig = score_carl(carl_sig, Xf(bkg_val))
    s_sv_carl_sbi = score_carl(carl_sbi, Xf(sbi_val))
    s_gv_carl_sig = score_carl(carl_sig, Xf(sig_val))
    print('  Done.')

    # ── EveNet: score all splits ──────────────────────────────────────────────
    print('\n── EveNet ────────────────────────────────────────────────────────')
    ev_bt = df_to_evenet(bkg_train)
    ev_bv = df_to_evenet(bkg_val)
    ev_sv = df_to_evenet(sbi_val)
    ev_gv = df_to_evenet(sig_val)

    print('SBI model:')
    s_bt_eve_sbi = score_evenet(evenet_sbi, ev_bt, 'bkg_train')
    s_bv_eve_sbi = score_evenet(evenet_sbi, ev_bv, 'bkg_val')
    s_sv_eve_sbi = score_evenet(evenet_sbi, ev_sv, 'sbi_val')
    print('SIG model:')
    s_bt_eve_sig = score_evenet(evenet_sig, ev_bt, 'bkg_train')
    s_bv_eve_sig = score_evenet(evenet_sig, ev_bv, 'bkg_val')
    s_gv_eve_sig = score_evenet(evenet_sig, ev_gv, 'sig_val')
    del ev_bt, ev_bv, ev_sv, ev_gv; gc.collect()

    # ── Calibration factors ───────────────────────────────────────────────────
    def compute_C(s, wt):
        return (wt.sum() / (wt * s / (1 - s)).sum()).item()

    C_carl_sbi = compute_C(s_bt_carl_sbi, wt_bt)
    C_carl_sig = compute_C(s_bt_carl_sig, wt_bt)
    C_eve_sbi  = compute_C(s_bt_eve_sbi,  wt_bt)
    C_eve_sig  = compute_C(s_bt_eve_sig,  wt_bt)
    print(f'\nCalibration factors (C = sum(wt) / sum(wt*r) on bkg_train):')
    print(f'  CARL   sbi/bkg: {C_carl_sbi:.6f}')
    print(f'  CARL   sig/bkg: {C_carl_sig:.6f}')
    print(f'  EveNet sbi/bkg: {C_eve_sbi:.6f}')
    print(f'  EveNet sig/bkg: {C_eve_sig:.6f}')

    # ── Calibrated ratios on val bkg ──────────────────────────────────────────
    def r_calib(s, C): return s / (1 - s) * C

    r_bv_carl_sbi = r_calib(s_bv_carl_sbi, C_carl_sbi)
    r_bv_carl_sig = r_calib(s_bv_carl_sig, C_carl_sig)
    r_bv_eve_sbi  = r_calib(s_bv_eve_sbi,  C_eve_sbi)
    r_bv_eve_sig  = r_calib(s_bv_eve_sig,  C_eve_sig)

    # ── Plot 1: Reweighting histograms ────────────────────────────────────────
    print('\nPlotting reweighting histograms ...')
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    plot_reweighting(axes[0,0], m4l_bkgv, wt_bkgv, r_bv_carl_sbi, m4l_sbiv, wt_sbiv, 'CARL  SBI/B')
    plot_reweighting(axes[0,1], m4l_bkgv, wt_bkgv, r_bv_eve_sbi,  m4l_sbiv, wt_sbiv, 'EveNet SBI/B')
    plot_reweighting(axes[1,0], m4l_bkgv, wt_bkgv, r_bv_carl_sig, m4l_sigv, wt_sigv, 'CARL  S/B')
    plot_reweighting(axes[1,1], m4l_bkgv, wt_bkgv, r_bv_eve_sig,  m4l_sigv, wt_sigv, 'EveNet S/B')
    plt.suptitle('Reweighting Diagnostic: D, N, and $r(x)\\times D$', fontsize=13)
    plt.tight_layout()
    out = PLOTS_DIR / 'reweighting.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved → {out}')

    # ── Plot 2: Calibration curves ────────────────────────────────────────────
    print('\nPlotting calibration curves ...')
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plot_calib_curve(axes[0,0], s_bv_carl_sbi, s_sv_carl_sbi, wt_bkgv, wt_sbiv, 'CARL  SBI/B')
    plot_calib_curve(axes[0,1], s_bv_eve_sbi,  s_sv_eve_sbi,  wt_bkgv, wt_sbiv, 'EveNet SBI/B')
    plot_calib_curve(axes[1,0], s_bv_carl_sig, s_gv_carl_sig, wt_bkgv, wt_sigv, 'CARL  S/B')
    plot_calib_curve(axes[1,1], s_bv_eve_sig,  s_gv_eve_sig,  wt_bkgv, wt_sigv, 'EveNet S/B')
    plt.suptitle('Calibration Curve: NSBI $\\hat{s}(x)$ vs MC $s(x)$', fontsize=13)
    plt.tight_layout()
    out = PLOTS_DIR / 'calibration_curves.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved → {out}')

    print('\nAll done.')


if __name__ == '__main__':
    main()
