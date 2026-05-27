#!/usr/bin/env python3
"""
run_ablation_study.py
─────────────────────
Ablation study on the EveNet ClassificationHead.

Imports from sibling scripts:
  run_diagnostics.py  →  df_to_evenet, compute_m4l, plot_reweighting, plot_calib_curve
  run_measurement.py  →  compute_t_shape, compute_t_rate, plot_nll

Per ablation, saves to <nsbi-tutorial>/plots/ablation/<name>/:
  nll.png           — NLL breakdown curve
  calibration.png   — NSBI ŝ(x) vs MC s(x)  (S/B and SBI/B)
  reweighting.png   — m4l histograms + ratio/truth panels (S/B and SBI/B)

Run (WSL, nsbi-venv activated):
  python3 .../run_ablation_study.py
  python3 .../run_ablation_study.py --only layers_2
  python3 .../run_ablation_study.py --resume
  python3 .../run_ablation_study.py --only layers_2 --skip-training
"""

import argparse, copy, gc, json, logging, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── Paths ──────────────────────────────────────────────────────────────────────
HOME        = Path.home()
CHECKPOINTS = HOME / 'checkpoints'
NSBI_DIR    = Path('/mnt/c/Users/Unnat/2025-lbnl/sessions/day2/nsbi-tutorial')
EVENET_REPO = Path('/mnt/c/Users/Unnat/2025-lbnl/EveNet-Lite-main')
REPO_ROOT   = Path('/mnt/c/Users/Unnat/2025-lbnl')
SCRIPTS_DIR = NSBI_DIR / 'scripts'

OBS_CSVS        = [REPO_ROOT / f'observed_{i}.csv' for i in range(5)]
EVENET_SBI_CKPT = CHECKPOINTS / 'EveNet_sbi_over_bkg-val_loss-epoch0020-0.6926.pt'
EVENET_SIG_CKPT = CHECKPOINTS / 'EveNet_sig_over_bkg-val_loss-epoch0007-0.4123.pt'
XS_JSON         = REPO_ROOT / 'ggzz4l_xs.json'

ABLATION_CKPT_DIR  = HOME / 'ablation_checkpoints'
ABLATION_PLOTS_DIR = NSBI_DIR / 'plots' / 'ablation'
RESULTS_JSON       = HOME / 'ablation_results.json'

# sys.path must be set before importing sibling scripts
for p in (str(SCRIPTS_DIR), str(NSBI_DIR), str(EVENET_REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Imports from sibling scripts ──────────────────────────────────────────────
import run_diagnostics as diag   # df_to_evenet, compute_m4l, plot_reweighting, plot_calib_curve
import run_measurement as meas   # compute_t_shape, compute_t_rate, plot_nll

# ── Constants ──────────────────────────────────────────────────────────────────
EVENET_FEATURE_NAMES = {'x': ['pt', 'energy', 'eta', 'phi'],
                        'globals': ['m4l', 'sum_pt', 'sum_energy']}
NORM_RULES = {
    'x':       {'pt': 'log_normalize', 'energy': 'log_normalize',
                'eta': 'normalize',    'phi': 'normalize_uniform'},
    'globals': {'m4l': 'log_normalize', 'sum_pt': 'log_normalize',
                'sum_energy': 'log_normalize'},
}

N_CALIB    = 50_000
N_VAL_DIAG = 20_000   # events per class for diagnostic plots
MAX_TRAIN  = 100_000  # per class cap for training
VAL_FRAC   = 0.2
SEED       = 42
BATCH_SIZE = 512
SBI_EPOCHS = 15
SIG_EPOCHS = 10
PATIENCE   = 5

# ── Ablation configs ───────────────────────────────────────────────────────────
ABLATIONS = [
    {"name": "baseline",   "cls": {}},
    {"name": "layers_0",   "cls": {"num_classification_layers": 0}},
    {"name": "layers_2",   "cls": {"num_classification_layers": 2}},
    {"name": "layers_4",   "cls": {"num_classification_layers": 4}},
    {"name": "layers_8",   "cls": {"num_classification_layers": 8}},
    {"name": "hidden_64",  "cls": {"hidden_dim": 64}},
    {"name": "hidden_128", "cls": {"hidden_dim": 128}},
    {"name": "hidden_512", "cls": {"hidden_dim": 512}},
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Model helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_ablation_clf(cls_overrides: dict):
    from evenet_lite import EvenetLiteClassifier
    from evenet_lite.model import EveNetLite
    from evenet.control.global_config import DotDict

    cfg_path = EVENET_REPO / 'evenet_lite' / 'config' / 'default_network_config.yaml'
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    for k, v in cls_overrides.items():
        cfg['Classification'][k] = v

    model = EveNetLite(config=DotDict(cfg), global_input_dim=3,
                       sequential_input_dim=4, cls_label=['bkg', 'num'])
    return EvenetLiteClassifier(
        class_labels=['bkg', 'num'], device='auto',
        lr=[1e-3], module_lists=[['Classification']],
        weight_decay=1e-2, num_workers=0,
        global_input_dim=3, sequential_input_dim=4, model=model,
    )


def load_backbone_weights(clf, ckpt_path: Path) -> dict:
    raw = torch.load(str(ckpt_path), map_location='cpu')
    clf._soft_load_state_dict(raw.get('model', raw))
    return raw.get('normalizer', None)


def freeze_backbone(clf) -> None:
    for single in clf.model.models:
        for p in single.backbone.parameters():
            p.requires_grad_(False)
    n = sum(p.numel() for p in clf.model.parameters() if p.requires_grad)
    print(f'    Trainable (head only): {n:,} params')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_training_pair(csv_a: Path, csv_b: Path):
    rng = np.random.default_rng(SEED)

    def cap_load(csv, label):
        df = pd.read_csv(csv)
        if len(df) > MAX_TRAIN:
            df = df.iloc[rng.choice(len(df), MAX_TRAIN, replace=False)].reset_index(drop=True)
        feats   = diag.df_to_evenet(df)
        labels  = torch.full((len(df),), label, dtype=torch.long)
        weights = torch.tensor(df['wt'].to_numpy(), dtype=torch.float32)
        return feats, labels, weights

    fa, la, wa = cap_load(csv_a, 0)
    fb, lb, wb = cap_load(csv_b, 1)
    feats = {k: torch.cat([fa[k], fb[k]]) for k in fa}
    lbls  = torch.cat([la, lb])
    wts   = torch.cat([wa, wb])

    perm  = rng.permutation(len(lbls))
    n_val = int(len(lbls) * VAL_FRAC)
    def split(idx):
        return {k: v[idx] for k, v in feats.items()}, lbls[idx], wts[idx]
    return split(perm[n_val:]), split(perm[:n_val])


def score_silent(clf, feats: dict) -> torch.Tensor:
    """Score returning class-1 softmax prob; suppresses INFO logs."""
    N, logits = len(feats['x']), []
    logging.disable(logging.INFO)
    try:
        with torch.no_grad():
            for s in range(0, N, BATCH_SIZE):
                b = {k: v[s:s+BATCH_SIZE] for k, v in feats.items()}
                logits.append(clf.predict(b, batch_size=len(b['x'])))
    finally:
        logging.disable(logging.NOTSET)
    return torch.softmax(torch.cat(logits), dim=-1)[:, 1].clamp(1e-7, 1-1e-7).cpu()


def compute_C(s: torch.Tensor, wt: torch.Tensor) -> float:
    return (wt.sum() / (wt * s / (1 - s)).sum()).item()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Training
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train_head(clf, train_data, val_data, normalizer_state,
               epochs: int, ckpt_dir: Path) -> float:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    clf.fit(
        train_data=train_data, val_data=val_data,
        feature_names=copy.deepcopy(EVENET_FEATURE_NAMES),
        normalization_rules=copy.deepcopy(NORM_RULES),
        normalization_stats=normalizer_state,
        epochs=epochs, batch_size=BATCH_SIZE,
        checkpoint_path=str(ckpt_dir / 'best'),
        save_top_k=1, monitor_metric='val_loss', minimize_metric=True,
        early_stop_metric='val_loss', early_stop_patience=PATIENCE,
        early_stop_minimize=True,
    )
    best_loss = float('nan')
    for f in sorted(ckpt_dir.glob('best*.pt')):
        try:
            v = (torch.load(str(f), map_location='cpu').get('extra') or {}).get('monitored_metric')
            if v is not None and (np.isnan(best_loss) or v < best_loss):
                best_loss = float(v)
        except Exception:
            pass
    return best_loss


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Plot: reweighting with ratio panels (new — wraps diag.plot_reweighting data)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _rw_panel(ax_top, ax_bot, m4l_bkg, wt_bkg, r_bkg, m4l_num, wt_num, title):
    bins = diag.M4L_BINS
    bw   = np.diff(bins)

    def dens(vals, wts):
        h, _ = np.histogram(vals, bins=bins, weights=wts)
        return h / (h.sum() * bw) if h.sum() > 0 else h.astype(float)

    d_d = dens(m4l_bkg, wt_bkg)
    n_d = dens(m4l_num, wt_num)
    r_d = dens(m4l_bkg, wt_bkg * r_bkg.numpy())

    kw = dict(where='post')
    ax_top.step(bins[:-1], d_d, label='Denominator ($D$)',      color='steelblue', **kw)
    ax_top.step(bins[:-1], n_d, label='Numerator ($N$) truth',  color='tab:orange',**kw)
    ax_top.step(bins[:-1], r_d, label='$r(x)\\times B$',        color='green', ls='--', **kw)
    ax_top.set_ylabel('Probability / bin')
    ax_top.legend(fontsize=7)
    ax_top.set_title(title)
    plt.setp(ax_top.get_xticklabels(), visible=False)

    ratio = np.where(n_d > 0, r_d / n_d, np.nan)
    ax_bot.step(bins[:-1], ratio, where='post', color='green')
    ax_bot.axhline(1.0, color='gray', ls='--', lw=0.8)
    ax_bot.set_ylim(0.5, 1.5)
    ax_bot.set_xlabel('$m_{4\\ell}$ [GeV]')
    ax_bot.set_ylabel('Ratio / truth')


def save_reweighting(m4l_bkg, wt_bkg,
                     r_sig_bkg, m4l_sig, wt_sig,
                     r_sbi_bkg, m4l_sbi, wt_sbi,
                     out: Path) -> None:
    fig = plt.figure(figsize=(13, 7))
    gs  = GridSpec(2, 2, figure=fig, height_ratios=[3, 1], hspace=0.05, wspace=0.3)
    _rw_panel(fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]),
              m4l_bkg, wt_bkg, r_sig_bkg, m4l_sig, wt_sig, 'Reweighting diagnostic: S/B')
    _rw_panel(fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1]),
              m4l_bkg, wt_bkg, r_sbi_bkg, m4l_sbi, wt_sbi, 'Reweighting diagnostic: SBI/B')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()


def save_calibration(s_bkg_sig, s_sig, wt_bkg, wt_sig,
                     s_bkg_sbi, s_sbi, wt_sbi, out: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    diag.plot_calib_curve(ax1, s_bkg_sig, s_sig,
                          torch.tensor(wt_bkg), torch.tensor(wt_sig), 'S/B')
    diag.plot_calib_curve(ax2, s_bkg_sbi, s_sbi,
                          torch.tensor(wt_bkg), torch.tensor(wt_sbi), 'SBI/B')
    plt.suptitle('Calibration: NSBI $\\hat{s}(x)$ vs MC $s(x)$', fontsize=12)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()


def save_nll(r_sbi, r_sig, n_obs, n_files, xs, out: Path) -> dict:
    xs_sig = float(np.prod(xs['sig'])); xs_bkg = float(np.prod(xs['bkg']))
    xs_int = float(np.prod(xs['int'])); xs_sbi = float(np.prod(xs['sbi']))

    mu_space    = torch.linspace(0.0, 4.0, 801)
    t_rate      = meas.compute_t_rate(n_obs.sum(), n_files, mu_space, xs_sig, xs_bkg, xs_int)
    t_shape     = meas.compute_t_shape(r_sig, r_sbi, n_obs, mu_space,
                                       xs_sig, xs_sbi, xs_bkg, xs_int)
    t_total     = t_rate + t_shape

    fig, ax = plt.subplots(figsize=(7, 5))
    meas.plot_nll(ax, mu_space, t_rate, t_shape, t_total, 'EveNet (ablated head)')
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()

    t_norm = (t_total - t_total.min()).numpy()
    mu_arr = mu_space.numpy()
    mu_hat = float(mu_arr[t_total.argmin()])
    in_1s  = mu_arr[t_norm < 1.0]
    in_2s  = mu_arr[t_norm < 4.0]

    return {
        'mu_hat':     round(mu_hat, 4),
        '1sig_lo':    round(float(in_1s[0]),  4) if len(in_1s) else float('nan'),
        '1sig_hi':    round(float(in_1s[-1]), 4) if len(in_1s) else float('nan'),
        '1sig_width': round(float(in_1s[-1]-in_1s[0]), 4) if len(in_1s) else float('nan'),
        '2sig_lo':    round(float(in_2s[0]),  4) if len(in_2s) else float('nan'),
        '2sig_hi':    round(float(in_2s[-1]), 4) if len(in_2s) else float('nan'),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Per-ablation runner
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_ablation(name: str, cls_overrides: dict,
                 bkg_calib_df, bkg_val_df, sbi_val_df, sig_val_df,
                 obs_df, xs: dict, n_files: int,
                 skip_training: bool = False) -> dict:

    print(f'\n{"="*62}')
    print(f'  Ablation: {name}  |  overrides: {cls_overrides or "(none)"}')
    print(f'{"="*62}')

    plots_dir    = ABLATION_PLOTS_DIR / name
    plots_dir.mkdir(parents=True, exist_ok=True)
    sbi_ckpt_dir = ABLATION_CKPT_DIR / name / 'sbi'
    sig_ckpt_dir = ABLATION_CKPT_DIR / name / 'sig'

    val_loss_sbi = val_loss_sig = float('nan')

    # ── Training ──────────────────────────────────────────────────────────────
    if not skip_training:
        print('  Loading training data ...')
        train_sbi, _ = load_training_pair(HOME / 'ggzz4l_bkg_slim.csv',
                                          HOME / 'ggzz4l_sbi_slim.csv')
        train_sig, _ = load_training_pair(HOME / 'ggzz4l_bkg_slim.csv',
                                          HOME / 'ggzz4l_sig_slim.csv')

        # val data for training comes from the diagnostic val sets
        ev_bkg_v = diag.df_to_evenet(bkg_val_df)
        ev_sbi_v = diag.df_to_evenet(sbi_val_df)
        ev_sig_v = diag.df_to_evenet(sig_val_df)
        wt_bkg_v = torch.tensor(bkg_val_df['wt'].to_numpy(), dtype=torch.float32)
        wt_sbi_v = torch.tensor(sbi_val_df['wt'].to_numpy(), dtype=torch.float32)
        wt_sig_v = torch.tensor(sig_val_df['wt'].to_numpy(), dtype=torch.float32)

        val_sbi_data = ({k: torch.cat([ev_bkg_v[k], ev_sbi_v[k]]) for k in ev_bkg_v},
                        torch.cat([torch.zeros(len(bkg_val_df), dtype=torch.long),
                                   torch.ones (len(sbi_val_df), dtype=torch.long)]),
                        torch.cat([wt_bkg_v, wt_sbi_v]))
        val_sig_data = ({k: torch.cat([ev_bkg_v[k], ev_sig_v[k]]) for k in ev_bkg_v},
                        torch.cat([torch.zeros(len(bkg_val_df), dtype=torch.long),
                                   torch.ones (len(sig_val_df), dtype=torch.long)]),
                        torch.cat([wt_bkg_v, wt_sig_v]))

        print(f'\n  Training SBI head ({SBI_EPOCHS} max epochs, patience={PATIENCE}) ...')
        clf_sbi  = build_ablation_clf(cls_overrides)
        norm_sbi = load_backbone_weights(clf_sbi, EVENET_SBI_CKPT)
        freeze_backbone(clf_sbi)
        val_loss_sbi = train_head(clf_sbi, train_sbi, val_sbi_data,
                                  norm_sbi, SBI_EPOCHS, sbi_ckpt_dir)
        print(f'    best val_loss_sbi = {val_loss_sbi:.5f}')
        del train_sbi; gc.collect()

        print(f'\n  Training SIG head ({SIG_EPOCHS} max epochs, patience={PATIENCE}) ...')
        clf_sig  = build_ablation_clf(cls_overrides)
        norm_sig = load_backbone_weights(clf_sig, EVENET_SIG_CKPT)
        freeze_backbone(clf_sig)
        val_loss_sig = train_head(clf_sig, train_sig, val_sig_data,
                                  norm_sig, SIG_EPOCHS, sig_ckpt_dir)
        print(f'    best val_loss_sig = {val_loss_sig:.5f}')
        del train_sig, ev_bkg_v, ev_sbi_v, ev_sig_v; gc.collect()

    else:
        print('  Loading saved checkpoints ...')
        sbi_f = sorted(sbi_ckpt_dir.glob('best*.pt'))
        sig_f = sorted(sig_ckpt_dir.glob('best*.pt'))
        if not sbi_f or not sig_f:
            raise FileNotFoundError(
                f'No checkpoints in {sbi_ckpt_dir} / {sig_ckpt_dir}. '
                'Run without --skip-training first.')
        clf_sbi = build_ablation_clf(cls_overrides)
        clf_sig = build_ablation_clf(cls_overrides)
        clf_sbi.load_checkpoint(str(sbi_f[-1]),
                                feature_names=copy.deepcopy(EVENET_FEATURE_NAMES))
        clf_sig.load_checkpoint(str(sig_f[-1]),
                                feature_names=copy.deepcopy(EVENET_FEATURE_NAMES))

    # ── Score val sets for diagnostics ────────────────────────────────────────
    print('  Scoring validation sets ...')
    ev_bkg = diag.df_to_evenet(bkg_val_df)
    ev_sbi = diag.df_to_evenet(sbi_val_df)
    ev_sig = diag.df_to_evenet(sig_val_df)

    s_bkg_sbi = score_silent(clf_sbi, ev_bkg)
    s_sbi_sbi = score_silent(clf_sbi, ev_sbi)
    s_bkg_sig = score_silent(clf_sig, ev_bkg)
    s_sig_sig = score_silent(clf_sig, ev_sig)

    # Calibrate on bkg_calib
    wt_calib = torch.tensor(bkg_calib_df['wt'].to_numpy(), dtype=torch.float32)
    ev_calib  = diag.df_to_evenet(bkg_calib_df)
    C_sbi = compute_C(score_silent(clf_sbi, ev_calib), wt_calib)
    C_sig = compute_C(score_silent(clf_sig, ev_calib), wt_calib)
    del ev_calib; gc.collect()
    print(f'    C_sbi={C_sbi:.4f}  C_sig={C_sig:.4f}')

    # ── Reweighting diagnostic ────────────────────────────────────────────────
    m4l_bkg = diag.compute_m4l(bkg_val_df)
    m4l_sbi = diag.compute_m4l(sbi_val_df)
    m4l_sig = diag.compute_m4l(sig_val_df)
    wt_bkg  = bkg_val_df['wt'].to_numpy()
    wt_sbi  = sbi_val_df['wt'].to_numpy()
    wt_sig  = sig_val_df['wt'].to_numpy()

    r_bkg_sbi = s_bkg_sbi / (1 - s_bkg_sbi) * C_sbi
    r_bkg_sig = s_bkg_sig / (1 - s_bkg_sig) * C_sig

    save_reweighting(m4l_bkg, wt_bkg,
                     r_bkg_sig, m4l_sig, wt_sig,
                     r_bkg_sbi, m4l_sbi, wt_sbi,
                     out=plots_dir / 'reweighting.png')
    print(f'  → reweighting.png')

    # ── Calibration curves ────────────────────────────────────────────────────
    save_calibration(s_bkg_sig, s_sig_sig, wt_bkg, wt_sig,
                     s_bkg_sbi, s_sbi_sbi, wt_sbi,
                     out=plots_dir / 'calibration.png')
    print(f'  → calibration.png')
    del ev_bkg, ev_sbi, ev_sig; gc.collect()

    # ── Score observed + NLL ──────────────────────────────────────────────────
    print('  Scoring observed data ...')
    ev_obs    = diag.df_to_evenet(obs_df)
    s_obs_sbi = score_silent(clf_sbi, ev_obs)
    s_obs_sig = score_silent(clf_sig, ev_obs)
    r_obs_sbi = s_obs_sbi / (1 - s_obs_sbi) * C_sbi
    r_obs_sig = s_obs_sig / (1 - s_obs_sig) * C_sig
    n_obs     = torch.tensor(obs_df['n'].to_numpy(), dtype=torch.float32)
    del ev_obs, clf_sbi, clf_sig; gc.collect()

    result = save_nll(r_obs_sbi, r_obs_sig, n_obs, n_files, xs,
                      out=plots_dir / 'nll.png')
    print(f'  → nll.png')
    print(f'  μ̂={result["mu_hat"]:.3f}  '
          f'1σ=[{result["1sig_lo"]:.3f}, {result["1sig_hi"]:.3f}]  '
          f'width={result["1sig_width"]:.3f}')

    result.update({'val_loss_sbi': round(val_loss_sbi, 6),
                   'val_loss_sig': round(val_loss_sig, 6),
                   'C_sbi': round(C_sbi, 6), 'C_sig': round(C_sig, 6),
                   'cls_overrides': cls_overrides})
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only',          type=str,  default=None)
    parser.add_argument('--skip-training', action='store_true')
    parser.add_argument('--resume',        action='store_true',
                        help='Skip ablations already saved in results JSON')
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING,
                        format='%(asctime)s | %(levelname)s | %(message)s')

    print('Loading shared data ...')
    rng = np.random.default_rng(SEED)

    # bkg_calib: 80% of first N_CALIB rows (matches score_observed.py split)
    bkg_all   = pd.read_csv(HOME / 'ggzz4l_bkg_slim.csv', nrows=N_CALIB)
    idx       = rng.permutation(len(bkg_all))
    bkg_calib = bkg_all.iloc[idx[int(len(bkg_all)*VAL_FRAC):]].reset_index(drop=True)

    # val sets for diagnostics (capped at N_VAL_DIAG)
    def load_val(name):
        df = pd.read_csv(HOME / f'ggzz4l_{name}_slim.csv', nrows=N_VAL_DIAG)
        return df.reset_index(drop=True)

    bkg_val_df = load_val('bkg')
    sbi_val_df = load_val('sbi')
    sig_val_df = load_val('sig')
    print(f'  calib {len(bkg_calib):,} | bkg_val {len(bkg_val_df):,} | '
          f'sbi_val {len(sbi_val_df):,} | sig_val {len(sig_val_df):,}')

    obs_df  = pd.concat([pd.read_csv(p) for p in OBS_CSVS], ignore_index=True)
    n_files = len(OBS_CSVS)
    print(f'  observed {len(obs_df):,} events from {n_files} files')

    with open(XS_JSON) as f:
        xs = json.load(f)

    ABLATION_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    results = json.loads(RESULTS_JSON.read_text()) if RESULTS_JSON.exists() else {}
    to_run  = ABLATIONS if not args.only else [
        a for a in ABLATIONS if a['name'] == args.only]
    if args.only and not to_run:
        print(f'Unknown: "{args.only}". Options: {[a["name"] for a in ABLATIONS]}')
        return

    for abl in to_run:
        name = abl['name']
        if args.resume and name in results:
            print(f'  Skipping {name} (already done)')
            continue
        try:
            result = run_ablation(
                name=name, cls_overrides=dict(abl['cls']),
                bkg_calib_df=bkg_calib,
                bkg_val_df=bkg_val_df, sbi_val_df=sbi_val_df, sig_val_df=sig_val_df,
                obs_df=obs_df, xs=xs, n_files=n_files,
                skip_training=args.skip_training,
            )
            results[name] = result
        except Exception as e:
            import traceback; traceback.print_exc()
            results[name] = {'error': str(e)}
        RESULTS_JSON.write_text(json.dumps(results, indent=2))
        print(f'  Results → {RESULTS_JSON}')

    # Summary table
    print('\n' + '='*70)
    print(f'{"Ablation":<22} {"μ̂":>6} {"1σ lo":>7} {"1σ hi":>7}'
          f' {"width":>6} {"vl_sbi":>8} {"vl_sig":>8}')
    print('-'*70)
    for name, r in results.items():
        if 'error' in r:
            print(f'{name:<22}  ERROR: {r["error"][:40]}')
        else:
            print(f'{name:<22} {r.get("mu_hat",float("nan")):6.3f}'
                  f' {r.get("1sig_lo",float("nan")):7.3f}'
                  f' {r.get("1sig_hi",float("nan")):7.3f}'
                  f' {r.get("1sig_width",float("nan")):6.3f}'
                  f' {r.get("val_loss_sbi",float("nan")):8.5f}'
                  f' {r.get("val_loss_sig",float("nan")):8.5f}')
    print('='*70)
    print(f'Plots  → {ABLATION_PLOTS_DIR}')
    print(f'Results → {RESULTS_JSON}')


if __name__ == '__main__':
    main()
