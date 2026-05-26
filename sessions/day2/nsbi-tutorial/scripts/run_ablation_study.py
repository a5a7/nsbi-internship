#!/usr/bin/env python3
"""
run_ablation_study.py
─────────────────────
Ablation study on the EveNet ClassificationHead only.

For each config:
  1. Build EveNetLite with modified Classification section
  2. Load pre-trained backbone (PET + GlobalEmbedding + ObjectEncoder) — frozen
  3. Fine-tune only the new Classification head on bkg+sbi and bkg+sig data
  4. Score observed data, calibrate, compute NLL measurement
  5. Append results to ~/ablation_results.json

Run (WSL, nsbi-venv activated):
  python3 /mnt/c/Users/Unnat/2025-lbnl/sessions/day2/nsbi-tutorial/scripts/run_ablation_study.py

  # Run a single ablation by name (skip others):
  python3 .../run_ablation_study.py --only layers_2

  # Skip to measurement (training already done):
  python3 .../run_ablation_study.py --only layers_2 --skip-training
"""

import argparse
import copy
import gc
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

# ── Paths ──────────────────────────────────────────────────────────────────────
HOME        = Path.home()
CHECKPOINTS = HOME / 'checkpoints'
NSBI_DIR    = Path('/mnt/c/Users/Unnat/2025-lbnl/sessions/day2/nsbi-tutorial')
EVENET_REPO = Path('/mnt/c/Users/Unnat/2025-lbnl/EveNet-Lite-main')
REPO_ROOT   = Path('/mnt/c/Users/Unnat/2025-lbnl')

OBS_CSVS = [
    REPO_ROOT / 'observed_0.csv',
    REPO_ROOT / 'observed_1.csv',
    REPO_ROOT / 'observed_2.csv',
    REPO_ROOT / 'observed_3.csv',
    REPO_ROOT / 'observed_4.csv',
]

EVENET_SBI_CKPT = CHECKPOINTS / 'EveNet_sbi_over_bkg-val_loss-epoch0020-0.6926.pt'
EVENET_SIG_CKPT = CHECKPOINTS / 'EveNet_sig_over_bkg-val_loss-epoch0007-0.4123.pt'
XS_JSON         = REPO_ROOT / 'ggzz4l_xs.json'

ABLATION_CKPT_DIR = HOME / 'ablation_checkpoints'
RESULTS_JSON      = HOME / 'ablation_results.json'

for p in (str(NSBI_DIR), str(EVENET_REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Constants ──────────────────────────────────────────────────────────────────
LEPTON_PREFIXES     = ['l1', 'l2', 'l3', 'l4']
EVENET_FEATURE_NAMES = {
    'x':       ['pt', 'energy', 'eta', 'phi'],
    'globals': ['m4l', 'sum_pt', 'sum_energy'],
}
NORM_RULES = {
    'x': {
        'pt':     'log_normalize',
        'energy': 'log_normalize',
        'eta':    'normalize',
        'phi':    'normalize_uniform',
    },
    'globals': {
        'm4l':        'log_normalize',
        'sum_pt':     'log_normalize',
        'sum_energy': 'log_normalize',
    },
}

N_CALIB    = 50_000   # bkg events for calibration C-factor
VAL_FRAC   = 0.2
SEED       = 42
MAX_EVENTS = 100_000  # per class for training (cap for memory)
BATCH_SIZE = 512
SBI_EPOCHS = 15
SIG_EPOCHS = 10
PATIENCE   = 5
LUMI       = 300.0    # ifb

# ── Ablation configs ───────────────────────────────────────────────────────────
# Each entry: name (str) + cls (dict of Classification overrides).
# Baseline (empty dict) retrains the same architecture as the original.

ABLATIONS = [
    # ── Baseline (no overrides) ────────────────────────────────────────────────
    {"name": "baseline",         "cls": {}},

    # ── F1: BranchLinear depth ─────────────────────────────────────────────────
    {"name": "layers_0",         "cls": {"num_classification_layers": 0}},
    {"name": "layers_2",         "cls": {"num_classification_layers": 2}},
    {"name": "layers_4",         "cls": {"num_classification_layers": 4}},
    {"name": "layers_8",         "cls": {"num_classification_layers": 8}},

    # ── F2: Head hidden dim ────────────────────────────────────────────────────
    {"name": "hidden_64",        "cls": {"hidden_dim": 64}},
    {"name": "hidden_128",       "cls": {"hidden_dim": 128}},
    {"name": "hidden_512",       "cls": {"hidden_dim": 512}},

    # ── F3: Cross-attention heads ──────────────────────────────────────────────
    {"name": "attn_heads_1",     "cls": {"num_attention_heads": 1}},
    {"name": "attn_heads_2",     "cls": {"num_attention_heads": 2}},
    {"name": "attn_heads_4",     "cls": {"num_attention_heads": 4}},
    {"name": "attn_heads_16",    "cls": {"num_attention_heads": 16}},

    # ── F5: Skip connection ────────────────────────────────────────────────────
    {"name": "no_skip",          "cls": {"skip_connection": False}},

    # ── F7: Dropout ────────────────────────────────────────────────────────────
    {"name": "dropout_0",        "cls": {"dropout": 0.0}},
    {"name": "dropout_02",       "cls": {"dropout": 0.2}},
    {"name": "dropout_05",       "cls": {"dropout": 0.5}},
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Model helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_ablation_clf(cls_overrides: dict) -> 'EvenetLiteClassifier':
    """Build EvenetLiteClassifier with modified Classification config."""
    from evenet_lite import EvenetLiteClassifier
    from evenet_lite.model import EveNetLite
    from evenet.control.global_config import DotDict

    cfg_path = EVENET_REPO / 'evenet_lite' / 'config' / 'default_network_config.yaml'
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    for key, val in cls_overrides.items():
        cfg['Classification'][key] = val

    model = EveNetLite(
        config=DotDict(cfg),
        global_input_dim=3,
        sequential_input_dim=4,
        cls_label=['bkg', 'num'],
    )

    # Use head-only LR groups; the backbone will be frozen
    clf = EvenetLiteClassifier(
        class_labels=['bkg', 'num'],
        device='auto',
        lr=[1e-3],
        module_lists=[['Classification']],
        weight_decay=1e-2,
        num_workers=0,
        global_input_dim=3,
        sequential_input_dim=4,
        model=model,
    )
    return clf


def load_backbone_weights(clf: 'EvenetLiteClassifier', ckpt_path: Path) -> dict:
    """
    Soft-load backbone weights from a pre-trained checkpoint into clf.
    Classification head keys are silently skipped if shapes differ.
    Returns normalizer state dict from the checkpoint.
    """
    raw = torch.load(str(ckpt_path), map_location='cpu')
    model_state = raw.get('model', raw)
    clf._soft_load_state_dict(model_state)
    return raw.get('normalizer', None)


def freeze_backbone(clf: 'EvenetLiteClassifier') -> None:
    """Freeze everything except the Classification head."""
    model = clf.model
    # EveNetLite with n_ensemble=1, ensemble_mode='independent'
    for single in model.models:
        for p in single.backbone.parameters():
            p.requires_grad_(False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'    Trainable params after freeze: {trainable:,}')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def df_to_evenet(df: pd.DataFrame) -> dict:
    """Convert lepton DataFrame to EveNet tensor dict."""
    pt  = np.stack([df[f'{l}_pt'].to_numpy()     for l in LEPTON_PREFIXES], 1).astype(np.float32)
    en  = np.stack([df[f'{l}_energy'].to_numpy() for l in LEPTON_PREFIXES], 1).astype(np.float32)
    eta = np.stack([df[f'{l}_eta'].to_numpy()    for l in LEPTON_PREFIXES], 1).astype(np.float32)
    phi = np.stack([df[f'{l}_phi'].to_numpy()    for l in LEPTON_PREFIXES], 1).astype(np.float32)
    x   = np.stack([pt, en, eta, phi], axis=2)   # (N, 4, 4)
    px  = pt * np.cos(phi); py = pt * np.sin(phi); pz = pt * np.sinh(eta)
    M2  = en.sum(1)**2 - px.sum(1)**2 - py.sum(1)**2 - pz.sum(1)**2
    m4l = np.sqrt(np.maximum(M2, 0)).astype(np.float32)
    N   = len(df)
    return {
        'x':       torch.from_numpy(x),
        'x_mask':  torch.ones((N, 4), dtype=torch.float32),
        'globals': torch.from_numpy(np.stack([m4l, pt.sum(1), en.sum(1)], axis=1)),
    }


def load_training_pair(csv_a: Path, csv_b: Path,
                       max_per_class: int = MAX_EVENTS,
                       val_frac: float = VAL_FRAC,
                       seed: int = SEED):
    """
    Build (train_data, val_data) from two CSVs.
    csv_a = bkg (label 0), csv_b = signal-type (label 1).
    Returns (train_data, val_data) each as (features_dict, labels, weights).
    """
    rng = np.random.default_rng(seed)

    def load_and_cap(csv, label, max_n):
        df = pd.read_csv(csv)
        if len(df) > max_n:
            idx = rng.choice(len(df), max_n, replace=False)
            df  = df.iloc[idx].reset_index(drop=True)
        feats  = df_to_evenet(df)
        labels = torch.full((len(df),), label, dtype=torch.long)
        weights = torch.tensor(df['wt'].to_numpy(), dtype=torch.float32)
        return feats, labels, weights

    feats_a, lbl_a, wt_a = load_and_cap(csv_a, 0, max_per_class)
    feats_b, lbl_b, wt_b = load_and_cap(csv_b, 1, max_per_class)

    N_a, N_b = len(lbl_a), len(lbl_b)
    feats = {k: torch.cat([feats_a[k], feats_b[k]], dim=0) for k in feats_a}
    lbls  = torch.cat([lbl_a, lbl_b])
    wts   = torch.cat([wt_a, wt_b])

    N     = N_a + N_b
    perm  = rng.permutation(N)
    n_val = int(N * val_frac)
    val_idx   = perm[:n_val]
    train_idx = perm[n_val:]

    def split(idx):
        f = {k: v[idx] for k, v in feats.items()}
        return f, lbls[idx], wts[idx]

    return split(train_idx), split(val_idx)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Training
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train_head(clf: 'EvenetLiteClassifier',
               train_data, val_data,
               normalizer_state: dict,
               epochs: int,
               ckpt_path: Path) -> float:
    """
    Fine-tune Classification head. Backbone must already be frozen.
    Returns best val_loss.
    """
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    clf.fit(
        train_data=train_data,
        val_data=val_data,
        feature_names=copy.deepcopy(EVENET_FEATURE_NAMES),
        normalization_rules=copy.deepcopy(NORM_RULES),
        normalization_stats=normalizer_state,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        checkpoint_path=str(ckpt_path),
        save_top_k=1,
        monitor_metric='val_loss',
        minimize_metric=True,
        early_stop_metric='val_loss',
        early_stop_patience=PATIENCE,
        early_stop_minimize=True,
    )

    # Load the saved best checkpoint to get val_loss
    saved = list(ckpt_path.parent.glob(f'{ckpt_path.stem}*.pt'))
    best_loss = float('nan')
    for f in saved:
        try:
            raw = torch.load(str(f), map_location='cpu')
            extra = raw.get('extra', {})
            v = extra.get('monitored_metric', None)
            if v is not None and (np.isnan(best_loss) or v < best_loss):
                best_loss = float(v)
        except Exception:
            pass

    return best_loss


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Scoring + calibration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def score_evenet_batch(clf: 'EvenetLiteClassifier',
                       feats: dict, batch_size: int = 512) -> torch.Tensor:
    """Return softmax prob of class-1, clamped."""
    N = len(feats['x'])
    all_logits = []
    logging.disable(logging.INFO)
    try:
        with torch.no_grad():
            for start in range(0, N, batch_size):
                b = {k: v[start:start+batch_size] for k, v in feats.items()}
                all_logits.append(clf.predict(b, batch_size=len(b['x'])))
    finally:
        logging.disable(logging.NOTSET)
    return torch.softmax(torch.cat(all_logits), dim=-1)[:, 1].clamp(1e-7, 1 - 1e-7).cpu()


def compute_C(s: torch.Tensor, wt: torch.Tensor) -> float:
    return (wt.sum() / (wt * s / (1 - s)).sum()).item()


def score_and_calibrate(clf_sbi: 'EvenetLiteClassifier',
                        clf_sig: 'EvenetLiteClassifier',
                        bkg_df: pd.DataFrame,
                        obs_df: pd.DataFrame):
    """
    Calibrate on bkg_df, score obs_df.
    Returns (r_sbi, r_sig, n_obs, C_sbi, C_sig).
    """
    wt_bkg = torch.tensor(bkg_df['wt'].to_numpy(), dtype=torch.float32)
    ev_bkg = df_to_evenet(bkg_df)
    ev_obs = df_to_evenet(obs_df)

    s_bkg_sbi = score_evenet_batch(clf_sbi, ev_bkg)
    s_bkg_sig = score_evenet_batch(clf_sig, ev_bkg)
    del ev_bkg; gc.collect()

    C_sbi = compute_C(s_bkg_sbi, wt_bkg)
    C_sig = compute_C(s_bkg_sig, wt_bkg)
    print(f'    C_sbi={C_sbi:.4f}  C_sig={C_sig:.4f}')

    s_obs_sbi = score_evenet_batch(clf_sbi, ev_obs)
    s_obs_sig = score_evenet_batch(clf_sig, ev_obs)
    del ev_obs; gc.collect()

    r_sbi = s_obs_sbi / (1 - s_obs_sbi) * C_sbi
    r_sig = s_obs_sig / (1 - s_obs_sig) * C_sig
    n_obs = torch.tensor(obs_df['n'].to_numpy(), dtype=torch.float32)

    return r_sbi, r_sig, n_obs, C_sbi, C_sig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NLL measurement (inline — no subprocess)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_t_shape(r_sig, r_sbi, n_obs, mu_space,
                    xs_sig, xs_sbi, xs_bkg, xs_int, chunk=20_000):
    sqrt_mu  = torch.sqrt(mu_space)
    mult_sig = mu_space - sqrt_mu
    mult_sbi = sqrt_mu
    mult_bkg = 1.0 - sqrt_mu
    denom    = xs_sig * mu_space + xs_int * sqrt_mu + xs_bkg

    t = torch.zeros(len(mu_space))
    N = len(n_obs)
    for start in range(0, N, chunk):
        rs = r_sig[start:start+chunk]
        rb = r_sbi[start:start+chunk]
        ni = n_obs[start:start+chunk]
        numer = (xs_sig * mult_sig[None, :] * rs[:, None]
               + xs_sbi * mult_sbi[None, :] * rb[:, None]
               + xs_bkg * mult_bkg[None, :])
        r_mu = (numer / denom[None, :]).clamp(min=1e-9)
        t   += -2.0 * (ni[:, None] * torch.log(r_mu)).sum(0)
    return t


def compute_t_rate(n_obs_total, n_files, mu_space, xs_sig, xs_bkg, xs_int):
    nu_mu = (xs_sig * mu_space + xs_int * torch.sqrt(mu_space) + xs_bkg) * LUMI
    return n_files * nu_mu - n_obs_total * torch.log(nu_mu)


def run_measurement(r_sbi, r_sig, n_obs, n_files, xs) -> dict:
    """Compute mu_hat, 1sigma interval. Returns result dict."""
    xs_sig = float(np.prod(xs['sig']))
    xs_bkg = float(np.prod(xs['bkg']))
    xs_int = float(np.prod(xs['int']))
    xs_sbi = float(np.prod(xs['sbi']))

    mu_space = torch.linspace(0.0, 4.0, 801)   # finer grid
    N_obs_total = n_obs.sum()

    t_rate  = compute_t_rate(N_obs_total, n_files, mu_space, xs_sig, xs_bkg, xs_int)
    t_shape = compute_t_shape(r_sig, r_sbi, n_obs, mu_space,
                              xs_sig, xs_sbi, xs_bkg, xs_int)
    t_total = t_rate + t_shape
    t_norm  = t_total - t_total.min()

    mu_hat    = mu_space[t_total.argmin()].item()
    mu_arr    = mu_space.numpy()
    t_arr     = t_norm.numpy()
    in_1sig   = mu_arr[t_arr < 1.0]
    in_2sig   = mu_arr[t_arr < 4.0]

    sig_lo = float(in_1sig[0])  if len(in_1sig) > 0 else float('nan')
    sig_hi = float(in_1sig[-1]) if len(in_1sig) > 0 else float('nan')
    sig2_lo = float(in_2sig[0])  if len(in_2sig) > 0 else float('nan')
    sig2_hi = float(in_2sig[-1]) if len(in_2sig) > 0 else float('nan')

    return {
        'mu_hat':    round(mu_hat, 4),
        '1sig_lo':   round(sig_lo, 4),
        '1sig_hi':   round(sig_hi, 4),
        '1sig_width': round(sig_hi - sig_lo, 4) if not np.isnan(sig_lo) else float('nan'),
        '2sig_lo':   round(sig2_lo, 4),
        '2sig_hi':   round(sig2_hi, 4),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Results I/O
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_results() -> dict:
    if RESULTS_JSON.exists():
        with open(RESULTS_JSON) as f:
            return json.load(f)
    return {}


def save_results(results: dict) -> None:
    with open(RESULTS_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'  Results saved → {RESULTS_JSON}')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main loop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_ablation(name: str, cls_overrides: dict,
                 bkg_df: pd.DataFrame, obs_df: pd.DataFrame,
                 xs: dict, n_files: int,
                 skip_training: bool = False) -> dict:
    print(f'\n{"="*60}')
    print(f'  Ablation: {name}')
    print(f'  Overrides: {cls_overrides or "(none — baseline)"}')
    print(f'{"="*60}')

    sbi_ckpt_dir = ABLATION_CKPT_DIR / name / 'sbi'
    sig_ckpt_dir = ABLATION_CKPT_DIR / name / 'sig'

    # ── Training ──────────────────────────────────────────────────────────────
    val_loss_sbi = float('nan')
    val_loss_sig = float('nan')

    if not skip_training:
        # Load training data
        print('  Loading training data ...')
        train_sbi, val_sbi = load_training_pair(
            HOME / 'ggzz4l_bkg_slim.csv',
            HOME / 'ggzz4l_sbi_slim.csv',
        )
        train_sig, val_sig = load_training_pair(
            HOME / 'ggzz4l_bkg_slim.csv',
            HOME / 'ggzz4l_sig_slim.csv',
        )
        print(f'    SBI: {len(train_sbi[1]):,} train / {len(val_sbi[1]):,} val')
        print(f'    SIG: {len(train_sig[1]):,} train / {len(val_sig[1]):,} val')

        # ── SBI model ─────────────────────────────────────────────────────────
        print(f'\n  Training SBI head ({SBI_EPOCHS} max epochs, patience={PATIENCE}) ...')
        clf_sbi = build_ablation_clf(cls_overrides)
        norm_sbi = load_backbone_weights(clf_sbi, EVENET_SBI_CKPT)
        freeze_backbone(clf_sbi)
        val_loss_sbi = train_head(clf_sbi, train_sbi, val_sbi, norm_sbi,
                                  SBI_EPOCHS, sbi_ckpt_dir / 'best')
        print(f'    Best val_loss_sbi = {val_loss_sbi:.5f}')
        del train_sbi, val_sbi; gc.collect()

        # ── SIG model ─────────────────────────────────────────────────────────
        print(f'\n  Training SIG head ({SIG_EPOCHS} max epochs, patience={PATIENCE}) ...')
        clf_sig = build_ablation_clf(cls_overrides)
        norm_sig = load_backbone_weights(clf_sig, EVENET_SIG_CKPT)
        freeze_backbone(clf_sig)
        val_loss_sig = train_head(clf_sig, train_sig, val_sig, norm_sig,
                                  SIG_EPOCHS, sig_ckpt_dir / 'best')
        print(f'    Best val_loss_sig = {val_loss_sig:.5f}')
        del train_sig, val_sig; gc.collect()

    else:
        # Load from saved checkpoints
        print('  Loading saved checkpoints (skipping training) ...')
        sbi_files = sorted(sbi_ckpt_dir.glob('best*.pt'))
        sig_files = sorted(sig_ckpt_dir.glob('best*.pt'))
        if not sbi_files or not sig_files:
            raise FileNotFoundError(
                f'No checkpoints found in {sbi_ckpt_dir} or {sig_ckpt_dir}. '
                'Run without --skip-training first.'
            )

        from evenet_lite import EvenetLiteClassifier
        clf_sbi = build_ablation_clf(cls_overrides)
        clf_sig = build_ablation_clf(cls_overrides)
        clf_sbi.load_checkpoint(str(sbi_files[-1]),
                                feature_names=copy.deepcopy(EVENET_FEATURE_NAMES))
        clf_sig.load_checkpoint(str(sig_files[-1]),
                                feature_names=copy.deepcopy(EVENET_FEATURE_NAMES))

    # ── Scoring + measurement ─────────────────────────────────────────────────
    print('\n  Scoring observed data ...')
    r_sbi, r_sig, n_obs, C_sbi, C_sig = score_and_calibrate(
        clf_sbi, clf_sig, bkg_df, obs_df
    )
    del clf_sbi, clf_sig; gc.collect()

    print('  Computing NLL measurement ...')
    result = run_measurement(r_sbi, r_sig, n_obs, n_files, xs)
    result['val_loss_sbi'] = round(val_loss_sbi, 6)
    result['val_loss_sig'] = round(val_loss_sig, 6)
    result['C_sbi']        = round(C_sbi, 6)
    result['C_sig']        = round(C_sig, 6)
    result['cls_overrides'] = cls_overrides

    print(f'  → μ̂={result["mu_hat"]:.3f}  '
          f'1σ=[{result["1sig_lo"]:.3f}, {result["1sig_hi"]:.3f}]  '
          f'width={result["1sig_width"]:.3f}')

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only',           type=str, default=None,
                        help='Run only this ablation name')
    parser.add_argument('--skip-training',  action='store_true',
                        help='Skip training; load existing checkpoints')
    parser.add_argument('--resume',         action='store_true',
                        help='Skip ablations already in results JSON')
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING,
                        format='%(asctime)s | %(levelname)s | %(message)s')

    # ── Shared data ────────────────────────────────────────────────────────────
    print('Loading shared data ...')
    bkg_all   = pd.read_csv(HOME / 'ggzz4l_bkg_slim.csv', nrows=N_CALIB)
    rng       = np.random.default_rng(SEED)
    idx       = rng.permutation(len(bkg_all))
    n_val     = int(len(bkg_all) * VAL_FRAC)
    bkg_calib = bkg_all.iloc[idx[n_val:]].reset_index(drop=True)   # same split as score_observed
    print(f'  Calibration bkg: {len(bkg_calib):,} events')

    obs_frames = []
    for p in OBS_CSVS:
        obs_frames.append(pd.read_csv(p))
    obs_df  = pd.concat(obs_frames, ignore_index=True)
    n_files = len(OBS_CSVS)
    print(f'  Observed: {len(obs_df):,} events from {n_files} files')

    with open(XS_JSON) as f:
        xs = json.load(f)

    # ── Ablation loop ──────────────────────────────────────────────────────────
    results = load_results()
    ablations = ABLATIONS if args.only is None else [
        a for a in ABLATIONS if a['name'] == args.only
    ]
    if args.only and not ablations:
        print(f'No ablation named "{args.only}". Available: '
              + ', '.join(a['name'] for a in ABLATIONS))
        return

    for abl in ablations:
        name = abl['name']
        if args.resume and name in results:
            print(f'  Skipping {name} (already in results)')
            continue

        try:
            result = run_ablation(
                name=name,
                cls_overrides=dict(abl['cls']),
                bkg_df=bkg_calib,
                obs_df=obs_df,
                xs=xs,
                n_files=n_files,
                skip_training=args.skip_training,
            )
            results[name] = result
            save_results(results)   # save after each ablation (fault-tolerant)
        except Exception as e:
            print(f'  ERROR in {name}: {e}')
            import traceback; traceback.print_exc()
            results[name] = {'error': str(e)}
            save_results(results)

    # ── Summary table ──────────────────────────────────────────────────────────
    print('\n\n' + '='*70)
    print(f'{"Ablation":<22} {"μ̂":>6} {"1σ lo":>7} {"1σ hi":>7} {"width":>6} {"vloss_sbi":>10} {"vloss_sig":>10}')
    print('-'*70)
    for name, r in results.items():
        if 'error' in r:
            print(f'{name:<22}  ERROR: {r["error"][:40]}')
            continue
        print(f'{name:<22} '
              f'{r.get("mu_hat", float("nan")):6.3f} '
              f'{r.get("1sig_lo", float("nan")):7.3f} '
              f'{r.get("1sig_hi", float("nan")):7.3f} '
              f'{r.get("1sig_width", float("nan")):6.3f} '
              f'{r.get("val_loss_sbi", float("nan")):10.5f} '
              f'{r.get("val_loss_sig", float("nan")):10.5f}')
    print('='*70)
    print(f'\nFull results: {RESULTS_JSON}')


if __name__ == '__main__':
    main()
