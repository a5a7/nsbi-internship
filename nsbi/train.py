"""Train the two CARL density-ratio estimators using EvenetLiteClassifier.

Models trained:
  1. sig_over_bkg  : S/B  ratio  (numerator=sig, denominator=bkg)
  2. sbi_over_bkg  : SBI/B ratio (numerator=sbi, denominator=bkg reweighted to bkg)

Usage (from EveNet-Lite repo root):
    python -m nsbi.train --data_dir <path_to_csvs> --out_dir ./nsbi_checkpoints
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from evenet_lite import EvenetLiteClassifier

from .data import (
    FEATURE_NAMES,
    NORMALIZATION_RULES,
    load_process,
    make_classifier_data,
    split_process,
)


# ---------------------------------------------------------------------------
# Network dims for NSBI data
#   sequential_input_dim = 4  (pt, energy, eta, phi per lepton)
#   global_input_dim     = 3  (m4l, sum_pt, sum_energy)
# ---------------------------------------------------------------------------
SEQ_DIM    = 4
GLOBAL_DIM = 3


def build_classifier(device: str = "auto") -> EvenetLiteClassifier:
    """Build a fresh EvenetLiteClassifier configured for 4-lepton NSBI data."""
    return EvenetLiteClassifier(
        class_labels=["bkg", "sig"],
        device=device,
        # Per-group learning rates: Classification head -> ObjectEncoder -> PET+GlobalEmbedding
        lr=[1e-4, 5e-5, 1e-5],
        weight_decay=1e-2,
        module_lists=[
            ["Classification"],
            ["ObjectEncoder"],
            ["PET", "GlobalEmbedding"],
        ],
        grad_clip=1.0,
        warmup_epochs=1,
        warmup_ratio=0.1,
        warmup_start_factor=0.1,
        sequential_input_dim=SEQ_DIM,
        global_input_dim=GLOBAL_DIM,
        pretrained=True,     # soft-loads pretrained weights; input-projection layers
                             # (wrong shape) are skipped automatically, but the deep
                             # PET transformer backbone (16 attention layers, 256-dim)
                             # matches and loads — giving us a strong starting point.
        loss_gamma=0.0,      # standard binary cross-entropy
        log_level=logging.INFO,
    )


def train_sig_over_bkg(
    data_dir: str | Path,
    out_dir: str | Path,
    epochs: int = 20,
    batch_size: int = 512,
    train_frac: float = 0.8,
    seed: int = 42,
    device: str = "auto",
) -> EvenetLiteClassifier:
    """Train S/B density ratio estimator.

    Numerator : signal-only events (y=1)
    Denominator: background-only events (y=0)
    """
    data_dir = Path(data_dir)
    out_dir  = Path(out_dir) / "sig_over_bkg"
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading sig and bkg data...")
    proc_sig = load_process(data_dir / "ggZZ_sig_spliced.csv")
    proc_bkg = load_process(data_dir / "ggZZ_bkg_spliced.csv")

    sig_train, sig_val = split_process(proc_sig, train_frac=train_frac, seed=seed)
    bkg_train, bkg_val = split_process(proc_bkg, train_frac=train_frac, seed=seed)

    logging.info("Preparing training tensors for S/B...")
    train_features, train_labels, train_weights = make_classifier_data(sig_train, bkg_train)
    val_features,   val_labels,   val_weights   = make_classifier_data(sig_val,   bkg_val)

    clf = build_classifier(device=device)
    clf.fit(
        train_data=(train_features, train_labels, train_weights),
        val_data=(val_features, val_labels, val_weights),
        feature_names=FEATURE_NAMES,
        normalization_rules=NORMALIZATION_RULES,
        epochs=epochs,
        batch_size=batch_size,
        sampler="weighted",
        checkpoint_path=str(out_dir),
        save_top_k=1,
        monitor_metric="val_loss",
    )

    clf.save_checkpoint(str(out_dir / "final.pt"))
    logging.info("S/B model saved to %s", out_dir)
    return clf


def train_sbi_over_bkg(
    data_dir: str | Path,
    out_dir: str | Path,
    epochs: int = 20,
    batch_size: int = 512,
    train_frac: float = 0.8,
    seed: int = 42,
    device: str = "auto",
) -> EvenetLiteClassifier:
    """Train SBI/B density ratio estimator.

    Numerator  : SBI events (y=1)
    Denominator: SBI events reweighted to background hypothesis (y=0)
                 using w_i *= msq_bkg / msq_sbi  (same events, different label/weight)

    This is the same trick as the tutorial's --data.denominator_reweight '["sbi","bkg"]'.
    Using the same underlying events for both hypotheses helps training converge faster
    because the network sees the exact same x_i under both y=0 and y=1 labels.
    """
    data_dir = Path(data_dir)
    out_dir  = Path(out_dir) / "sbi_over_bkg"
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading sbi data...")
    proc_sbi = load_process(data_dir / "ggZZ_sbi_spliced.csv")

    sbi_train, sbi_val = split_process(proc_sbi, train_frac=train_frac, seed=seed)

    logging.info("Preparing training tensors for SBI/B...")
    # numerator=sbi, denominator=sbi reweighted to bkg
    train_features, train_labels, train_weights = make_classifier_data(
        sbi_train, sbi_train, denominator_reweight=("sbi", "bkg")
    )
    val_features, val_labels, val_weights = make_classifier_data(
        sbi_val, sbi_val, denominator_reweight=("sbi", "bkg")
    )

    clf = build_classifier(device=device)
    clf.fit(
        train_data=(train_features, train_labels, train_weights),
        val_data=(val_features, val_labels, val_weights),
        feature_names=FEATURE_NAMES,
        normalization_rules=NORMALIZATION_RULES,
        epochs=epochs,
        batch_size=batch_size,
        sampler="weighted",
        checkpoint_path=str(out_dir),
        save_top_k=1,
        monitor_metric="val_loss",
    )

    clf.save_checkpoint(str(out_dir / "final.pt"))
    logging.info("SBI/B model saved to %s", out_dir)
    return clf


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train NSBI CARL classifiers using EveNet-Lite")
    p.add_argument("--data_dir",   required=True, help="Directory containing ggZZ_*.csv files")
    p.add_argument("--out_dir",    default="./nsbi_checkpoints", help="Where to save checkpoints")
    p.add_argument("--model",      choices=["sig_over_bkg", "sbi_over_bkg", "both"], default="both")
    p.add_argument("--epochs",     type=int,   default=20)
    p.add_argument("--batch_size", type=int,   default=512)
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--device",     default="auto")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = _parse_args()

    if args.model in ("sig_over_bkg", "both"):
        train_sig_over_bkg(
            args.data_dir, args.out_dir,
            epochs=args.epochs, batch_size=args.batch_size,
            train_frac=args.train_frac, seed=args.seed, device=args.device,
        )

    if args.model in ("sbi_over_bkg", "both"):
        train_sbi_over_bkg(
            args.data_dir, args.out_dir,
            epochs=args.epochs, batch_size=args.batch_size,
            train_frac=args.train_frac, seed=args.seed, device=args.device,
        )
