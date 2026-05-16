"""End-to-end NSBI pipeline script.

Steps:
  1. Train S/B and SBI/B classifiers  (or load from checkpoints)
  2. Compute calibration factors on training denominator events
  3. Generate (or load) an 'observed' dataset at a known mu_true
  4. Scan mu and report the best-fit + 1-sigma interval

Usage:
    python -m nsbi.run_nsbi \\
        --data_dir  "C:/Users/Ling Gie/Documents/Research/NSBI/zz4l/spliced_min_needed" \\
        --out_dir   ./nsbi_checkpoints \\
        --mu_true   2.0 \\
        --epochs    20 \\
        --device    cpu

To skip training and only do inference (if checkpoints already exist):
    python -m nsbi.run_nsbi ... --skip_training
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from evenet_lite import EvenetLiteClassifier

from .data import (
    FEATURE_NAMES,
    NORMALIZATION_RULES,
    load_process,
    split_process,
)
from .measure import (
    calibration_factor, scan_mu, _cross_sections,
    plot_reweighting_diagnostic, plot_calibration_curve,
)
from .train import train_sig_over_bkg, train_sbi_over_bkg, SEQ_DIM, GLOBAL_DIM


def _load_clf(checkpoint_path: Path, device: str = "auto") -> EvenetLiteClassifier:
    clf = EvenetLiteClassifier(
        class_labels=["bkg", "sig"],
        device=device,
        sequential_input_dim=SEQ_DIM,
        global_input_dim=GLOBAL_DIM,
        pretrained=False,
    )
    clf.load_checkpoint(str(checkpoint_path), feature_names=FEATURE_NAMES)
    return clf


def _make_obs_process(proc_sbi, mu_true: float):
    """Create a 'pseudo-observed' dataset from SBI events scaled to mu_true.

    In a real experiment this would be actual data. Here we simulate it by
    reweighting the SBI sample with the signal strength formula:
        w_obs = w_sbi * (mu * msq_sig + sqrt(mu) * msq_int + msq_bkg) / msq_sbi
    """
    import numpy as np
    import pandas as pd
    from .data import Process

    w = proc_sbi.weights.values
    c = proc_sbi.components
    mu = mu_true

    msq_sig = c["msq_sig_sm"].values
    msq_int = c["msq_int_sm"].values
    msq_bkg = c["msq_bkg_sm"].values
    msq_sbi = c["msq_sbi_sm"].values

    scaling = (mu * msq_sig + np.sqrt(mu) * msq_int + msq_bkg) / msq_sbi
    w_obs = w * scaling

    return Process(
        kinematics=proc_sbi.kinematics,
        components=proc_sbi.components,
        weights=pd.Series(w_obs),
    )


def plot_results(results: dict, out_path: Path, mu_true: float | None = None) -> None:
    """Plot the NLL profile curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, key, title in zip(
        axes,
        ["t_shape", "t_rate", "t_total"],
        ["Shape term", "Rate term", "Combined (Rate + Shape)"],
    ):
        mu   = results["mu"].numpy()
        t    = results[key].numpy()
        ax.plot(mu, t, lw=2)
        ax.axhline(1.0, color="gray", linestyle="--", label="1 sigma")
        ax.axhline(4.0, color="gray", linestyle=":",  label="2 sigma")
        ax.scatter(results["mu_hat"], 0.0, color="red", zorder=5,
                   label=f"mu_hat = {results['mu_hat']:.3f}")
        if mu_true is not None:
            ax.axvline(mu_true, color="green", linestyle="--", alpha=0.7,
                       label=f"mu_true = {mu_true}")
        ax.set_xlabel("Signal strength $\\mu$")
        ax.set_ylabel("$-2 \\log \\lambda$")
        ax.set_title(title)
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 12)
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    logging.info("NLL plot saved to %s", out_path)


def run(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # 1. Train (or load) classifiers
    # -----------------------------------------------------------------
    sig_ckpt = out_dir / "sig_over_bkg" / "final.pt"
    sbi_ckpt = out_dir / "sbi_over_bkg" / "final.pt"

    if args.skip_training and sig_ckpt.exists() and sbi_ckpt.exists():
        logging.info("Loading existing checkpoints...")
        clf_sig = _load_clf(sig_ckpt, device=args.device)
        clf_sbi = _load_clf(sbi_ckpt, device=args.device)
    else:
        logging.info("Training S/B classifier...")
        clf_sig = train_sig_over_bkg(
            data_dir, out_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            train_frac=args.train_frac,
            seed=args.seed,
            device=args.device,
        )
        logging.info("Training SBI/B classifier...")
        clf_sbi = train_sbi_over_bkg(
            data_dir, out_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            train_frac=args.train_frac,
            seed=args.seed,
            device=args.device,
        )

    # -----------------------------------------------------------------
    # 2. Calibration (over training background events)
    # -----------------------------------------------------------------
    logging.info("Computing calibration factors...")
    proc_bkg = load_process(data_dir / "ggZZ_bkg_spliced.csv")
    proc_sbi = load_process(data_dir / "ggZZ_sbi_spliced.csv")

    bkg_train, _ = split_process(proc_bkg, train_frac=args.train_frac, seed=args.seed)
    C_sig = calibration_factor(clf_sig, bkg_train)
    C_sbi = calibration_factor(clf_sbi, bkg_train)
    logging.info("C(S/B) = %.6f", C_sig)
    logging.info("C(SBI/B) = %.6f", C_sbi)

    # -----------------------------------------------------------------
    # 2b. Diagnostic plots (on validation data)
    # -----------------------------------------------------------------
    _, bkg_val = split_process(proc_bkg, train_frac=args.train_frac, seed=args.seed)
    proc_sig   = load_process(data_dir / "ggZZ_sig_spliced.csv")
    _, sig_val = split_process(proc_sig, train_frac=args.train_frac, seed=args.seed)
    _, sbi_val = split_process(proc_sbi, train_frac=args.train_frac, seed=args.seed)

    logging.info("Saving diagnostic plots...")

    plot_reweighting_diagnostic(
        clf_sig, sig_val, bkg_val,
        out_path=str(out_dir / "diag_reweight_sig_over_bkg.png"),
        title="S/B",
    )
    plot_reweighting_diagnostic(
        clf_sbi, sbi_val, bkg_val,
        out_path=str(out_dir / "diag_reweight_sbi_over_bkg.png"),
        title="SBI/B",
    )
    plot_calibration_curve(
        clf_sig, sig_val, bkg_val, C_sig,
        out_path=str(out_dir / "diag_calibration_sig_over_bkg.png"),
        title="S/B",
    )
    plot_calibration_curve(
        clf_sbi, sbi_val, bkg_val, C_sbi,
        out_path=str(out_dir / "diag_calibration_sbi_over_bkg.png"),
        title="SBI/B",
    )

    # -----------------------------------------------------------------
    # 3. Generate pseudo-observed dataset
    # -----------------------------------------------------------------
    logging.info("Generating pseudo-observed data at mu_true = %.2f ...", args.mu_true)
    obs_process = _make_obs_process(proc_sbi, mu_true=args.mu_true)

    # -----------------------------------------------------------------
    # 4. Scan mu
    # -----------------------------------------------------------------
    logging.info("Scanning mu ...")
    results = scan_mu(
        clf_sig=clf_sig,
        clf_sbi=clf_sbi,
        obs_process=obs_process,
        proc_sbi_for_xs=proc_sbi,
        C_sig=C_sig,
        C_sbi=C_sbi,
        mu_range=(0.0, 4.0),
        mu_steps=401,
        lumi=args.lumi,
    )

    mu_lo, mu_hi = results["mu_1sigma"]
    logging.info("mu_true       = %.3f", args.mu_true)
    logging.info("mu_hat        = %.3f", results["mu_hat"])
    logging.info("1-sigma range = [%.3f, %.3f]", mu_lo, mu_hi)

    # save numerical results
    torch.save(results, str(out_dir / "mu_scan_results.pt"))

    # plot
    plot_results(results, out_dir / "nll_profile.png", mu_true=args.mu_true)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="End-to-end NSBI pipeline with EveNet-Lite")
    p.add_argument("--data_dir",      required=True)
    p.add_argument("--out_dir",       default="./nsbi_checkpoints")
    p.add_argument("--mu_true",       type=float, default=2.0,  help="True signal strength for pseudo-data")
    p.add_argument("--epochs",        type=int,   default=20)
    p.add_argument("--batch_size",    type=int,   default=512)
    p.add_argument("--train_frac",    type=float, default=0.8)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--lumi",          type=float, default=300.0, help="Luminosity in fb^-1")
    p.add_argument("--device",        default="auto")
    p.add_argument("--skip_training", action="store_true", help="Load existing checkpoints instead of training")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run(_parse_args())
