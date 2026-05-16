"""NSBI measurement utilities.

Implements:
  - calibration_factor : ensure <r>_B = 1
  - compute_r          : run classifier + likelihood ratio trick on a dataset
  - neg_log_likelihood : rate + shape NLL as a function of mu
  - scan_mu            : scan over mu values and return the NLL profile
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from evenet_lite import EvenetLiteClassifier

from .data import (
    FEATURE_NAMES,
    NORMALIZATION_RULES,
    Process,
    load_process,
    split_process,
    to_evenet_tensors,
)


# ---------------------------------------------------------------------------
# Density ratio from classifier output
# ---------------------------------------------------------------------------

def compute_r(
    clf: EvenetLiteClassifier,
    process: Process,
    batch_size: int = 1024,
    signal_class_idx: int = 1,
) -> torch.Tensor:
    """Run classifier on events and apply the likelihood ratio trick.

    r(x) = s(x) / (1 - s(x))

    where s(x) is the classifier's predicted *probability* for the signal class.

    Notes
    -----
    EvenetLiteClassifier.predict() returns raw **logits** (the model uses
    F.cross_entropy internally which applies softmax).  We apply softmax here
    to convert logits -> probabilities before the ratio trick.

    Parameters
    ----------
    clf : fitted EvenetLiteClassifier
    process : Process to evaluate
    batch_size : inference batch size
    signal_class_idx : index of the signal class in the output
                       (1 by default, since class_labels=["bkg", "sig"])

    Returns
    -------
    r : [N] tensor of per-event density ratios
    """
    features = to_evenet_tensors(process)
    logits = clf.predict(features, batch_size=batch_size)  # [N, 2] raw logits

    # convert logits -> probabilities
    probs = torch.softmax(logits.float(), dim=1)
    s = probs[:, signal_class_idx]                          # p(signal | x)
    r = s / (1.0 - s.clamp(max=1.0 - 1e-7))               # likelihood ratio
    return r


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibration_factor(
    clf: EvenetLiteClassifier,
    denominator: Process,
    batch_size: int = 1024,
) -> float:
    """Compute 1 / <r>_denominator.

    A well-trained DRE satisfies <r(x; N, D)>_D = 1 by construction.
    Any deviation can be corrected by multiplying r by this factor C = 1/<r>_D.

    Should be computed on *training* data (not test data) to avoid unblinding.

    Returns
    -------
    C : float calibration factor
    """
    r = compute_r(clf, denominator, batch_size=batch_size)

    # probability-weight the denominator events
    w = torch.tensor(denominator.weights.values, dtype=torch.float32)
    p = w / w.sum()

    expectation = (r * p).sum()
    C = 1.0 / expectation.item()
    return C


# ---------------------------------------------------------------------------
# Physics: signal strength scaling
# ---------------------------------------------------------------------------

def _cross_sections(proc_sbi: Process) -> Dict[str, float]:
    """Extract per-component cross sections from the SBI process."""
    w = proc_sbi.weights.values
    c = proc_sbi.components
    msq_sbi = c["msq_sbi_sm"].values

    xs = {}
    for comp in ("sig", "int", "bkg", "sbi"):
        xs[comp] = float((w * c[f"msq_{comp}_sm"].values / msq_sbi).sum())
    return xs


def _nu_mu(xs: Dict[str, float], mu_space: torch.Tensor, lumi: float) -> torch.Tensor:
    """Expected number of events as a function of mu.

    nu(mu) = mu * nu_S + sqrt(mu) * nu_I + nu_B
    """
    nu_sig = xs["sig"] * lumi
    nu_int = xs["int"] * lumi
    nu_bkg = xs["bkg"] * lumi
    return mu_space * nu_sig + torch.sqrt(mu_space) * nu_int + nu_bkg


# ---------------------------------------------------------------------------
# Likelihood
# ---------------------------------------------------------------------------

def neg_log_likelihood(
    r_sig_over_bkg: torch.Tensor,
    r_sbi_over_bkg: torch.Tensor,
    event_weights: torch.Tensor,
    xs: Dict[str, float],
    mu_space: torch.Tensor,
    lumi: float = 300.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute rate + shape NLL as a function of mu.

    Shape term
    ----------
    p_SBI(x|mu) / p_B(x) = [ (mu - sqrt(mu)) * xs_S * r_S(x)
                              + sqrt(mu) * xs_SBI * r_SBI(x)
                              + (1 - sqrt(mu)) * xs_B ]
                            / (mu * xs_S + sqrt(mu) * xs_I + xs_B)

    t_shape = -2 * sum_i  w_i * log( p_SBI(x_i|mu) / p_B(x_i) )

    Rate term (Poisson)
    -------------------
    t_rate = -2 * [ n * log(nu(mu)) - nu(mu) ]
    where n = sum of event weights (simulated dataset).

    Parameters
    ----------
    r_sig_over_bkg : [N]  density ratio r(x; S, B) per event
    r_sbi_over_bkg : [N]  density ratio r(x; SBI, B) per event
    event_weights  : [N]  per-event weights (wt column)
    xs             : dict of component cross sections from _cross_sections()
    mu_space       : [M]  values of mu to scan
    lumi           : integrated luminosity in fb^-1

    Returns
    -------
    t_total : [M]  total NLL profile
    t_rate  : [M]  Poisson rate term
    t_shape : [M]  per-event shape term
    """
    N = len(r_sig_over_bkg)
    M = len(mu_space)

    # broadcast [N] -> [N, M]
    r_sig = r_sig_over_bkg[:, None]
    r_sbi = r_sbi_over_bkg[:, None]

    sqrt_mu  = torch.sqrt(mu_space)[None, :]  # [1, M]
    mu       = mu_space[None, :]               # [1, M]

    # numerator of p_SBI(x|mu)/p_B(x)
    numerator = (
        xs["sig"] * (mu - sqrt_mu) * r_sig
        + xs["sbi"] * sqrt_mu * r_sbi
        + xs["bkg"] * (1.0 - sqrt_mu)
    )
    # denominator (expected xs at mu)
    denom = xs["sig"] * mu_space + xs["int"] * torch.sqrt(mu_space) + xs["bkg"]  # [M]

    ratio = numerator / denom[None, :]   # [N, M]
    ratio = ratio.clamp(min=1e-9)        # numerical safety

    # shape term: weighted sum of log-ratios
    w = event_weights[:, None].double()
    t_shape = -2.0 * (w * torch.log(ratio.double())).sum(dim=0)  # [M]

    # rate term: Poisson
    n_obs   = float(event_weights.sum())
    nu_mu   = _nu_mu(xs, mu_space, lumi).double()
    t_rate  = -2.0 * (n_obs * torch.log(nu_mu) - nu_mu)          # [M]

    t_total = t_rate + t_shape
    return t_total, t_rate, t_shape


# ---------------------------------------------------------------------------
# Mu scan
# ---------------------------------------------------------------------------

def scan_mu(
    clf_sig: EvenetLiteClassifier,
    clf_sbi: EvenetLiteClassifier,
    obs_process: Process,
    proc_sbi_for_xs: Process,
    C_sig: float = 1.0,
    C_sbi: float = 1.0,
    mu_range: Tuple[float, float] = (0.0, 4.0),
    mu_steps: int = 401,
    lumi: float = 300.0,
    batch_size: int = 1024,
) -> Dict[str, torch.Tensor]:
    """Full mu profile likelihood scan.

    Parameters
    ----------
    clf_sig      : fitted S/B classifier
    clf_sbi      : fitted SBI/B classifier
    obs_process  : 'observed' dataset (may be simulated at a true mu value)
    proc_sbi_for_xs : SBI process used to extract component cross sections
    C_sig, C_sbi : calibration factors (from calibration_factor())
    mu_range     : (mu_min, mu_max)
    mu_steps     : number of mu values to test
    lumi         : luminosity in fb^-1

    Returns
    -------
    dict with keys:
        "mu"       : [M] mu values scanned
        "t_total"  : [M] total NLL (zeroed at minimum)
        "t_rate"   : [M] rate-only NLL (zeroed at minimum)
        "t_shape"  : [M] shape-only NLL (zeroed at minimum)
        "mu_hat"   : best-fit mu
        "mu_1sigma": (mu_lo, mu_hi) 1-sigma interval
    """
    mu_space = torch.linspace(*mu_range, mu_steps)

    # run classifiers
    r_sig = compute_r(clf_sig, obs_process, batch_size=batch_size) * C_sig
    r_sbi = compute_r(clf_sbi, obs_process, batch_size=batch_size) * C_sbi

    # event weights
    w = torch.tensor(obs_process.weights.values, dtype=torch.float32)

    # cross sections
    xs = _cross_sections(proc_sbi_for_xs)

    t_total, t_rate, t_shape = neg_log_likelihood(r_sig, r_sbi, w, xs, mu_space, lumi)

    # zero at minimum
    t_total = t_total - t_total.min()
    t_rate  = t_rate  - t_rate.min()
    t_shape = t_shape - t_shape.min()

    mu_hat = mu_space[t_total.argmin()].item()

    # 1-sigma interval: where t_total < 1.0
    below_1sig = mu_space[t_total < 1.0]
    if len(below_1sig) > 0:
        mu_1sigma = (below_1sig[0].item(), below_1sig[-1].item())
    else:
        mu_1sigma = (float("nan"), float("nan"))

    return {
        "mu":        mu_space,
        "t_total":   t_total,
        "t_rate":    t_rate,
        "t_shape":   t_shape,
        "mu_hat":    mu_hat,
        "mu_1sigma": mu_1sigma,
    }


# ---------------------------------------------------------------------------
# Diagnostic 1: Reweighting check
# ---------------------------------------------------------------------------

def plot_reweighting_diagnostic(
    clf: EvenetLiteClassifier,
    numerator_val: Process,
    denominator_val: Process,
    out_path: str,
    title: str = "",
    batch_size: int = 1024,
) -> None:
    """Plot reweighting diagnostic: does r(x)*w_B reproduce the N distribution?

    Three histograms of m4l on the same axes:
      - Denominator (B) — original
      - Denominator reweighted to numerator via r(x)
      - True numerator (N) — ground truth to match against
    """
    import matplotlib.pyplot as plt
    from .data import _m4l

    r = compute_r(clf, denominator_val, batch_size=batch_size)

    w_denom = torch.tensor(denominator_val.weights.values, dtype=torch.float32)
    w_num   = torch.tensor(numerator_val.weights.values,   dtype=torch.float32)
    w_reweighted = w_denom * r

    # normalise to probability
    p_denom      = (w_denom      / w_denom.sum()).numpy()
    p_num        = (w_num        / w_num.sum()).numpy()
    p_reweighted = (w_reweighted / w_reweighted.sum()).numpy()

    m4l_denom = _m4l(denominator_val.kinematics)
    m4l_num   = _m4l(numerator_val.kinematics)

    bins = np.linspace(180, 1000, 42)

    fig, axes = plt.subplots(2, 1, figsize=(7, 6),
                             gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

    h_denom,      _ = np.histogram(m4l_denom, bins=bins, weights=p_denom)
    h_num,        _ = np.histogram(m4l_num,   bins=bins, weights=p_num)
    h_reweighted, _ = np.histogram(m4l_denom, bins=bins, weights=p_reweighted)

    centers = 0.5 * (bins[:-1] + bins[1:])

    axes[0].step(bins[:-1], h_denom,      where="post", label="Denominator (B)",       color="steelblue")
    axes[0].step(bins[:-1], h_num,        where="post", label="Numerator (N) truth",   color="tomato")
    axes[0].step(bins[:-1], h_reweighted, where="post", label=r"$r(x) \times$ B",      color="green", linestyle="--")
    axes[0].set_ylabel("Probability / bin")
    axes[0].set_title(f"Reweighting diagnostic{': ' + title if title else ''}")
    axes[0].legend()

    # ratio panel: reweighted / truth
    ratio = np.where(h_num > 0, h_reweighted / np.where(h_num > 0, h_num, 1), np.nan)
    axes[1].step(bins[:-1], ratio, where="post", color="green")
    axes[1].axhline(1.0, color="gray", linestyle="--")
    axes[1].set_ylim(0.5, 1.5)
    axes[1].set_ylabel("Ratio")
    axes[1].set_xlabel(r"$m_{4\ell}$ [GeV]")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Diagnostic 2: Calibration curve
# ---------------------------------------------------------------------------

def plot_calibration_curve(
    clf: EvenetLiteClassifier,
    numerator_val: Process,
    denominator_val: Process,
    C: float,
    out_path: str,
    title: str = "",
    batch_size: int = 1024,
    n_bins: int = 40,
) -> None:
    """Plot calibration curve: NSBI estimate s_hat vs MC estimate of s.

    A well-calibrated model sits on the diagonal. Deviations indicate
    per-event miscalibration that C (global scalar) cannot fix.
    """
    import matplotlib.pyplot as plt
    from .data import make_classifier_data

    # build balanced val dataset and run classifier on it
    features, labels, weights = make_classifier_data(numerator_val, denominator_val)
    logits = clf.predict(features, batch_size=batch_size)          # [N, 2] logits
    probs  = torch.softmax(logits.float(), dim=1)
    s_hat  = probs[:, 1].numpy()                                   # signal score

    # apply calibration: r -> r*C -> s = r*C / (1 + r*C)
    r_cal = (s_hat / np.clip(1.0 - s_hat, 1e-7, None)) * C
    s_cal = r_cal / (1.0 + r_cal)

    w  = weights.numpy()
    y  = labels.numpy()

    bins    = np.linspace(0, 1, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    widths  = 0.5 * (bins[1:] - bins[:-1])

    h_num   = np.zeros(n_bins)
    h_denom = np.zeros(n_bins)
    v_num   = np.zeros(n_bins)
    v_denom = np.zeros(n_bins)

    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        mask_sig = (s_cal >= lo) & (s_cal < hi) & (y == 1)
        mask_bkg = (s_cal >= lo) & (s_cal < hi) & (y == 0)
        h_num[i]   = w[mask_sig].sum()
        h_denom[i] = w[mask_bkg].sum()
        v_num[i]   = (w[mask_sig] ** 2).sum()
        v_denom[i] = (w[mask_bkg] ** 2).sum()

    total = h_num + h_denom
    s_mc  = np.where(total > 0, h_num / total, np.nan)
    err   = np.where(
        total > 0,
        np.sqrt((h_denom / total**2)**2 * v_num + (h_num / total**2)**2 * v_denom),
        np.nan,
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideal")
    ax.errorbar(centers, s_mc, xerr=widths, yerr=err,
                linestyle="none", marker="none", color="steelblue", label="MC estimate")
    ax.set_xlabel(r"NSBI estimate $\hat{s}(x)$ (calibrated)")
    ax.set_ylabel(r"MC estimate $s(x)$")
    ax.set_title(f"Calibration curve{': ' + title if title else ''}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
