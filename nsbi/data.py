"""Data loading and conversion utilities for NSBI with EveNet-Lite.

Converts 4-lepton CSV files (MCFM format) into the tensor format expected
by EvenetLiteClassifier: x [N, 4, F], globals [N, G], mask [N, 4].

Per-object feature ordering: [pt, energy, eta, phi]
  - Positions 2 & 3 (eta, phi) match default_network_config.yaml local_point_index=[2,3]
    which EveNet uses for spatial k-NN local embedding.

Global features derived from 4-lepton kinematics: [m4l, sum_pt, sum_energy]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------------
# Feature name constants (used for normalization_rules in fit())
# ---------------------------------------------------------------------------

LEPTON_FEATURES = ["pt", "energy", "eta", "phi"]   # per-object, order matters!
GLOBAL_FEATURES = ["m4l", "sum_pt", "sum_energy"]

NORMALIZATION_RULES = {
    "x": {
        "pt":     "log_normalize",
        "energy": "log_normalize",
        "eta":    "normalize",
        "phi":    "normalize_uniform",
    },
    "globals": {
        "m4l":        "log_normalize",
        "sum_pt":     "log_normalize",
        "sum_energy": "log_normalize",
    },
}

FEATURE_NAMES = {
    "x":       LEPTON_FEATURES,
    "globals": GLOBAL_FEATURES,
}

# CSV column groups
LEPTON_COLS = {
    "pt":     ["l1_pt",     "l2_pt",     "l3_pt",     "l4_pt"],
    "eta":    ["l1_eta",    "l2_eta",    "l3_eta",    "l4_eta"],
    "phi":    ["l1_phi",    "l2_phi",    "l3_phi",    "l4_phi"],
    "energy": ["l1_energy", "l2_energy", "l3_energy", "l4_energy"],
}


# ---------------------------------------------------------------------------
# Process dataclass (mirrors the tutorial's mcfm.Process)
# ---------------------------------------------------------------------------

@dataclass
class Process:
    """Holds kinematics, matrix-element components, and event weights."""
    kinematics: pd.DataFrame   # shape (N, 16)
    components: pd.DataFrame   # msq_sig_sm, msq_int_sm, msq_sbi_sm, msq_bkg_sm
    weights: pd.Series         # wt column

    def __len__(self) -> int:
        return len(self.kinematics)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_process(csv_path: str | Path) -> Process:
    """Load a MCFM CSV file into a Process object."""
    df = pd.read_csv(csv_path)
    kin_cols = [c for lep in zip(*LEPTON_COLS.values()) for c in lep]
    # just keep all 16 kinematic cols
    kin_cols = (
        LEPTON_COLS["pt"]
        + LEPTON_COLS["eta"]
        + LEPTON_COLS["phi"]
        + LEPTON_COLS["energy"]
    )
    kinematics = df[kin_cols]
    components = df[[c for c in df.columns if c.startswith("msq_")]]
    weights    = df["wt"]
    return Process(kinematics, components, weights)


def split_process(
    process: Process,
    train_frac: float = 0.8,
    seed: int = 42,
) -> Tuple[Process, Process]:
    """Split a Process into train/val subsets."""
    n = len(process)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    split = int(n * train_frac)
    train_idx = idx[:split]
    val_idx   = idx[split:]

    def _subset(proc: Process, idx: np.ndarray) -> Process:
        return Process(
            kinematics=proc.kinematics.iloc[idx].reset_index(drop=True),
            components=proc.components.iloc[idx].reset_index(drop=True),
            weights=proc.weights.iloc[idx].reset_index(drop=True),
        )

    return _subset(process, train_idx), _subset(process, val_idx)


# ---------------------------------------------------------------------------
# 4-vector helpers
# ---------------------------------------------------------------------------

def _m4l(kin: pd.DataFrame) -> np.ndarray:
    """Compute 4-lepton invariant mass from pt/eta/phi/energy columns."""
    E  = kin["l1_energy"].values + kin["l2_energy"].values + kin["l3_energy"].values + kin["l4_energy"].values
    px = (
        kin["l1_pt"].values * np.cos(kin["l1_phi"].values)
        + kin["l2_pt"].values * np.cos(kin["l2_phi"].values)
        + kin["l3_pt"].values * np.cos(kin["l3_phi"].values)
        + kin["l4_pt"].values * np.cos(kin["l4_phi"].values)
    )
    py = (
        kin["l1_pt"].values * np.sin(kin["l1_phi"].values)
        + kin["l2_pt"].values * np.sin(kin["l2_phi"].values)
        + kin["l3_pt"].values * np.sin(kin["l3_phi"].values)
        + kin["l4_pt"].values * np.sin(kin["l4_phi"].values)
    )
    # pz from pt and eta: pz = pt * sinh(eta)
    pz = (
        kin["l1_pt"].values * np.sinh(kin["l1_eta"].values)
        + kin["l2_pt"].values * np.sinh(kin["l2_eta"].values)
        + kin["l3_pt"].values * np.sinh(kin["l3_eta"].values)
        + kin["l4_pt"].values * np.sinh(kin["l4_eta"].values)
    )
    m2 = E**2 - px**2 - py**2 - pz**2
    return np.sqrt(np.maximum(m2, 0.0))


# ---------------------------------------------------------------------------
# Core conversion: Process -> EveNet tensors
# ---------------------------------------------------------------------------

def to_evenet_tensors(
    process: Process,
) -> Dict[str, torch.Tensor]:
    """Convert a Process into the feature dict expected by EvenetLiteClassifier.

    Returns
    -------
    features : dict with keys
        "x"       - [N, 4, 4]  per-lepton [pt, energy, eta, phi]
        "globals" - [N, 3]     event-level [m4l, sum_pt, sum_energy]
        "mask"    - [N, 4]     all True (no padding for 4-lepton events)
    """
    kin = process.kinematics
    N = len(kin)

    # --- per-object tensor: [N, 4, 4]  (4 leptons, 4 features each) ---
    # feature order: [pt, energy, eta, phi]  <- eta=2, phi=3 for local k-NN
    x = np.stack(
        [
            np.stack([kin[f"l{i}_pt"].values,
                      kin[f"l{i}_energy"].values,
                      kin[f"l{i}_eta"].values,
                      kin[f"l{i}_phi"].values], axis=1)   # (N, 4)
            for i in range(1, 5)
        ],
        axis=1,  # (N, 4_leptons, 4_features)
    )

    # --- global features: [N, 3] ---
    m4l      = _m4l(kin)
    sum_pt   = sum(kin[f"l{i}_pt"].values for i in range(1, 5))
    sum_e    = sum(kin[f"l{i}_energy"].values for i in range(1, 5))
    g = np.stack([m4l, sum_pt, sum_e], axis=1)  # (N, 3)

    # --- mask: all True (4 leptons always present) ---
    # Key must be "x_mask" to match EveNetLite.forward(x, x_mask, globals) signature.
    mask = np.ones((N, 4), dtype=bool)

    return {
        "x":       torch.tensor(x,    dtype=torch.float32),
        "globals": torch.tensor(g,    dtype=torch.float32),
        "x_mask":  torch.tensor(mask, dtype=torch.bool),
    }


# ---------------------------------------------------------------------------
# Prepare classifier training data
# ---------------------------------------------------------------------------

def make_classifier_data(
    numerator: Process,
    denominator: Process,
    denominator_reweight: Optional[Tuple[str, str]] = None,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Concatenate numerator (y=1) and denominator (y=0) into a balanced dataset.

    Parameters
    ----------
    numerator : Process
        Events from the numerator hypothesis (signal=1).
    denominator : Process
        Events from the denominator hypothesis (background=0).
    denominator_reweight : optional tuple ("from_component", "to_component")
        If provided, reweights denominator event weights via matrix-element ratio.
        Example: ("sbi", "bkg") reweights SBI events to background hypothesis.

    Returns
    -------
    features : dict  {"x": ..., "globals": ..., "mask": ...}
    labels   : [N_num + N_denom]  float tensor (1 or 0)
    weights  : [N_num + N_denom]  float tensor (physics weights, balanced)
    """
    feat_num  = to_evenet_tensors(numerator)
    feat_denom = to_evenet_tensors(denominator)

    w_num   = torch.tensor(numerator.weights.values, dtype=torch.float32)
    w_denom = torch.tensor(denominator.weights.values, dtype=torch.float32)

    # optional reweighting (SBI -> B via matrix element ratio)
    if denominator_reweight is not None:
        from_comp, to_comp = denominator_reweight
        msq_from = torch.tensor(
            denominator.components[f"msq_{from_comp}_sm"].values, dtype=torch.float32
        )
        msq_to = torch.tensor(
            denominator.components[f"msq_{to_comp}_sm"].values, dtype=torch.float32
        )
        w_denom = w_denom * (msq_to / msq_from)

    # balance: normalise each hypothesis to sum to 1
    w_num   = w_num   / w_num.sum()
    w_denom = w_denom / w_denom.sum()

    # concatenate
    features = {
        k: torch.cat([feat_num[k], feat_denom[k]], dim=0)
        for k in feat_num
    }
    labels  = torch.cat([torch.ones(len(numerator)),  torch.zeros(len(denominator))])
    weights = torch.cat([w_num, w_denom])

    return features, labels, weights
