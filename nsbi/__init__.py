"""NSBI (Neural Simulation-Based Inference) module for EveNet-Lite.

Trains two CARL density-ratio estimators using EveNet as the classifier backbone:
  - S/B  : p(x | Signal) / p(x | Background)
  - SBI/B: p(x | SBI)    / p(x | Background)

Then uses them to perform a fully-differential likelihood-ratio measurement
of the Higgs signal strength mu.
"""

from .data import load_process, to_evenet_tensors, split_process
from .measure import calibration_factor, compute_r, neg_log_likelihood, scan_mu

__all__ = [
    "load_process",
    "to_evenet_tensors",
    "split_process",
    "calibration_factor",
    "compute_r",
    "neg_log_likelihood",
    "scan_mu",
]
