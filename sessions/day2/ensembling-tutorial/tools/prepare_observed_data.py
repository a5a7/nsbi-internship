import os
import json
import argparse
import numpy as np
import pandas as pd

from physics.simulation import mcfm
from physics.hstar import sigstr

def main(args):

    events = mcfm.from_csv(file_path = args.sm_events, kinematics=args.kinematics)
    w_observed, _ = sigstr.scale(events, signal_strength = args.signal_strength)
    events.weights = pd.Series(w_observed, name=mcfm.csv_weight)
    mcfm.to_csv(events, file_path = 'observed.csv')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Merge MCFM event CSVs from multiple processes.")
    parser.add_argument('samples', type=str, help='List of process names to merge')

    parser.add_argument('--sm-events', required=True, default='analyzed.csv', help='Input events')
    parser.add_argument('--kinematics', required=False, default=['l1_pt', 'l1_eta', 'l1_phi', 'l1_energy', 'l2_pt', 'l2_eta', 'l2_phi', 'l2_energy', 'l3_pt', 'l3_eta', 'l3_phi', 'l3_energy', 'l4_pt', 'l4_eta', 'l4_phi', 'l4_energy'], help='Input events')
    parser.add_argument('--signal-strength', required=False, default=0.5, help='Signal strength')
    parser.add_argument('--output-events', required=True, default='analyzed.csv', help='Input events')

    args = parser.parse_args()

    main(args)
