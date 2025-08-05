import os
import json
import argparse
import numpy as np
import pandas as pd
from nsbi import carl

from physics.simulation import mcfm
from physics.hstar import sigstr
from physics.analysis import zz4l

def main(args):

    events = mcfm.from_csv(cross_section=args.xsec, file_path = args.sm_input)
    events = zz4l.analyze(events)

    w_obs, _ = sigstr.scale(events, signal_strength = float(args.signal_strength))
    n_obs = w_obs * args.lumi

    df = pd.concat([events.kinematics[args.kinematics], pd.Series(n_obs).to_frame(name='n')], axis=1)

    df.to_csv(args.bsm_output, index=False)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Merge MCFM event CSVs from multiple processes.")

    parser.add_argument('--sm-input', required=False, help='Input events')
    parser.add_argument('--xsec', required=False, type=float, default=5.659260726, help='Luminosity')
    parser.add_argument('--lumi', required=False, type=float, default=300., help='Luminosity')
    parser.add_argument('--kinematics', required=False, default=['l1_pt', 'l1_eta', 'l1_phi', 'l1_energy', 'l2_pt', 'l2_eta', 'l2_phi', 'l2_energy', 'l3_pt', 'l3_eta', 'l3_phi', 'l3_energy', 'l4_pt', 'l4_eta', 'l4_phi', 'l4_energy'], help='Input events')
    parser.add_argument('--signal-strength', required=False, default=1.0, help='Signal strength')
    parser.add_argument('--bsm-output', required=False, default='observed.csv', help='Input events')

    args = parser.parse_args()

    main(args)
