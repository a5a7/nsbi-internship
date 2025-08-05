#!/usr/bin/env bash

SEED=$((RANDOM % 101))

mkdir -p sig_over_bkg && cd sig_over_bkg
python -m nsbi.carl fit \
    --data.features '["l1_pt", "l1_eta", "l1_phi", "l1_energy", "l2_pt", "l2_eta", "l2_phi", "l2_energy", "l3_pt", "l3_eta", "l3_phi", "l3_energy", "l4_pt", "l4_eta", "l4_phi", "l4_energy"]' \
    --data.numerator_events '/ptmp/mpp/taepa/higgs-offshell-interpretation/data/zz4l/ggZZ_sig/analyzed.csv' \
    --data.denominator_events '/ptmp/mpp/taepa/higgs-offshell-interpretation/data/zz4l/ggZZ_bkg/analyzed.csv' \
    --data.sample_size 10_000_000 \
    --data.batch_size 1024 \
    --model.learning_rate 1e-4 \
    --model.n_layers 16 \
    --model.n_nodes 1024 \
    --trainer.seed_everything $SEED \
    --trainer.max_epochs 500
cd ..

mkdir -p sbi_over_bkg && cd sbi_over_bkg
python -m nsbi.carl fit \
    --data.features '["l1_pt", "l1_eta", "l1_phi", "l1_energy", "l2_pt", "l2_eta", "l2_phi", "l2_energy", "l3_pt", "l3_eta", "l3_phi", "l3_energy", "l4_pt", "l4_eta", "l4_phi", "l4_energy"]' \
    --data.numerator_events '/ptmp/mpp/taepa/higgs-offshell-interpretation/data/zz4l/ggZZ_sbi/analyzed.csv' \
    --data.denominator_events '/ptmp/mpp/taepa/higgs-offshell-interpretation/data/zz4l/ggZZ_sbi/analyzed.csv' \
    --data.denominator_reweight '["sbi","bkg"]' \
    --data.sample_size 10_000_000 \
    --data.batch_size 1024 \
    --model.learning_rate 1e-5 \
    --model.n_layers 16 \
    --model.n_nodes 1024 \
    --trainer.seed_everything $SEED \
    --trainer.max_epochs 500