#!/bin/bash

foundry=../../../

export PYTHONPATH="$foundry/src:$foundry/models/rfd3/src/"


outdir=./na_tutorial_outputs/
rm $outdir/*
ckpt_path=rfd3_foundry_2025_12_01.ckpt
uv run python $foundry/models/rfd3/src/rfd3/run_inference.py ckpt_path=$ckpt_path out_dir=$outdir inputs=./na_binder_design.json n_batches=2 diffusion_batch_size=3 cleanup_virtual_atoms=True

#some cleanup
rm *.hb2
rm *.pdb
rm *.dat
