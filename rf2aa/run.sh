#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

python -u ./train_multi_EMA.py \
    -model_name BFF20h \
    -p_drop 0.0 \
    -maxcycle 4 \
    -n_extra_block 4 \
    -n_main_block 32 \
    -n_ref_block 4 \
    -n_finetune_block 0 \
    -ref_num_layers 2 \
    -accum 4 \
    -crop 256 \
    -w_bond 0.0 \
    -w_bind 0.0 \
    -w_clash 0.0 \
    -w_hb 0.0 \
    -lj_lin 0.7 \
    -w_dist 1.0 \
    -w_str 10.0 \
    -w_lddt 0.1 \
    -w_aa 3.0 \
    -subsmp UNI \
    -num_epochs 400 \
    -slice CONT \
    -lr 0.001 \
    -port 12345 \
    -eval
