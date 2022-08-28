#!/bin/bash
#SBATCH -p gpu-long
#SBATCH -c 24
#SBATCH --mem=128g
#SBATCH --gres=gpu:a100:4
#SBATCH -o train.log
#SBATCH -J ligand_dock_sm

#export CUDA_VISIBLE_DEVICES=0

source activate SE3nv
python -u ./train_multi_EMA.py \
    -model_name BFF20h \
    -p_drop 0.0 \
    -maxcycle 4 \
    -n_extra_block 2 \
    -n_main_block 4 \
    -n_ref_block 2 \
    -n_finetune_block 0 \
    -ref_num_layers 2 \
    -d_msa 64 \
    -d_pair 64 \
    -accum 1 \
    -crop 256 \
    -w_bond 0.0 \
    -w_dih 0.0 \
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
    -port 12346 \
    -wandb_prefix ligand_dock_sm \
    #-eval
