#!/bin/sh

python predict.py -msa pep1.fa -disulfidize_residues 0:10 -n_cycle 20
python predict.py -msa pep2.fa -disulfidize_residues 0:7 -n_cycle 20
