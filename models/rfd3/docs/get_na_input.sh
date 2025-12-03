#!/bin/bash
# dsDNA
wget https://files.rcsb.org/download/1bna.pdb1.gz
# dsDNA with protein
wget https://files.rcsb.org/download/2r5z.pdb1.gz
# ssDNA
wget https://files.rcsb.org/download/5o4d.pdb1.gz
# RNA
wget https://files.rcsb.org/download/1q75.pdb1.gz

gunzip *.gz

mkdir input_pdbs

mv 1bna.pdb1 ./input_pdbs/1bna.pdb
mv 2r5z.pdb1 ./input_pdbs/2r5z.pdb
mv 5o4d.pdb1 ./input_pdbs/5o4d.pdb
mv 1q75.pdb1 ./input_pdbs/1q75.pdb