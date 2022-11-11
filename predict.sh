# for testing predict.py
python predict.py \
    -pt /projects/ml/TrRosetta/PDB-2021AUG02/torch/pdb/fn/6fnr_A.pt \
    -fasta /projects/ml/TrRosetta/PDB-2021AUG02/a3m/019/019582.a3m.gz \
    -mol2 /projects/ml/RF2_allatom/by-pdb/fn/6fnr_DYT_1_A_403__E___.mol2 \
    -out 6fnr_A_pred -dump_extra_pdbs -dump_traj -dump_aux -n_cycle 2
    #-pdb /projects/ml/TrRosetta/PDB-2021AUG02/pdb/pdb/fn/6fnr_A.pdb \
