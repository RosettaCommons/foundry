# Option conversion for existing scripts

To help convert existing usage, this page lists the (rough) option correspondence for the standalone ProteinMPNN and LigandMPNN programs.

Note that the conversion is approximate and functional. Input formats may not match up exactly.

## ProteinMPNN/SolubleMPNN

    --pdb_path               --structure_path
    --jsonl_path             --structure_path
    --out_folder             --out_directory
    --seed                   --seed # The same seed will NOT produce the same output.
    --batch_size             --batch_size
    --num_seq_per_target     # No exact correspondence: use --batch_size & --number_of_batches together to specify.
    --sampling_temp          --temperature
    --pdb_path_chains        --designed_chains # Though comma separated, rather than space separated
    --chain_id_jsonl         # Use --fixed_chains/--designed_chains/--fixed_residues/--designed_residues
    --fixed_positions_jsonl  --fixed_residues
    --tied_positions_jsonl   # Use --homo_oligomer_chains or --symmetry_residues instead
    --bias_AA_jsonl          --bias
    --bias_by_res_jsonl      --bias_per_residue
    --path_to_model_weights  --checkpoint_path # Set --is_legacy_weights too.
    --model_name             --checkpoint_path
    --use_soluble_model      --model_type soluble_mpnn
    --backbone_noise         --structure_noise
    --omit_AAs               --omit # Three letter code instead of one letter
    --omit_AA_jsonl          --omit_per_residue
    
Options which are not currently supported or are unneeded:

    --suppress_print
    --score_only
    --save_score
    --save_probs
    --conditional_probs_only
    --conditional_probs_only_backbone
    --unconditional_probs_only
    --path_to_fasta
    --ca_only
    --unconditional_probs_only
    --pssm_jsonl
    --pssm_multi 
    --pssm_threshold
    --pssm_log_odds_flag
    --pssm_bias_flag
    --max_length

## LigandMPNN
    
    --pdb_path                            --structure_path
    --out_folder                          --out_directory
    --model_type                          --model_type
    --seed                                --seed # The same seed will NOT produce the same output.
    --batch_size                          --batch_size
    --number_of_batches                   --number_of_batches
    --temperature                         --temperature
    --fixed_residues                      --fixed_residues
    --redesigned_residues                 --designed_residues
    --bias_AA                             --bias
    --bias_AA_per_residue                 --bias_per_residue
    --omit_AA                             --omit
    --omit_AA_per_residue                 --omit_per_residue
    --symmetry_residues                   --symmetry_residues
    --symmetry_weights                    --symmetry_residues_weights
    --homo_oligomer                       --homo_oligomer # As a correspondence, rather than a bool
    --chains_to_design                    --designed_chains
    --checkpoint_protein_mpnn             --checkpoint_path
    --checkpoint_ligand_mpnn              --checkpoint_path
    --checkpoint_soluble_mpnn             --checkpoint_path
    --use_sequence                        --initialize_sequence_embedding_with_ground_truth
    --ligand_mpnn_use_atom_context        # Can mock with --remove_ccds
    --ligand_mpnn_use_side_chain_context  --atomize_side_chains
    --parse_atoms_with_zero_occupancy     --occupancy_threshold_sidechain & --occupancy_threshold_backbone
    
Multiple structure inputs are not supported on the command line -- provide them in a `--config_json` file.
 
    --pdb_path_multi
    --fixed_residues_multi
    --redesigned_residues_multi
    --omit_AA_per_residue_multi
    --bias_AA_per_residue_multi

The ability to pack sidechains in the output models is currently not supported:

    --pack_side_chains
    --checkpoint_path_sc
    --packed_suffix
    --number_of_packs_per_design
    --pack_with_ligand_context
    --repack_everything
    --sc_num_denoising_steps
    --sc_num_samples

Options which are not currently or are unneeded:

    --verbose
    --save_stats
    --file_ending    # --name can possibly substitute
    --zero_indexed   # Zero indexed is the only option for foundry
    --parse_these_chains_only
    --model_type global_label_membrane_mpnn
    --checkpoint_global_label_membrane_mpnn
    --global_transmembrane_label
    --model_type per_residue_label_membrane_mpnn
    --checkpoint_per_residue_label_membrane_mpnn
    --transmembrane_buried
    --transmembrane_interface
    --fasta_seq_separation
    --ligand_mpnn_cutoff_for_score
    --autoregressive_score
    --single_aa_score
    --force_hetatm
