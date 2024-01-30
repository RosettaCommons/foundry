import argparse
from rf2aa import data_loader
from typing import Optional, List
import datetime


DATASET_PARAMS = [
    "fraction_pdb",
    "fraction_fb",
    "fraction_compl",
    "fraction_neg_compl",
    "fraction_na_compl",
    "fraction_neg_na_compl",
    "fraction_distil_tf",
    "fraction_tf",
    "fraction_neg_tf",
    "fraction_rna",
    "fraction_dna",
    "fraction_sm_compl",
    "fraction_sm_compl_multi",
    "fraction_metal_compl",
    "fraction_sm_compl_covale",
    "fraction_sm",
    "fraction_atomize_pdb",
    "fraction_atomize_complex",
    "fraction_sm_compl_asmb",
    "fraction_sm_compl_furthest_neg",
    "fraction_sm_compl_permuted_neg",
    "fraction_sm_compl_docked_neg",
    "n_train",
    "n_valid_pdb",
    "n_valid_atomize_pdb",
    "n_valid_homo",
    "n_valid_dslf",
    "n_valid_compl",
    "n_valid_neg_compl",
    "n_valid_na_compl",
    "n_valid_neg_na_compl",
    "n_valid_distil_tf",
    "n_valid_tf",
    "n_valid_neg_tf",
    "n_valid_rna",
    "n_valid_dna",
    "n_valid_sm_compl",
    "n_valid_sm_compl_multi",
    "n_valid_metal_compl",
    "n_valid_sm_compl_covale",
    "n_valid_sm_compl_strict",
    "n_valid_sm",
    "n_valid_sm_compl_asmb",
    "n_valid_sm_compl_furthest_neg",
    "n_valid_sm_compl_permuted_neg",
    "n_valid_sm_compl_docked_neg",
    "n_valid_atomize_complex",
    "n_valid_dude_actives",
    "n_valid_dude_inactives",
    "p_short_crop",
    "p_dslf_crop",
    "dslf_fb_upsample",
]
TRUNK_PARAMS = [
    "n_extra_block",
    "n_main_block",
    "n_ref_block",
    "n_finetune_block",
    "d_msa",
    "d_msa_full",
    "d_pair",
    "d_templ",
    "n_head_msa",
    "n_head_pair",
    "n_head_templ",
    "d_hidden",
    "d_hidden_templ",
    "p_drop",
    "use_extra_l1",
    "use_atom_frames",
]
SE3_PARAMS = [
    "num_layers",
    "num_channels",
    "num_degrees",
    "n_heads",
    "div",
    "l0_in_features",
    "l0_out_features",
    "l1_in_features",
    "l1_out_features",
    "num_edge_features",
]


def get_args(parser: Optional[argparse.ArgumentParser] = None, input_args: Optional[List[str]] = None):
    if parser is None:
        parser = argparse.ArgumentParser()

    # training parameters
    train_group = parser.add_argument_group("training parameters")
    train_group.add_argument(
        "-shuffle_dataloader",
        action="store_true",
        help="set to True to shuffle data loaders during training",
    )
    train_group.add_argument(
        "-dataloader_num_workers",
        type=int,
        default=5,
        help="Number of loader workers to use while training",
    )
    train_group.add_argument(
        "-dont_pin_memory",
        action="store_true",
        help="Set to True to disable CPU buffered memory pinning during data loading",
    )
    train_group.add_argument("-model_name", default=None, help="model name for saving")
    train_group.add_argument("-checkpoint_path", default=None, help="path to a stored checkpoint")
    train_group.add_argument("-batch_size", type=int, default=1, help="Batch size [1]")
    train_group.add_argument(
        "-lr", type=float, default=2.0e-4, help="Learning rate [5.0e-4]"
    )
    train_group.add_argument(
        "-num_epochs", type=int, default=300, help="Number of epochs [300]"
    )
    train_group.add_argument(
        "-skip_valid",
        type=int,
        default=1,
        help="Do valid cycles every <skip_valid> epochs [1]",
    )
    train_group.add_argument(
        "-step_lr", type=int, default=300, help="Parameter for Step LR scheduler [300]"
    )
    train_group.add_argument(
        "-port",
        type=int,
        default=12319,
        help="PORT for ddp training, should be randomized [12319]",
    )
    train_group.add_argument(
        "-accum", type=int, default=1, help="Gradient accumulation when it's > 1 [1]"
    )
    train_group.add_argument(
        "-eval",
        action="store_true",
        default=False,
        help="No training, just run validation cycles and output structures",
    )
    train_group.add_argument(
        "-start_epoch",
        type=int,
        default=None,
        help="When using -eval, which epoch to start at. "
        "A checkpoint must exist for this epoch.",
    )
    train_group.add_argument(
        "-out_dir", type=str, default="test/", help="Output folder."
    )
    train_group.add_argument(
        "-wandb_prefix",
        type=str,
        help="Prefix for name of session on Weights and Biases.",
    )
    train_group.add_argument(
        "-datestamp",
        action="store_true",
        default=False,
        help="Adds datestamp to output folder and/or wandb prefix.",
    )
    train_group.add_argument(
        "-model_dir",
        type=str,
        default="models/",
        help="Output folder for model weights. [models/]",
    )
    train_group.add_argument(
        "-interactive",
        action="store_true",
        default=False,
        help="Start training in interactive mode. [False]",
    )
    train_group.add_argument(
        "-debug",
        action="store_true",
        default=False,
        help="Run in debugging mode. [False]",
    )

    # data-loading parameters
    data_group = parser.add_argument_group("data loading parameters")
    data_group.add_argument(
        "-maxseq", type=int, default=1024, help="Maximum depth of subsampled MSA [1024]"
    )
    data_group.add_argument(
        "-maxtoken",
        type=int,
        default=2**18,
        help="Maximum depth of subsampled MSA [2**18]",
    )
    data_group.add_argument(
        "-maxlat", type=int, default=128, help="Maximum depth of subsampled MSA [128]"
    )
    data_group.add_argument(
        "-crop", type=int, default=260, help="Upper limit of crop size [260]"
    )
    data_group.add_argument(
        "-rescut", type=float, default=4.5, help="Resolution cutoff [4.5]"
    )
    data_group.add_argument(
        "-slice",
        type=str,
        default="DISCONT",
        help="How to make crops [CONT / DISCONT (default)]",
    )
    data_group.add_argument(
        "-subsmp",
        type=str,
        default="UNI",
        help="How to subsample MSAs [UNI (default) / LOG / CONST]",
    )
    data_group.add_argument(
        "-mintplt",
        type=int,
        default=1,
        help="Minimum number of templates to select [1]",
    )
    data_group.add_argument(
        "-maxtplt",
        type=int,
        default=4,
        help="maximum number of templates to select [4]",
    )
    data_group.add_argument(
        "-seqid",
        type=float,
        default=150.0,
        help="maximum sequence identity cutoff for template selection [150.0]",
    )
    data_group.add_argument(
        "-maxcycle", type=int, default=4, help="maximum number of recycle [4]"
    )
    data_group.add_argument(
        "-nres_atomize_min",
        type=int,
        default=3,
        help="minimum number of residues to atomize [3]",
    )
    data_group.add_argument(
        "-nres_atomize_max",
        type=int,
        default=5,
        help="maximum number of residues to atomize [5]",
    )
    data_group.add_argument(
        "-atomize_flank",
        type=int,
        default=0,
        help="flanking residues to remove when atomizing [0]",
    )
    data_group.add_argument(
        "-p_metal",
        type=float,
        default=1,
        help="probability of a given metal ion being included [1.0]",
    )
    data_group.add_argument(
        "-p_atomize_modres",
        type=float,
        default=1,
        help="probability of a given non-standard residue being atomized, rather "
        "than being converted to a standard equivalent [1.0]",
    )
    data_group.add_argument(
        "-batch_by_dataset",
        action="store_true",
        default=False,
        help="Batch examples by dataset, e.g., all nodes receive an example from the same dataset. [False]",
    )
    data_group.add_argument(
        "-batch_by_length",
        action="store_true",
        default=False,
        help="Batch examples by example length, e.g., all nodes receive a similarly-sized example. [False]",
    )


    # dataset parameters
    dataset_group = parser.add_argument_group("data loading parameters")
    dataset_group.add_argument('-fraction_pdb', type=float, default=0.09, 
            help="how often to sample PDB monomers during training")
    dataset_group.add_argument('-fraction_fb', type=float, default=0.09, 
            help="how often to sample AF2 predictions from FB during training")
    dataset_group.add_argument('-fraction_compl', type=float, default=0.09, 
            help="how often to sample protein-protein complexes during training")
    dataset_group.add_argument('-fraction_neg_compl', type=float, default=0.09, 
            help="how often to sample negative protein-protein complexes during training")
    dataset_group.add_argument('-fraction_na_compl', type=float, default=0.09,
            help="how often to sample protein-nucleic acid complexes during training")
    dataset_group.add_argument('-fraction_neg_na_compl', type=float, default=0.09,
            help="how often to sample negative protein-nucleic acid complexes during training")
    dataset_group.add_argument('-fraction_distil_tf', type=float, default=0.00,
            help="how often to sample distilled protein-DNA complexes from TF data during training")
    dataset_group.add_argument('-fraction_tf', type=float, default=0.00,
            help="how often to sample protein-DNA complexes from TF profile data during training")
    dataset_group.add_argument('-fraction_neg_tf', type=float, default=0.00,
            help="how often to sample negative protein-DNA complexes from TF profile data during training")
    dataset_group.add_argument('-fraction_rna', type=float, default=0.09,
            help="how often to sample rna during training")
    dataset_group.add_argument('-fraction_dna', type=float, default=0.00,
            help="how often to sample dna during training")
    dataset_group.add_argument('-fraction_sm_compl', type=float, default=0.1,
            help="how often to sample protein small molecule complexes during training")
    dataset_group.add_argument('-fraction_metal_compl', type=float, default=0.09,
            help="how often to sample protein/metal complexes during training")    
    dataset_group.add_argument('-fraction_sm_compl_multi', type=float, default=0.09,
            help="how often to sample protein/multiresidue small molecule complexes during training")
    dataset_group.add_argument('-fraction_sm_compl_covale', type=float, default=0.09,
            help="how often to sample covalent protein/small molecule complexes during training")    
    dataset_group.add_argument('-fraction_sm', type=float, default=0.00,
            help="how often to sample small molecule crystals during training")
    dataset_group.add_argument('-fraction_atomize_pdb', type=float, default=0.00,
            help="how often to sample atomized pdb monomers during training")
    dataset_group.add_argument('-fraction_atomize_complex', type=float, default=0.00,
            help="how often to sample atomized pdb monomers during training")
    dataset_group.add_argument('-fraction_sm_compl_asmb', type=float, default=0.00,
            help="how often to sample atomized pdb monomers during training")
    dataset_group.add_argument('-fraction_sm_compl_furthest_neg', type=float, default=0.0,
            help="how often to sample protein small molecule complexes that are defined by the furthest residues")
    dataset_group.add_argument('-fraction_sm_compl_permuted_neg', type=float, default=0.0,
            help="how often to sample protein small molecule complexes with permuted ligands")
    dataset_group.add_argument('-fraction_sm_compl_docked_neg', type=float, default=0.0,
            help="how often to sample protein/small molecule complexes validated to be negatives by VINA")
    

    dataset_group.add_argument('-n_train', type=int, default=12288)
    dataset_group.add_argument('-n_valid_pdb', type=int)
    dataset_group.add_argument('-n_valid_homo', type=int)
    dataset_group.add_argument('-n_valid_dslf', type=int)
    dataset_group.add_argument('-n_valid_compl', type=int)
    dataset_group.add_argument('-n_valid_neg_compl', type=int)
    dataset_group.add_argument('-n_valid_na_compl', type=int)
    dataset_group.add_argument('-n_valid_neg_na_compl', type=int)
    dataset_group.add_argument('-n_valid_distil_tf', type=int)
    dataset_group.add_argument('-n_valid_tf', type=int)
    dataset_group.add_argument('-n_valid_neg_tf', type=int)
    dataset_group.add_argument('-n_valid_rna', type=int)
    dataset_group.add_argument('-n_valid_dna', type=int)
    dataset_group.add_argument('-n_valid_sm_compl', type=int)
    dataset_group.add_argument('-n_valid_metal_compl', type=int)
    dataset_group.add_argument('-n_valid_sm_compl_multi', type=int)
    dataset_group.add_argument('-n_valid_sm_compl_covale', type=int)
    dataset_group.add_argument('-n_valid_sm_compl_strict', type=int)
    dataset_group.add_argument('-n_valid_sm', type=int)
    dataset_group.add_argument('-n_valid_atomize_pdb', type=int)
    dataset_group.add_argument('-n_valid_atomize_complex', type=int)
    dataset_group.add_argument('-n_valid_sm_compl_asmb', type=int)
    dataset_group.add_argument(
        "-n_valid_sm_compl_furthest_neg",
        type=int,
        help="Set this to a number to validate on negative examples of sm complexes that are generated by taking the furthest residues in a protein from the ligand (assumedly, such residues do not bind to the ligand).",
    )
    dataset_group.add_argument(
        "-n_valid_sm_compl_permuted_neg",
        type=int,
        help="Set this to a number to validate on negative examples created by randomly shuffling which ligand gets assigned to which protein, with some filtering to make sure the swapped ligands are dissimilar.",
    )
    dataset_group.add_argument(
        "-n_valid_sm_compl_docked_neg",
        type=int,
        help="Number to validate from the autodocked negative set.",
    )
    dataset_group.add_argument(
        "-n_valid_dude_actives",
        type=int,
        help="Number to validate from the DUD-e set, true binders.",
    )
    dataset_group.add_argument(
        "-n_valid_dude_inactives",
        type=int,
        help="Number to validate from the DUD-e set, non binders.",
    )
    dataset_group.add_argument(
        "-p_short_crop",
        type=float,
        default=0.0,
        help="The probability (0-1) of sampling a crop size between 8 and 16 residues",
    )
    dataset_group.add_argument(
        "-p_dslf_crop",
        type=float,
        default=0.0,
        help="The probability (0-1) of cropping a disulfide-linked loop",
    )
    dataset_group.add_argument(
        "-dslf_fb_upsample", 
        type=float,
        default=1.0,
        help="Upsample disulfide-containing FB models by this factor"
    )


    # Trunk module properties
    trunk_group = parser.add_argument_group("Trunk module parameters")
    trunk_group.add_argument(
        "-n_extra_block",
        type=int,
        default=4,
        help="Number of iteration blocks for extra sequences [4]",
    )
    trunk_group.add_argument(
        "-n_main_block",
        type=int,
        default=8,
        help="Number of iteration blocks for main sequences [8]",
    )
    trunk_group.add_argument(
        "-n_ref_block", type=int, default=4, help="Number of refinement layers"
    )
    trunk_group.add_argument(
        "-n_finetune_block", type=int, default=0, help="Number of finetune layers"[0]
    )
    trunk_group.add_argument(
        "-d_msa", type=int, default=256, help="Number of MSA features [256]"
    )
    trunk_group.add_argument(
        "-d_msa_full", type=int, default=64, help="Number of MSA features [64]"
    )
    trunk_group.add_argument(
        "-d_pair", type=int, default=128, help="Number of pair features [128]"
    )
    trunk_group.add_argument(
        "-d_templ", type=int, default=64, help="Number of templ features [64]"
    )
    trunk_group.add_argument(
        "-n_head_msa",
        type=int,
        default=8,
        help="Number of attention heads for MSA2MSA [8]",
    )
    trunk_group.add_argument(
        "-n_head_pair",
        type=int,
        default=4,
        help="Number of attention heads for Pair2Pair [4]",
    )
    trunk_group.add_argument(
        "-n_head_templ",
        type=int,
        default=4,
        help="Number of attention heads for template [4]",
    )
    trunk_group.add_argument(
        "-d_hidden", type=int, default=32, help="Number of hidden features [32]"
    )
    trunk_group.add_argument(
        "-d_hidden_templ",
        type=int,
        default=64,
        help="Number of hidden features for templates [64]",
    )
    trunk_group.add_argument(
        "-p_drop", type=float, default=0.15, help="Dropout ratio [0.15]"
    )
    trunk_group.add_argument(
        "-no_extra_l1",
        dest="use_extra_l1",
        default="True",
        action="store_false",
        help="Turn off chirality and LJ grad inputs to SE3 layers (for backwards compatibility).",
    )
    trunk_group.add_argument(
        "-no_atom_frames",
        dest="use_atom_frames",
        default="True",
        action="store_false",
        help="Turn off l1 features from atom frames in SE3 layers (for backwards compatibility).",
    )

    # Structure module properties
    str_group = parser.add_argument_group("structure module parameters")
    str_group.add_argument(
        "-num_layers",
        type=int,
        default=1,
        help="Number of equivariant layers in structure module block [1]",
    )
    str_group.add_argument(
        "-num_channels", type=int, default=32, help="Number of channels [32]"
    )
    str_group.add_argument(
        "-num_degrees",
        type=int,
        default=2,
        help="Number of degrees for SE(3) network [2]",
    )
    str_group.add_argument(
        "-l0_in_features",
        type=int,
        default=64,
        help="Number of type 0 input features [64]",
    )
    str_group.add_argument(
        "-l0_out_features",
        type=int,
        default=64,
        help="Number of type 0 output features [64]",
    )
    str_group.add_argument(
        "-l1_in_features",
        type=int,
        default=3,
        help="Number of type 1 input features [3]",
    )
    str_group.add_argument(
        "-l1_out_features",
        type=int,
        default=2,
        help="Number of type 1 output features [2]",
    )
    str_group.add_argument(
        "-num_edge_features", type=int, default=64, help="Number of edge features [64]"
    )
    str_group.add_argument(
        "-n_heads",
        type=int,
        default=4,
        help="Number of attention heads for SE3-Transformer [4]",
    )
    str_group.add_argument(
        "-div", type=int, default=4, help="Div parameter for SE3-Transformer [4]"
    )
    str_group.add_argument(
        "-ref_num_layers",
        type=int,
        default=1,
        help="Number of equivariant layers in structure module block [1]",
    )
    str_group.add_argument(
        "-ref_num_channels", type=int, default=32, help="Number of channels [32]"
    )

    # Loss function parameters
    loss_group = parser.add_argument_group("loss parameters")
    loss_group.add_argument('-w_dist', type=float, default=1.0,
            help="Weight on distd in loss function [1.0]")
    loss_group.add_argument('-w_str', type=float, default=10.0,
            help="Weight on strd in loss function [10.0]")
    loss_group.add_argument('-w_inter_fape', type=float, default=2.0,
            help="Weight on inter-chain backbone fape in loss function [2.0]")
    loss_group.add_argument('-w_lig_fape', type=float, default=10,
            help="Weight on ligand fape in loss function [10.0]")
    loss_group.add_argument('-w_lddt', type=float, default=0.1,
            help="Weight on predicted lddt loss [0.1]")
    loss_group.add_argument('-w_aa', type=float, default=3.0,
            help="Weight on MSA masked token prediction loss [3.0]")
    loss_group.add_argument('-w_bond', type=float, default=0.0,
            help="Weight on predicted bond loss [0.0]")
    loss_group.add_argument('-w_bind', type=float, default=0.0,
            help="Weight on bind/no-bind predictions [0.0]")
    loss_group.add_argument("-binder_loss_label_smoothing", type=float, default=0.0,
            help="Label smoothing for binder loss. Must be in range [0.0, 0.5)")
    loss_group.add_argument('-w_clash', type=float, default=0.0,
            help="Weight on clash loss [0.0]")
    loss_group.add_argument('-w_atom_bond', type=float, default=0.0,
            help="Weight on atom bond loss [0.0]")
    loss_group.add_argument('-w_skip_bond', type=float, default=0.0,
            help="Weight on skip bond distance loss [0.0]") 
    loss_group.add_argument('-w_rigid', type=float, default=0.0,
            help="Weight on rigid body distance loss [0.0]")      
    loss_group.add_argument('-w_hb', type=float, default=0.0,
            help="Weight on hydrogen bond loss [0.0]")
    loss_group.add_argument('-w_pae', type=float, default=0.05,
            help="Weight on pae loss [0.05]")      
    loss_group.add_argument('-w_pde', type=float, default=0.05,
            help="Weight on pde loss [0.05]")      
    loss_group.add_argument('-lj_lin', type=float, default=0.75,
            help="linear inflection for lj [0.75]")
    
    # parse arguments
    args = parser.parse_args(args=input_args)

    if args.datestamp:
        datestr = (
            str(datetime.datetime.now()).replace(":", "").replace(" ", "_")
        )  # YYYY-MM-DD_HHMMSS.xxxxxx
        if args.out_dir is not None:
            args.out_dir = (args.out_dir + "_" + datestr).replace("/_", "/") + "/"
        if args.wandb_prefix is not None:
            args.wandb_prefix = args.wandb_prefix + "_" + datestr

    # Setup dataloader parameters:
    loader_param = data_loader.set_data_loader_params(args)

    # make dictionary for each parameters
    dataset_param = {}
    for param in DATASET_PARAMS:
        dataset_param[param] = getattr(args, param)
    trunk_param = {}
    for param in TRUNK_PARAMS:
        trunk_param[param] = getattr(args, param)
    SE3_param = {}
    for param in SE3_PARAMS:
        if hasattr(args, param):
            SE3_param[param] = getattr(args, param)

    SE3_ref_param = SE3_param.copy()

    for param in SE3_PARAMS:
        if hasattr(args, "ref_" + param):
            SE3_ref_param[param] = getattr(args, "ref_" + param)

    # print (SE3_param)
    # print (SE3_ref_param)
    trunk_param["SE3_param"] = SE3_param
    trunk_param["SE3_ref_param"] = SE3_ref_param

    loss_param = {}
    for param in ['w_dist', 'w_str', 'w_inter_fape', 'w_lig_fape', 'w_aa', 'w_lddt', 'w_bond', 
                  'w_bind', 'binder_loss_label_smoothing', 'w_clash', 'w_hb', 'lj_lin', 'w_atom_bond', 'w_skip_bond', 'w_rigid', 'w_pae',
                  'w_pde']:
        loss_param[param] = getattr(args, param)

    return args, dataset_param, trunk_param, loader_param, loss_param
