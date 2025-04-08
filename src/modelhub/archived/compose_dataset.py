import logging
import os
import pickle
from collections import OrderedDict

import numpy as np
import torch
import torch.utils.data as data

logger = logging.getLogger(__name__)

from rf2aa.data.data_loader import (
    Dataset,
    DatasetNAComplex,
    DatasetRNA,
    DatasetSM,
    DatasetSMComplexAssembly,
    DistilledDataset,
    _load_df,
    get_train_valid_set,
    loader_atomize_complex,
    loader_atomize_pdb,
    loader_complex,
    loader_distil_tf,
    loader_dna_rna,
    loader_fb,
    loader_na_complex,
    loader_pdb,
    loader_sm,
    loader_tf_complex,
)
from rf2aa.data.loaders.rcsb_loader import (
    loader_sm_compl_assembly,
    loader_sm_compl_assembly_single,
)
from rf2aa.data.loaders.spoofing import spoofed_loader
from rf2aa.data.sampler import sampler_factory

#### handle defaults
# TODO: shouldn't have to do this in the future, should all be handled in config

base_dir = "/projects/ml/TrRosetta/PDB-2021AUG02"
compl_dir = "/projects/ml/RoseTTAComplex"
na_dir = "/projects/ml/nucleic"
fb_dir = "/projects/ml/TrRosetta/fb_af"
sm_compl_dir = "/projects/ml/RF2_allatom"
mol_dir = "/projects/ml/RF2_allatom/rcsb/pkl"  # for phase 3 dataloaders
# mol_dir = "/projects/ml/RF2_allatom/isdf" # for legacy datasets
tf_dir = "/projects/ml/prot_dna"
csd_dir = "/databases/csd543"

# fd store the pickle file in rf2aa directory
#   rf2aa is parent dir to this one
rf2aa_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)

# posebusters_csv = os.path.join(rf2aa_file_path,"posebusters_benchmark.csv")
posebusters_csv = os.path.join(rf2aa_file_path, "valid_sets/posebusters_clean.csv")

default_dataloader_params = {
    "COMPL_LIST": "%s/list.hetero.csv" % compl_dir,
    "HOMO_LIST": "%s/list.homo.csv" % compl_dir,
    "NEGATIVE_LIST": "%s/list.negative.csv" % compl_dir,
    "RNA_LIST": "%s/list.rnaonly.csv" % na_dir,
    "DNA_LIST": "%s/list.dnaonly.v3.csv" % na_dir,
    "NA_COMPL_LIST": "%s/list.nucleic.v3.csv" % na_dir,
    "NEG_NA_COMPL_LIST": "%s/list.na_negatives.v3.csv" % na_dir,
    "TF_DISTIL_LIST": "%s/prot_na_distill.v3.csv" % tf_dir,
    "TF_COMPL_LIST": "%s/tf_compl_list.v4.csv" % tf_dir,
    "SM_COMPL_DIR": sm_compl_dir,
    "SM_LIST": "%s/sm_compl_all_20230418.csv" % sm_compl_dir,
    "PDB_LIST": "%s/list_v02_w_taxid.csv" % sm_compl_dir,  # on digs
    "PDB_METADATA": "%s/list_v00_w_taxid_20230201.csv" % sm_compl_dir,  # on digs
    "FB_LIST": "%s/list_b1-3.csv" % fb_dir,
    "CSD_LIST": "%s/csd543_cleaned01.csv" % csd_dir,
    "VAL_PDB": "%s/valid_remapped" % sm_compl_dir,
    "VAL_RNA": "%s/rna_valid.csv" % na_dir,
    "VAL_DNA": "%s/dna_valid.csv" % na_dir,
    "VAL_COMPL": "%s/val_lists/xaa" % compl_dir,
    "VAL_NEG": "%s/val_lists/xaa.neg" % compl_dir,
    "VAL_TF": "%s/tf_valid_clusters_v4.txt" % tf_dir,
    "VAL_SM_STRICT": "%s/sm_compl_valid_strict_20230418.csv" % sm_compl_dir,
    "TEST_SM": "%s/sm_test_heldout_test_clusters.txt" % sm_compl_dir,
    "DATAPKL": "%s/dataset_20240318.pkl" % rf2aa_file_path,  # cache for faster loading
    "DSLF_LIST": "%s/list.dslf.csv" % na_dir,
    "DSLF_FB_LIST": "%s/list.dslf_fb.csv" % na_dir,
    "DUDE_LIST": "/home/dnan/projects/gald_distil_set/nbs/dude_dataset_cutoff_-5.csv",  # on digs (dnan)
    "DUDE_MSAS": "/home/dnan/projects/gald_distil_set/DUDE/fastas",  # on digs (dnan)
    "DUDE_PDB_DIR": "/home/dnan/projects/gald_distil_set/DUDE/pdbs_all",
    # See rf2aa/tools/generate_sample_lengths.py and rf2aa/tools/edit_lengths_for_removed_datasets.py for how this "EXAMPLE_LENGTHS" is generated
    "EXAMPLE_LENGTHS": "/projects/ml/RF2_allatom/all_sample_lengths_crop_1K_no_negatives.pt",
    "PDB_DIR": base_dir,
    "FB_DIR": fb_dir,
    "COMPL_DIR": compl_dir,
    "NA_DIR": na_dir,
    "TF_DIR": tf_dir,
    "MOL_DIR": mol_dir,
    "CSD_DIR": csd_dir,
    "MINTPLT": 0,
    "MAXTPLT": 5,
    "MINSEQ": 1,
    "MAXSEQ": 1024,
    "MAXLAT": 128,
    "CROP": 256,
    "DATCUT": "2021-Aug-1",
    "RESCUT": 4.5,
    "BLOCKCUT": 5,
    "PLDDTCUT": 70.0,
    "SCCUT": 90.0,
    "ROWS": 1,
    "SEQID": 95.0,
    "MAXCYCLE": 4,
    "RMAX": 5.0,
    "MAXRES": 1,
    "MINATOMS": 5,
    "MAXATOMS": 100,
    "MAXSIM": 0.85,
    "MAXNSYMM": 1024,
    "NRES_ATOMIZE_MIN": 5,
    "NRES_ATOMIZE_MAX": 15,
    "ATOMIZE_FLANK": 0,
    "MAXPROTCHAINS": 6,
    "MAXLIGCHAINS": 10,
    "MAXMASKEDLIGATOMS": 30,
    "P_METAL": 0.75,
    "P_ATOMIZE_MODRES": 0.75,
    "MAXMONOMERLENGTH": None,
    "ATOMIZE_CLUSTER": True,
    "P_ATOMIZE_TEMPLATE": 0.0,
    "NUM_SEQS_SUBSAMPLE": 50,
    "BLACK_HOLE_INIT": False,
    "SHOW_SM_TEMPLATES": False,
    "BATCH_BY_DATASET": False,
    "BATCH_BY_LENGTH": False,
    "ligands_to_remove": [],
    "min_metal_contacts": 0,
    "crop_params": {},
}


def set_data_loader_params(loader_params):
    """add things from config into default dataloader params"""
    for param in default_dataloader_params:
        if hasattr(loader_params, param.lower()):
            default_dataloader_params[param] = getattr(loader_params, param.lower())
    # cursed but add things in the param but not the default params back
    loader_params_dict = dict(loader_params)
    for param, value in loader_params_dict.items():
        if param.upper() not in default_dataloader_params:
            default_dataloader_params[param] = value
    return default_dataloader_params


def get_distilled_dataset(dataset_params, loader_params):
    (
        train_ID_dict,
        valid_ID_dict,
        weights_dict,
        train_dict,
        valid_dict,
        homo,
        chid2hash,
        chid2taxid,
        chid2smpartners,
    ) = get_train_valid_set(loader_params, no_match_okay=loader_params["no_match_okay"])
    # define atomize_pdb train/valid sets, which use the same examples as pdb set
    train_ID_dict["atomize_pdb"] = train_ID_dict["pdb"]
    valid_ID_dict["atomize_pdb"] = valid_ID_dict["pdb"]
    weights_dict["atomize_pdb"] = weights_dict["pdb"]
    train_dict["atomize_pdb"] = train_dict["pdb"]
    valid_dict["atomize_pdb"] = valid_dict["pdb"]

    # define atomize_pdb train/valid sets, which use the same examples as pdb set
    train_ID_dict["atomize_complex"] = train_ID_dict["compl"]
    valid_ID_dict["atomize_complex"] = valid_ID_dict["compl"]
    weights_dict["atomize_complex"] = weights_dict["compl"]
    train_dict["atomize_complex"] = train_dict["compl"]
    valid_dict["atomize_complex"] = valid_dict["compl"]

    # reweight fb examples containing disulfide loops
    to_reweight_ex = train_dict["fb"]["HAS_DSLF_LOOP"]
    to_reweight_cluster = train_dict["fb"][to_reweight_ex].CLUSTER.unique()
    reweight_mask = np.in1d(train_ID_dict["fb"], to_reweight_cluster)
    weights_dict["fb"][reweight_mask] *= dataset_params["dslf_fb_upsample"]

    # set number of validation examples being used
    for k in valid_dict:
        if dataset_params["n_valid_" + k] is None:
            dataset_params["n_valid_" + k] = len(valid_dict[k])

    loader_dict = dict(
        pdb=spoofed_loader,
        peptide=loader_pdb,
        compl=spoofed_loader,
        neg_compl=loader_complex,
        na_compl=loader_na_complex,
        neg_na_compl=loader_na_complex,
        distil_tf=loader_distil_tf,
        tf=loader_tf_complex,
        neg_tf=loader_tf_complex,
        fb=loader_fb,
        rna=loader_dna_rna,
        dna=loader_dna_rna,
        sm_compl=loader_sm_compl_assembly_single,
        metal_compl=loader_sm_compl_assembly_single,
        sm_compl_multi=loader_sm_compl_assembly_single,
        sm_compl_covale=loader_sm_compl_assembly_single,
        sm_compl_asmb=loader_sm_compl_assembly,
        sm=loader_sm,
        atomize_pdb=loader_atomize_pdb,
        atomize_complex=loader_atomize_complex,
    )

    train_set = DistilledDataset(
        train_ID_dict,
        train_dict,
        loader_dict,
        homo,
        chid2hash,
        chid2taxid,
        chid2smpartners,
        loader_params,
        native_NA_frac=0.25,
        p_homo_cut=dataset_params["p_homo_cut"],
        p_short_crop=dataset_params["p_short_crop"],
        p_dslf_crop=dataset_params["p_dslf_crop"],
    )
    return (
        train_set,
        train_ID_dict,
        valid_ID_dict,
        weights_dict,
        train_dict,
        valid_dict,
        homo,
        chid2hash,
        chid2taxid,
        chid2smpartners,
    )


def compose_dataset(loader_fn, dataset_params, loader_params, rank, world_size):
    # define dataset & data loader
    # this function overrides the default dataloader params with those in the config
    # TODO: cache this in checkpoints so checkpoints use the same dataloder params as training
    loader_params = set_data_loader_params(loader_params=loader_params)

    (
        train_set,
        train_ID_dict,
        valid_ID_dict,
        weights_dict,
        train_dict,
        valid_dict,
        homo,
        chid2hash,
        chid2taxid,
        chid2smpartners,
    ) = get_distilled_dataset(dataset_params, loader_params)

    sampler_class = sampler_factory[
        loader_params.get("sampler_class", "DistributedWeightedSampler")
    ]
    train_sampler = sampler_class(
        dataset=train_set,
        weights_dict=weights_dict,
        num_example_per_epoch=dataset_params["n_train"],
        fractions=OrderedDict(
            [(k, dataset_params["fraction_" + k]) for k in train_dict]
        ),
        num_replicas=world_size,
        rank=rank,
        lengths=loader_params["EXAMPLE_LENGTHS"],
        batch_by_dataset=loader_params["BATCH_BY_DATASET"],
        batch_by_length=loader_params["BATCH_BY_LENGTH"],
    )
    train_loader = data.DataLoader(
        train_set,
        sampler=train_sampler,
        batch_size=1,
        worker_init_fn=loader_fn,
        **loader_params["dataloader_kwargs"],
    )

    valid_sets = dict(
        atomize_pdb=Dataset(
            valid_ID_dict["atomize_pdb"][: dataset_params["n_valid_atomize_pdb"]],
            loader_atomize_pdb,
            valid_dict["atomize_pdb"],
            loader_params,
            homo,
            p_homo_cut=-1.0,
            n_res_atomize=9,
            flank=0,
            p_short_crop=-1.0,
        ),
        atomize_complex=Dataset(
            valid_ID_dict["atomize_complex"][
                : dataset_params["n_valid_atomize_complex"]
            ],
            loader_atomize_complex,
            valid_dict["atomize_complex"],
            loader_params,
            homo,
            p_homo_cut=-1.0,
            n_res_atomize=9,
            flank=0,
            p_short_crop=-1.0,
        ),
        pdb=Dataset(
            valid_ID_dict["pdb"][: dataset_params["n_valid_pdb"]],
            spoofed_loader,
            valid_dict["pdb"],
            loader_params,
            homo,
            p_homo_cut=-1.0,
            p_short_crop=-1.0,
            p_dslf_crop=-1.0,
            chid2hash=chid2hash,
            chid2taxid=chid2taxid,
        ),
        dslf=Dataset(
            valid_ID_dict["dslf"][: dataset_params["n_valid_dslf"]],
            loader_pdb,
            valid_dict["dslf"],
            loader_params,
            homo,
            p_homo_cut=-1.0,
            p_short_crop=-1.0,
            p_dslf_crop=1.0,
        ),
        homo=Dataset(
            valid_ID_dict["homo"][: dataset_params["n_valid_homo"]],
            loader_pdb,
            valid_dict["homo"],
            loader_params,
            homo,
            p_homo_cut=1.0,
            p_short_crop=-1.0,
            p_dslf_crop=-1.0,
        ),
        rna=DatasetRNA(
            valid_ID_dict["rna"][: dataset_params["n_valid_rna"]],
            loader_dna_rna,
            valid_dict["rna"],
            loader_params,
        ),
        dna=DatasetRNA(
            valid_ID_dict["dna"][: dataset_params["n_valid_dna"]],
            loader_dna_rna,
            valid_dict["dna"],
            loader_params,
        ),
        distil_tf=DatasetNAComplex(
            valid_ID_dict["distil_tf"][: dataset_params["n_valid_distil_tf"]],
            loader_distil_tf,
            valid_dict["distil_tf"],
            loader_params,
            negative=False,
            native_NA_frac=0.0,
        ),
        metal_compl=DatasetSMComplexAssembly(
            valid_ID_dict["metal_compl"][: dataset_params["n_valid_metal_compl"]],
            loader_sm_compl_assembly,
            valid_dict["metal_compl"],
            chid2hash,
            chid2taxid,  # used for MSA generation of assemblies
            loader_params,
            task="metal_compl",
            num_protein_chains=1,
        ),
        sm_compl=DatasetSMComplexAssembly(
            valid_ID_dict["sm_compl"][: dataset_params["n_valid_sm_compl"]],
            loader_sm_compl_assembly,
            valid_dict["sm_compl"],
            chid2hash,
            chid2taxid,  # used for MSA generation of assemblies
            loader_params,
            task="sm_compl",
            num_protein_chains=1,
        ),
        sm_compl_multi=DatasetSMComplexAssembly(
            valid_ID_dict["sm_compl_multi"][: dataset_params["n_valid_sm_compl_multi"]],
            loader_sm_compl_assembly,
            valid_dict["sm_compl_multi"],
            chid2hash,
            chid2taxid,  # used for MSA generation of assemblies
            loader_params,
            task="sm_compl_multi",
            num_protein_chains=1,
        ),
        sm_compl_covale=DatasetSMComplexAssembly(
            valid_ID_dict["sm_compl_covale"][
                : dataset_params["n_valid_sm_compl_covale"]
            ],
            loader_sm_compl_assembly,
            valid_dict["sm_compl_covale"],
            chid2hash,
            chid2taxid,  # used for MSA generation of assemblies
            loader_params,
            task="sm_compl_covale",
            num_protein_chains=1,
        ),
        sm_compl_strict=DatasetSMComplexAssembly(
            valid_ID_dict["sm_compl_strict"][
                : dataset_params["n_valid_sm_compl_strict"]
            ],
            loader_sm_compl_assembly,
            valid_dict["sm_compl_strict"],
            chid2hash,
            chid2taxid,  # used for MSA generation of assemblies
            loader_params,
            task="sm_compl_strict",
            num_protein_chains=1,
        ),
        sm_compl_asmb=DatasetSMComplexAssembly(
            valid_ID_dict["sm_compl_asmb"][: dataset_params["n_valid_sm_compl_asmb"]],
            loader_sm_compl_assembly,
            valid_dict["sm_compl_asmb"],
            chid2hash,
            chid2taxid,  # used for MSA generation of assemblies
            loader_params,
            task="sm_compl_asmb",
        ),
        sm=DatasetSM(
            valid_ID_dict["sm"][: dataset_params["n_valid_sm"]],
            loader_sm,
            valid_dict["sm"],
            loader_params,
        ),
    )

    valid_headers = dict(
        distil_tf="TF_Distil",
        pdb="Monomer",
        dslf="Disulfide_loop",
        homo="Homo",
        rna="RNA",
        dna="DNA",
        sm_compl="SM_Compl",
        metal_compl="Metal_ion",
        sm_compl_multi="Multires_ligand",
        sm_compl_covale="Covalent_ligand",
        sm_compl_strict="SM_Compl_(strict)",
        sm="SM_CSD",
        atomize_pdb="Monomer_atomize",
        atomize_complex="Complex_atomize",
        sm_compl_asmb="SMCompl_Assembly",
    )
    valid_samplers = OrderedDict(
        [
            (
                k,
                data.distributed.DistributedSampler(
                    v, num_replicas=world_size, rank=rank
                ),
            )
            for k, v in valid_sets.items()
        ]
    )
    valid_loaders = OrderedDict(
        [
            (
                k,
                data.DataLoader(
                    v,
                    sampler=valid_samplers[k],
                    worker_init_fn=loader_fn,
                    **loader_params["dataloader_kwargs"],
                ),
            )
            for k, v in valid_sets.items()
        ]
    )
    return train_loader, train_sampler, valid_loaders, valid_samplers


def compose_posebusters(loader_fn, loader_params, rank, world_size):
    loader_params = set_data_loader_params(loader_params=loader_params)
    valid_ID_dict, valid_dict = {}, {}

    valid_dict["benchmark"] = _load_df(
        posebusters_csv, pad_hash=False, eval_cols=["LIGAND", "PARTNERS", "LIGXF"]
    )
    valid_ID_dict["benchmark"] = valid_dict["benchmark"]["CLUSTER"]

    with open(
        "/projects/ml/RF2_allatom/posebusters/posebusters_chid2hash_081723.pkl", "rb"
    ) as f:
        chid2hash = pickle.load(f)

    with open(
        "/projects/ml/RF2_allatom/posebusters/posebusters_chid2taxid_081723.pkl", "rb"
    ) as f:
        chid2taxid = pickle.load(f)
    loader_params["MINTPLT"] = 0
    loader_params["MAXTPLT"] = 0
    loader_params["PDB_DIR"] = "/projects/ml/RF2_allatom/benchmark"

    benchmark = DatasetSMComplexAssembly(
        valid_ID_dict["benchmark"],
        loader_sm_compl_assembly,
        valid_dict["benchmark"],
        chid2hash,
        chid2taxid,  # used for MSA generation of assemblies
        loader_params,
        task="sm_compl",
        num_protein_chains=1,
        num_ligand_chains=999,  # all ligands
    )
    sampler = data.distributed.DistributedSampler(
        benchmark, rank=rank, num_replicas=world_size
    )
    loader = data.DataLoader(
        benchmark,
        sampler=sampler,
        worker_init_fn=loader_fn,
        **loader_params["dataloader_kwargs"],
    )
    return loader


def compose_single_item_dataset(loader_fn, item, loader_params, loader, loader_kwargs):
    class SpoofDataset(data.Dataset):
        def __init__(self, loader_params, loader, loader_kwargs) -> None:
            super().__init__()
            self.loader_params = loader_params
            self.loader = loader
            self.loader_kwargs = loader_kwargs

        def __getitem__(self, idx):
            return self.loader(item, self.loader_params, **self.loader_kwargs)

        def __len__(self):
            return 1

    dataset = SpoofDataset(loader_params, loader, loader_kwargs)
    if False:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = data.DataLoader(
        dataset,
        # sampler=sampler,
        worker_init_fn=loader_fn,
        **loader_params["dataloader_kwargs"],
    )
    return loader


def compose_similar_posebusters(loader_params, rank, world_size):
    loader_params = set_data_loader_params(loader_params=loader_params)
    valid_ID_dict, valid_dict = {}, {}

    valid_dict["benchmark"] = _load_df(
        posebusters_csv, pad_hash=False, eval_cols=["LIGAND", "PARTNERS", "LIGXF"]
    )
    valid_ID_dict["benchmark"] = valid_dict["benchmark"]["CLUSTER"]

    with open(
        "/projects/ml/RF2_allatom/posebusters/posebusters_chid2hash_081723.pkl", "rb"
    ) as f:
        chid2hash = pickle.load(f)

    with open(
        "/projects/ml/RF2_allatom/posebusters/posebusters_chid2taxid_081723.pkl", "rb"
    ) as f:
        chid2taxid = pickle.load(f)
    loader_params["MINTPLT"] = 1
    loader_params["MAXTPLT"] = 4
    loader_params["PDB_DIR"] = "/projects/ml/RF2_allatom/benchmark"

    benchmark = DatasetSMComplexAssembly(
        valid_ID_dict["benchmark"],
        loader_sm_compl_assembly,
        valid_dict["benchmark"],
        chid2hash,
        chid2taxid,  # used for MSA generation of assemblies
        loader_params,
        task="sm_compl",
        num_protein_chains=1,
        num_ligand_chains=2,
    )
    sampler = data.distributed.DistributedSampler(
        benchmark, rank=rank, num_replicas=world_size
    )
    loader = data.DataLoader(
        benchmark, sampler=sampler, **loader_params["dataloader_kwargs"]
    )
    return loader


class LegacyDatasetConstructor:
    def __call__(self, init_db, dataset_params, loader_params, rank, world_size):
        return compose_dataset(
            loader_fn=init_db,
            dataset_params=dataset_params,
            loader_params=loader_params,
            rank=rank,
            world_size=world_size,
        )
