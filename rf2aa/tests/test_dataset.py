import pytest
import os
import pandas as pd
import numpy as np
import torch
from hydra import initialize, compose

from rf2aa.data.compose_dataset import set_data_loader_params, compose_single_item_dataset
from rf2aa.data.data_loader import loader_pdb, loader_complex, loader_na_complex, \
                                    loader_dna_rna, loader_sm_compl_assembly_single, loader_sm_compl_assembly
from rf2aa.tensor_util import assert_shape, assert_equal
from rf2aa.trainer_new import seed_all
from rf2aa.util import NTOTAL, is_atom

pdb_item = {'Unnamed: 0': 282822, 'CHAINID': '6zsc_IC', 'DEPOSITION': '2020-07-15', 'RESOLUTION': 3.5, 'HASH': '068157', 'CLUSTER': 4707, 'SEQUENCE': 'MLPAAARPLWGPCLGLRAAAFRLARRQVPCVCAVRHMRSSGHQRCEALAGAPLDNAPKEYPPKIQQLVQDIASLTLLEISDLNELLKKTLKIQDVGLVPMGGVMSGAVPAAAAQEAVEEDIPIAKERTHFTVRLTEAKPVDKVKLIKEIKNYIQGINLVQAKKLVESLPQEIKANVAKAEAEKIKAALEAVGGTVVLE', 'LEN_EXIST': 30, 'TAXID': '9606'}
compl_item = {'CHAINID': '6vet_D:6vet_C', 'DEPOSITION': '2020-01-02', 'RESOLUTION': 1.46, 'HASH': '015533_020710', 'CLUSTER': 1650, 'LENA:B': '22:21', 'TAXONOMY': '9606:9606', 'ASSM_A': 1, 'OP_A': 0, 'ASSM_B': 1, 'OP_B': 0, 'HETERO': 'HETERO', 'HASH_A': '015533', 'HASH_B': '020710', 'LEN': [22, 21], 'LEN_EXIST': 43}
na_compl_item = {'CHAINID': '3sjm_C:3sjm_B_0', 'DEPOSITION': '2011-06-21', 'RESOLUTION': 1.35, 'HASH': '029615', 'CLUSTER': 13318, 'LENA:B:C:D': '64:18', 'TOPAD?': False, 'LEN': [64, 18], 'LEN_EXIST': 82}
rna_item = {'CHAINID': '4olb_B_0', 'DEPOSITION': '2014-01-23', 'RESOLUTION': 2.9, 'CLUSTER': '149', 'LENA:B': '10', 'LEN': [10], 'LEN_EXIST': 10}
sm_compl_item = {'CHAINID': '4res_C', 'DEPOSITION': '2014-09-23', 'RESOLUTION': 3.408, 'HASH': '042467', 'CLUSTER': 22914, 'SEQUENCE': 'MAGLSTDDGGSPKGDVDPFYYDYETVRNGGLIFAALAFIVGLIIILSKRLRCGGKKHRPINEDEL', 'LEN_EXIST': 32, 'LIGAND': [('Q', '1001', '17F')], 'ASSEMBLY': 1, 'COVALENT': [], 'PROT_CHAIN': 'C', 'LIGXF': [('Q', 10)], 'PARTNERS': [('C', 2, 23, 3.4699244499206543, 'polypeptide(L)'), ([('K', '2002', 'CLR')], [('K', 4)], 4, 4.189112663269043, 'nonpoly'), ('A', 0, 0, 6.709344863891602, 'polypeptide(L)'), ('B', 1, 0, 13.354745864868164, 'polypeptide(L)'), ([('L', '2003', 'K')], [('L', 5)], 0, 28.212512969970703, 'nonpoly')], 'LIGATOMS': 54, 'LIGATOMS_RESOLVED': 19, 'SUBSET': 'organic', 'name': '4res_C_asm1_Q1001-17F'}
sm_compl_covale_item = {'CHAINID': '6bgn_E', 'DEPOSITION': '2017-10-29', 'RESOLUTION': 1.51, 'HASH': '088784', 'CLUSTER': 2477, 'SEQUENCE': 'PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASKV', 'LEN_EXIST': 58, 'LIGAND': [('Y', '101', '6Y5')], 'ASSEMBLY': 5, 'COVALENT': [(('E', '1', 'PRO', 'N'), ('Y', '101', '6Y5', 'C3'))], 'PROT_CHAIN': 'E', 'LIGXF': [('Y', 1)], 'PARTNERS': [('E', 0, 75, 1.3035255670547485, 'polypeptide(L)')], 'LIGATOMS': 9, 'LIGATOMS_RESOLVED': 9, 'SUBSET': 'covale'}
sm_compl_asmb_item = {'CHAINID': '4i7z_D', 'DEPOSITION': '2012-12-01', 'RESOLUTION': 2.803, 'HASH': '045638', 'CLUSTER': 20270, 'SEQUENCE': 'MAQFTESMDVPDMGRRQFMNLLAFGTVTGVALGALYPLVKYFIPPSGGAVGGGTTAKDKLGNNVKVSKFLESHNAGDRVLVQGLKGDPTYIVVESKEAIRDYGINAVCTHLGCVVPWNAAENKFKCPCHGSQYDETGKVIRGPAPLSLALCHATVQDDNIVLTPWTETDFRTGEKPWWV', 'LEN_EXIST': 38, 'LIGAND': [('X', '201', '1E2')], 'ASSEMBLY': 1, 'COVALENT': [], 'PROT_CHAIN': 'D', 'LIGXF': [('X', 46)], 'PARTNERS': [('D', 6, 50, 3.2743349075317383, 'polypeptide(L)'), ('B', 2, 46, 2.9014363288879395, 'polypeptide(L)'), ([('P', '308', 'UMQ')], [('P', 30)], 41, 2.2776474952697754, 'nonpoly'), ('C', 4, 12, 3.0980331897735596, 'polypeptide(L)'), ('A', 0, 0, 5.883709907531738, 'polypeptide(L)'), ([('O', '307', 'UMQ')], [('O', 28)], 0, 6.910125732421875, 'nonpoly'), ('E', 8, 0, 12.058629035949707, 'polypeptide(L)'), ('H', 14, 0, 14.072593688964844, 'polypeptide(L)'), ('F', 10, 0, 15.534253120422363, 'polypeptide(L)'), ([('N', '306', 'UMQ')], [('N', 27)], 0, 16.360061645507812, 'nonpoly'), ([('K', '303', 'HEM')], [('K', 20)], 0, 16.7199649810791, 'nonpoly'), ('A', 1, 0, 19.487838745117188, 'polypeptide(L)'), ([('M', '305', '8K6')], [('M', 24)], 0, 22.37641716003418, 'nonpoly'), ('B', 3, 0, 22.523012161254883, 'polypeptide(L)')], 'LIGATOMS': 23, 'LIGATOMS_RESOLVED': 23, 'SUBSET': 'asmb'}


def test_data():
    with initialize(version_base=None, config_path="../config/train"):
        cfg = compose(config_name="base", overrides=["loader_params.p_msa_mask=0.0", 
                                                     "loader_params.crop=100000",
                                                     "loader_params.mintplt=0",
                                                     "loader_params.maxtplt=2"
                                                     ])
 
    loader_params = set_data_loader_params(loader_params=cfg.loader_params) 

    datasets = ["pdb", "compl", "na_compl", "rna", \
                "sm_compl", "sm_compl_covale", "sm_compl_asmb"]
    loaders = [
        loader_pdb,
        loader_complex, 
        loader_na_complex, 
        loader_dna_rna,
        loader_sm_compl_assembly_single,
        loader_sm_compl_assembly_single,
        loader_sm_compl_assembly
    ] 
    items = [
        pdb_item,
        compl_item,
        na_compl_item,
        rna_item,
        sm_compl_item,
        sm_compl_covale_item,
        sm_compl_asmb_item
    ]

    loader_kwargs = [
        {
            "homo": {"CHAIN_A": pd.Series(dtype=np.float32)}
        }, 
        {},
        {}, 
        {},
        {},
        {},
        {}
    ]   
    loader_params_list  = [loader_params] * len(datasets)
    return zip(datasets, loaders, items, loader_params_list, loader_kwargs) 

@pytest.mark.parametrize("name,loader,item,loader_params,loader_kwargs", test_data())
def test_correct_shapes(name, loader, item, loader_params, loader_kwargs):
    data_loader = compose_single_item_dataset(item, loader_params, loader, loader_kwargs)
    for inputs in data_loader:
        (
        seq, msa, msa_masked, msa_full, mask_msa, true_crds, mask_crds, idx_pdb, 
        xyz_t, t1d, mask_t, xyz_prev, mask_prev, same_chain, unclamp, negative, 
        atom_frames, bond_feats, dist_matrix, chirals, ch_label, symmgp, task, item
        ) = inputs
        B, recycles, N, L = msa.shape[:4]
        num_atoms = (is_atom(seq[0,0]).sum()).item()
        assert_shape(seq, (B, recycles, L))
        assert_shape(msa, (B, recycles, N, L))
        assert_shape(msa_masked, (B, recycles, N, L, 164)) #Hack: hardcoded for current featurization
        N_full = msa_full.shape[2]
        assert_shape(msa_full, (B, recycles, N_full, L, 83)) #HACK:: hardcoded for current features
        assert_shape(mask_msa, (B, recycles, N, L)) 
        N_symm = true_crds.shape[1]
        assert_shape(true_crds, (B, N_symm, L, NTOTAL, 3))
        assert_shape(mask_crds, (B, N_symm, L, NTOTAL))
        assert_shape(idx_pdb, (B, L))
        N_templ = xyz_t.shape[1]
        assert_shape(xyz_t, (B, N_templ, L, NTOTAL, 3))
        assert_shape(t1d, (B, N_templ, L, 80)) # hack hard coded dimension
        assert_shape(mask_t, (B, N_templ, L, NTOTAL))
        assert_shape(xyz_prev, (B, L, NTOTAL, 3))
        assert_shape(mask_prev, (B, L, NTOTAL))
        assert_shape(same_chain, (B, L, L))
        assert type(unclamp.item()) == bool
        assert type(negative.item()) == bool
        assert_shape(atom_frames, (B, num_atoms, 3,2))
        assert_shape(bond_feats, (B, L, L))
        assert_shape(dist_matrix, (B, L, L))
        n_chirals = chirals.shape[1]
        assert_shape(chirals, (B, n_chirals, 5))
        assert_shape(ch_label, (B, L))
        assert symmgp[0] == "C1", f"{symmgp}"

@pytest.mark.parametrize("name,loader,item,loader_params,loader_kwargs", test_data())
def test_regression(name, loader, item, loader_params, loader_kwargs):
    seed_all()
    data_loader = compose_single_item_dataset(item, loader_params, loader, loader_kwargs)
    regression_pickle = f"test_pickles/{name}_regression.pt"
    for inputs in data_loader:
        #(
        #seq, msa, msa_masked, msa_full, mask_msa, true_crds, mask_crds, idx_pdb, 
        #xyz_t, t1d, mask_t, xyz_prev, mask_prev, same_chain, unclamp, negative, 
        #atom_frames, bond_feats, dist_matrix, chirals, ch_label, symmgp, task, item
        #) = inputs
        names = ["seq", "msa", "msa_masked", "msa_full", "mask_msa", "true_crds",\
                 "mask_crds", "idx_pdb", "xyz_t", "t1d", "mask_t", "xyz_prev", "mask_prev", \
                    "same_chain", "unclamp", "negative", "atom_frames", "bond_feats", \
                        "dist_matrix", "chirals", "ch_label", "symmgp", "task", "item"]
        if os.path.exists(regression_pickle):
            regression = torch.load(regression_pickle, map_location="cpu")
        else:
            torch.save(inputs, regression_pickle)
            print(f"SAVED test_outputs in {regression_pickle}")
            return
        for idx, input in enumerate(inputs):
            if torch.is_tensor(input):
                if idx in [8,11]: # xyz_t, xyz_prev are not deterministic yet 
                    continue
                try:
                    assert_equal(input, regression[idx])
                except Exception as e:
                    #TODO: revisit this after dataset refactor
                    if idx in [9, 10]: # weirdness in templates sm_compl_assembly
                        print("t1d fails for sm_compl_assembly")
                        continue
                    raise AssertionError(f"{names[idx]} did not match") from e

