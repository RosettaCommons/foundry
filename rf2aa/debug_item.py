import unittest
from hydra import compose, initialize
import torch
from rf2aa.chemical import NBTYPES, NTOTAL

from rf2aa.data.compose_dataset import compose_single_item_dataset, set_data_loader_params
from rf2aa.data.data_loader import loader_atomize_pdb
from rf2aa.data.dataloader_adaptor import prepare_input
from rf2aa.util import is_atom, writepdb
from rf2aa.tensor_util import assert_shape
from rf2aa.trainer_new import trainer_factory
from rf2aa.training.recycling import recycle_step_legacy



#### Setup test case hyperparams

ITEM = \
{'Unnamed: 0': 262672, 'CHAINID': '6ywe_UB', 'DEPOSITION': '2020-04-29', 'RESOLUTION': 2.9900, 'HASH': '072380', 'CLUSTER': 9905, 'SEQUENCE': 'MPNKPIRLPPLKQLRVRQANKAEENPCIAVMSSVLACWASAGYNSAGCATVENALRACMDAPKPAPKPNNTINYHLSRFQERLTQGKSKK', 'LEN_EXIST': 88, 'TAXID': '5141'}

CONFIG = "legacy_train"
LOADER_FN = loader_atomize_pdb
LOADER_KWARGS = {
            "homo": None,
            "n_res_atomize": 5,
            "flank": 0
        }

class DebugTestCase(unittest.TestCase):
    
    def setUp(self) -> None:
        with initialize(version_base=None, config_path="config/train"):
            self.cfg = compose(config_name=CONFIG)
        loader_params = set_data_loader_params(self.cfg.loader_params)
        loader = compose_single_item_dataset(
            ITEM, 
            loader_params, 
            LOADER_FN,
            LOADER_KWARGS
        )
        self.loader = loader

    def test_correct_shapes(self):
        """ test shapes are all consistent with each other """
        for inputs in self.loader:
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

    def test_forward_pass(self):
        trainer = trainer_factory[self.cfg.experiment.trainer](self.cfg)
        trainer.construct_model()
        trainer.model.device = "cpu"
        trainer.move_constants_to_device(gpu="cpu")
        for inputs in self.loader:
            loss, loss_dict = trainer.train_step(inputs, 1)
    
    def test_forward_pass_with_checkpoint(self):
        trainer = trainer_factory[self.cfg.experiment.trainer](self.cfg)
        trainer.construct_model()
        trainer.model.device = "cpu"
        trainer.move_constants_to_device(gpu="cpu")
        checkpoint_path = "/home/rohith/rf2a-fd3/models/rf2a_fd3_20221125_714.pt"
        trainer.checkpoint = torch.load(checkpoint_path, map_location="cpu")
        trainer.model.model.load_state_dict(trainer.checkpoint["final_state_dict"])
        trainer.model.shadow.load_state_dict(trainer.checkpoint["model_state_dict"])
        for inputs in self.loader:
            loss, loss_dict = trainer.train_step(inputs, 1)
            #TODO: check something about the loss

    def test_forward_pass_outputs(self):
        trainer = trainer_factory[self.cfg.experiment.trainer](self.cfg)
        trainer.construct_model()
        trainer.model.device = "cpu"
        trainer.move_constants_to_device(gpu="cpu")
        checkpoint_path = "/home/rohith/rf2a-fd3/models/rf2a_fd3_20221125_714.pt"
        trainer.checkpoint = torch.load(checkpoint_path, map_location="cpu")
        trainer.model.model.load_state_dict(trainer.checkpoint["final_state_dict"])
        trainer.model.shadow.load_state_dict(trainer.checkpoint["model_state_dict"])
        for inputs in self.loader:
            gpu = trainer.model.device
            # HACK: certain features are constructed during the train step
            # in the future this should only promote the constructed features onto gpu
            task, item, network_input, true_crds, \
                atom_mask, msa, mask_msa, unclamp, negative, symmRs, Lasu, ch_label \
                = prepare_input(inputs, trainer.xyz_converter, gpu)
            n_cycle = 1
            output_i = recycle_step_legacy(trainer.model, network_input, n_cycle, trainer.config.training_params.use_amp) 
            c6d, mlm, pae, pde, p_bind, xyz, alphas, _, _, _, _, _ = output_i
            seq_unmasked = network_input["seq_unmasked"]
            writepdb("test.pdb", xyz[-1], seq_unmasked)


if __name__ == "__main__":
    unittest.main()
