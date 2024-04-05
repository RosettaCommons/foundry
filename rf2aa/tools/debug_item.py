import unittest
from hydra import compose, initialize
import torch
from rf2aa.chemical import ChemicalData as ChemData 

from rf2aa.data.compose_dataset import compose_single_item_dataset, set_data_loader_params
from rf2aa.data.data_loader import loader_atomize_pdb, loader_sm_compl_assembly
from rf2aa.data.dataloader_adaptor import prepare_input
from rf2aa.util import is_atom, writepdb
from rf2aa.tensor_util import assert_shape
from rf2aa.trainer_new import trainer_factory
from rf2aa.training.recycling import recycle_step_legacy

from icecream import ic 
import tensor_util
import pdb 
#### Setup test case hyperparams

#ITEM = \
#{'Unnamed: 0': 262672, 'CHAINID': '6ywe_UB', 'DEPOSITION': '2020-04-29', 'RESOLUTION': 2.9900, 'HASH': '072380', 'CLUSTER': 9905, 'SEQUENCE': 'MPNKPIRLPPLKQLRVRQANKAEENPCIAVMSSVLACWASAGYNSAGCATVENALRACMDAPKPAPKPNNTINYHLSRFQERLTQGKSKK', 'LEN_EXIST': 88, 'TAXID': '5141'}
ITEM = {'CHAINID': '3p55_A', 'DEPOSITION': '2010-10-07', 'RESOLUTION': 2.0, 'HASH': '078142', 'CLUSTER': 8667, 'SEQUENCE': 'MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGHAFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFGKAVQQPDGLAVLGIFLKVGSAKPGLQKVVDVLDSIKTKGKSADFTNFDPRGLLPESLDYWTYPGSLTTPPLLECVTWIVLKEPISVSSEQVLKFRKLNFNGEGEPEELMVDNWRPAQPLKNRQIKASFK', 'LEN_EXIST': 257, 'LIGAND': [('B', '300', '670')], 'ASSEMBLY': 1, 'COVALENT': '[]', 'PROT_CHAIN': 'A', 'LIGXF': [('B', 1)], 'PARTNERS': [('A', 0, 194, 2.612750291824341, 'polypeptide(L)'), ([('E', '261', 'ZN')], [('E', 4)], 6, 2.0762431621551514, 'nonpoly')], 'LIGATOMS': 26, 'LIGATOMS_RESOLVED': 26, 'SUBSET': 'organic'}

class DotDict:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, DotDict(value))
            else:
                setattr(self, key, value)


# DJ -  make fake params for Chemdata init, 
#       then future inits elsewhere will be good 
cd_params = DotDict({'use_phospate_frames_for_NA':False})
ChemData(cd_params)

CONFIG = "legacy_train"
LOADER_FN = loader_sm_compl_assembly
LOADER_KWARGS = {
            #"homo": None,
            #"n_res_atomize": 5,
            #"flank": 0
        }


# set device to your liking
DEVICE = 'cuda'

def transfer_tensors_to_device(obj_list, device):
        """
        Transfer all torch.Tensor objects within a list to a specified device.

        Args:
        - obj_list (list): List of objects possibly containing torch.Tensor objects.
        - device (torch.device): Device to transfer tensors to.

        Returns:
        - list: List of objects with tensors transferred to the specified device.
        """
        for i in range(len(obj_list)):
            if isinstance(obj_list[i], torch.Tensor):
                obj_list[i] = obj_list[i].to(device)
            elif isinstance(obj_list[i], list):
                obj_list[i] = transfer_tensors_to_device(obj_list[i], device)
            elif isinstance(obj_list[i], dict):
                for key, value in obj_list[i].items():
                    obj_list[i][key] = transfer_tensors_to_device(value, device)
        return obj_list

# class Debug():
class DebugTestCase(unittest.TestCase):
    
    def setUp(self) -> None:
        with initialize(version_base=None, config_path="config/train"):
            self.cfg = compose(config_name=CONFIG)
        loader_params = set_data_loader_params(self.cfg.loader_params)
        loader = compose_single_item_dataset(
            None,
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
        assert_shape(true_crds, (B, N_symm, L, ChemData().NTOTAL, 3))
        assert_shape(mask_crds, (B, N_symm, L, ChemData().NTOTAL))
        assert_shape(idx_pdb, (B, L))
        N_templ = xyz_t.shape[1]
        assert_shape(xyz_t, (B, N_templ, L, ChemData().NTOTAL, 3))
        assert_shape(t1d, (B, N_templ, L, 80)) # hack hard coded dimension
        assert_shape(mask_t, (B, N_templ, L, ChemData().NTOTAL))
        assert_shape(xyz_prev, (B, L, ChemData().NTOTAL, 3))
        assert_shape(mask_prev, (B, L, ChemData().NTOTAL))
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

        print('Done with test_correct_shapes ')

    def test_forward_pass(self):
        trainer = trainer_factory[self.cfg.experiment.trainer](self.cfg)
        trainer.construct_model(device=DEVICE)
        trainer.model.device = DEVICE
        trainer.move_constants_to_device(gpu=DEVICE)
        for inputs in self.loader: 
            transfer_tensors_to_device(inputs, DEVICE)

            loss, loss_dict = trainer.train_step(inputs, 1)
        print('Done with test_forward_pass')
    
    def test_forward_pass_with_checkpoint(self):
        trainer = trainer_factory[self.cfg.experiment.trainer](self.cfg)
        trainer.construct_model(device=DEVICE)
        trainer.model.device = DEVICE
        trainer.move_constants_to_device(gpu=DEVICE)
        checkpoint_path = "/home/rohith/rf2a-fd3/models/rf2a_fd3_20221125_714.pt"
        #checkpoint_path='/home/davidcj/software/clean_rf2aa/RF2-allatom/ckpts/rf2a_fd3_20221125_714.pt'
        trainer.checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        trainer.model.model.load_state_dict(trainer.checkpoint["final_state_dict"])
        trainer.model.shadow.load_state_dict(trainer.checkpoint["model_state_dict"])
        
        
        for inputs in self.loader:
            transfer_tensors_to_device(inputs, DEVICE)
            loss, loss_dict = trainer.train_step(inputs, 1)
        print('Done with test_forward_pass_with_checkpoint')
            #TODO: check something about the loss

    def test_forward_pass_outputs(self):
        trainer = trainer_factory[self.cfg.experiment.trainer](self.cfg)
        trainer.construct_model(device=DEVICE)
        trainer.model.device = DEVICE
        trainer.move_constants_to_device(gpu=DEVICE)
        #checkpoint_path = "/home/rohith/rf2a-fd3/models/rf2a_fd3_20221125_714.pt"
        checkpoint_path='/home/davidcj/software/clean_rf2aa/RF2-allatom/ckpts/rf2a_fd3_20221125_714.pt' 
        trainer.checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        trainer.model.model.load_state_dict(trainer.checkpoint["final_state_dict"])
        trainer.model.shadow.load_state_dict(trainer.checkpoint["model_state_dict"])
        for inputs in self.loader:
            gpu = trainer.model.device
            
            transfer_tensors_to_device(inputs, gpu)

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

            print('Done with test_forward_pass_outputs')


if __name__ == "__main__":
    unittest.main()
