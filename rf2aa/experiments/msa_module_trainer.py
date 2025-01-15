import torch
import torch.nn as nn
import tree
from hydra.utils import instantiate
from torch.nn.parallel import DistributedDataParallel as DDP

from rf2aa.trainer_new import ComposedTrainer

from rf2aa.model.AF3_structure import MSAModule, FeatureInitializer, DistogramHead, PairformerBlock
from rf2aa.model.layers.Embeddings import MSA_emb
from rf2aa.loss.af3_losses import DistogramLoss
from rf2aa.training.EMA import EMA


class MsaModuleTrainer(ComposedTrainer):
    def construct_model(self, device="cpu"):
        model = instantiate(self.config.model).to(device)
        model = EMA(model, decay=0.999) 
        self.model = DDP(model, device_ids=[device], find_unused_parameters=True, broadcast_buffers=False)
        self.distogram_loss = DistogramLoss(**self.config.loss.distogram_loss)

    def train_step(self, inputs, n_cycle, no_grads=False, return_outputs=False):
        gpu = self.model.device
        example = inputs[0]
        network_input = {
            #TODO: make a transform that places unresolved ground truth coordinates on their closest real atomshh
            "X_noisy_L": torch.nan_to_num(example["ground_truth"]["coord_atom_lvl"]) + example["noise"],
            "t": example["t"],
            "f": example["feats"],
        } 

        loss_input = {
            "X_gt_L": example["ground_truth"]["coord_atom_lvl"],
            "crd_mask_L": example["ground_truth"]["mask_atom_lvl"],
            "X_rep_atoms_I": example["ground_truth"]["coord_token_lvl"],
            "crd_mask_rep_atoms_I": example["ground_truth"]["mask_token_lvl"],
        }
        network_input=tree.map_structure(lambda x: x.to(gpu) if hasattr(x, 'cpu') else x, network_input)
        loss_input = tree.map_structure(lambda x: x.to(gpu) if hasattr(x, 'cpu') else x, loss_input)
        network_output = self.model(**network_input)
        loss, distogram_loss_dict = self.distogram_loss(
                                             network_input,
                                             network_output,
                                            loss_input,
                                             )
        
        return loss, distogram_loss_dict

class MsaModuleRFAATrainer(MsaModuleTrainer):
    def construct_model(self, device="cpu"):
        model = MSAModuleRFAAEmbedding(**self.config.model).to(device)
        model = EMA(model, decay=0.999) 
        self.model = DDP(model, device_ids=[device], find_unused_parameters=True, broadcast_buffers=False)
        self.distogram_loss = DistogramLoss(**self.config.loss.distogram_loss)

class MsaModulewithDist(nn.Module):

    def __init__(self,
                c_s,
                c_z,
                c_atom,
                c_atompair,
                c_s_inputs,
                feature_initializer, 
                msa_module, 
                n_pairformer_blocks,
                pairformer_block,
                distogram_head,
                **kwargs
                ):
        super().__init__()
        self.feature_initializer = FeatureInitializer( 
                c_s,
                c_z,
                c_atom,
                c_atompair,
                **feature_initializer)
        self.msa_module = MSAModule(**msa_module)
        self.pairformer_stack = nn.ModuleList([
            PairformerBlock(c_s=c_s, c_z=c_z, **pairformer_block) for _ in range(n_pairformer_blocks)
        ])
        self.dist_head = DistogramHead(c_z, **distogram_head)

    def forward(self, **network_input):
        S_inputs_I, S_init_I, Z_init_II = self.feature_initializer(network_input["f"])
        S_I, Z_II = S_init_I, Z_init_II
        Z_II = self.msa_module(network_input["f"], Z_II, S_inputs_I)
        for i in range(len(self.pairformer_stack)):
            S_I, Z_II = self.pairformer_stack[i](S_I, Z_II)
    
        dist_logits = self.dist_head(Z_II)
        return dict(
            distogram=dist_logits,
        )

class MSAModuleRFAAEmbedding(nn.Module):

    def __init__(self, 
        c_s,
        c_z,
        c_atom,
        c_atompair,
        c_s_inputs,
        feature_initializer,
        msa_module,
        distogram_head,
        **kwargs
        ):
        super(MSAModuleRFAAEmbedding, self).__init__()
        self.embedding = MSA_emb(
            d_msa=c_s,
            d_pair=c_z,
            d_state=c_s_inputs,
            d_init=32
        )
        self.msa_module = MSAModule(**msa_module)
        self.dist_head = DistogramHead(c_z, **distogram_head)
    
    def forward(self, **network_input):
        import pdb; pdb.set_trace()
        f = network_input["f"]
        msa = f["msa"]
        bond_feats = f["bond_feats"]
        seq = msa[0].argmax(-1)

        pass

class MSAModuleRFAALoss(nn.Module):
    pass

def compute_neff(msa):
    # pairwise identity between all sequences in the MSA
    pass

class OldDistogramLoss(nn.Module):

    def __init__(self, weight=1.0):
        super(DistogramLoss, self).__init__()
        self.weight = weight
        self.cce_loss = nn.CrossEntropyLoss(reduction="none")
        self.eps = 1e-8
    
    def forward(
        self,
        distogram_pred,
        X_gt_I,
        crd_mask_I,
        seq,
    ):
        pass
