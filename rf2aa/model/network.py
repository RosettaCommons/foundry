import hydra
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from rf2aa.debug import debug_nans
from rf2aa.model.embedding_blocks import embedding_factory
from rf2aa.model.refinement_blocks import refinement_factory
from rf2aa.model.simulator_blocks import block_factory
from rf2aa.model.layers.AuxiliaryPredictor import aux_predictor_factory
from rf2aa.training.checkpoint import create_custom_forward
from rf2aa.util import is_atom


class RosettaFold(nn.Module):
    """ creates an instance of RosettaFold which includes an embedder, trunk, refinement layers and aux predictor"""
    def __init__(self, config):
        super(RosettaFold, self).__init__()
        model_params = config.model
         
        assert len(model_params.embedding.keys()) == 1, "only can have one embedder"
        embedding_type = next(iter(model_params.embedding.keys()))
        self.embedding = embedding_factory[embedding_type](model_params["global_params"], model_params.embedding[embedding_type]["params"])

        ## instantiate blocks of network
        blocks = []
        for block in model_params.blocks.keys():
            if block not in block_factory:
                raise ValueError(f"User specified {block} type, but this block is not registered in rf2aa.Trunk_blocks.")
            blocks_to_add  = [block_factory[block](
                global_config=model_params["global_params"], 
                block_params=model_params.blocks[block]["params"]) 
                              for _ in range(model_params.blocks[block]["num_blocks"])]
            blocks.extend(blocks_to_add) 
        
        self.simulator = nn.ModuleList(blocks)
        
        assert len(model_params.refinement.keys()) == 1, "only can have one refinment block"
        refinement_type = next(iter(model_params.refinement.keys()))

        self.refinement = refinement_factory[refinement_type](
            model_params["global_params"], 
            model_params.refinement[refinement_type]["params"]
        )

        aux_tasks = {}
        for aux_task in model_params.auxiliary_predictors.keys():
            aux_tasks.update({
                aux_task:
                aux_predictor_factory[aux_task](
                    model_params.auxiliary_predictors[aux_task]["n_feat"])
            }
            ) #HACK: eventually this will just use the correct n_feat from the global config
        self.auxiliary_predictors = nn.ModuleDict(aux_tasks)
        self.auxiliary_predictor_input_feats = {
            aux_task:model_params.auxiliary_predictors[aux_task]["input_feature"] \
            for aux_task in model_params.auxiliary_predictors.keys()
        }

    def forward(self, rf_inputs, use_checkpoint, use_amp):
        latent_feats = self.embedding(rf_inputs)
        #load useful primitives into latent_features
        latent_feats.update(
            {
                "is_atom": is_atom(rf_inputs["seq_unmasked"]),
                "atom_frames": rf_inputs["atom_frames"],
                "chirals": rf_inputs["chirals"],
                "xyz": rf_inputs["xyz"],
                "idx": rf_inputs["idx"],
                "bond_feats": rf_inputs["bond_feats"],
                "dist_matrix": rf_inputs["dist_matrix"],
            }
        )
        for block in self.simulator:
            latent_feats = block(latent_feats, use_checkpoint, use_amp)

        rf_outputs = self.refinement(latent_feats)

        for aux_task, aux_predictor in self.auxiliary_predictors.items():
            input_feature = self.auxiliary_predictor_input_feats[aux_task]
            auxiliary_predictions = aux_predictor(latent_feats[input_feature])
            rf_outputs.update({aux_task: auxiliary_predictions})


        return rf_outputs, latent_feats


@hydra.main(version_base=None, config_path='../config/train', config_name='base')
def main(config):
    model = RosettaFold(config)
    import pdb; pdb.set_trace()


if __name__ =="__main__":
    main()