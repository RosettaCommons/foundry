import re
import torch
import logging
import tree
import time

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from rf2aa.trainer_new import FlowMatchingTrainer
from rf2aa.model import AF3_structure
from rf2aa.data.compose_data_datahub_new import NewDatapipeTrainer

from rf2aa.training.EMA import EMA, count_parameters
from rf2aa.flow_matching.sampler import AF3Sampler, AF3PartialSampler
from rf2aa.loss.af3_losses import Loss as AF3Loss
from rf2aa.loss.af3_losses import SubunitSymmetryResolution, ResidueSymmetryResolution
from rf2aa.metrics.metrics_base import MetricManager
from rf2aa.metrics.predicted_error import GetConfidenceIndices
from rf2aa.metrics.metric_utils import unbin_logits
from rf2aa.debug import pretty_describe_dict

from rf2aa.chemical import ChemicalData as ChemData
from functools import partial
from rf2aa.chemical import initialize_chemdata
from icecream import ic

import numpy as np

logger = logging.getLogger(__name__)

class AF3Trainer(FlowMatchingTrainer):

    def construct_model(self, device="cpu", inference=False):
        self.model = AF3_structure.Model(**self.config.model).to(device)

        if self.config.training_params.EMA is not None:
            self.model = EMA(self.model, self.config.training_params.EMA)

        if inference is False:
            self.model = DDP(self.model, device_ids=None, find_unused_parameters=False, broadcast_buffers=False)
        else:
            from rf2aa.training.EMA import FakeDDPWrapper
            self.model = FakeDDPWrapper(self.model)
        if "partial_t" in self.config.af3_data_prep:
            self.sampler = AF3PartialSampler(self.config, self.model.module.shadow)
        else:
            self.sampler = AF3Sampler(self.config, self.model.module.shadow)
        self.loss = AF3Loss(**self.config.loss)
        self.subunit_symm_resolve = SubunitSymmetryResolution()
        self.residue_symm_resolve = ResidueSymmetryResolution()
        self.metrics = MetricManager(**self.config.metrics)

    def train_step(self, inputs, n_cycle, no_grads=False, return_outputs=False):
        gpu = self.model.device

        example = inputs[0]
        network_input = {
            #TODO: make a transform that places unresolved ground truth coordinates on their closest real atomshh
            "X_noisy_L": torch.nan_to_num(example["coord_atom_lvl_to_be_noised"]) + example["noise"],
            "t": example["t"],
            "f": example["feats"],
        } 
        del network_input["f"]["ref_automorphs"]
        del network_input["f"]["ref_automorphs_mask"]

        loss_input = {
            "X_gt_L": example["ground_truth"]["coord_atom_lvl"][None].expand(self.config.dataset_params.diffusion_batch_size, -1, -1),
            "crd_mask_L": example["ground_truth"]["mask_atom_lvl"][None].expand(self.config.dataset_params.diffusion_batch_size, -1),
            "X_rep_atoms_I": example["ground_truth"]["coord_token_lvl"],
            "crd_mask_rep_atoms_I": example["ground_truth"]["mask_token_lvl"],
        }

        def _inmap(path, x):
            if hasattr(x, 'cpu') and path != ('f','msa_stack'):
                return x.to(gpu) 
            else:
                return x
        network_input = tree.map_structure_with_path(_inmap, network_input)

        loss_input = tree.map_structure(lambda x: x.to(gpu) if hasattr(x, 'cpu') else x, loss_input)
        logger.debug('network_input:\n' + pretty_describe_dict(network_input))
        logger.debug('loss_input:\n' + pretty_describe_dict(loss_input))

        network_input["f"]["msa_stack"] = network_input["f"]["msa_stack"].to(torch.bfloat16)
        network_input["f"]["profile"] = network_input["f"]["profile"].to(torch.bfloat16)

        for x in ['template_distogram','template_restype','template_unit_vector']:
            network_input["f"][x] = network_input["f"][x].to(torch.bfloat16)

        output_i = self.model(
            network_input,
            n_cycle,
            no_sync=self.model.no_sync,
            use_amp=self.config.training_params.use_amp
        )

        # uncomment for symmetry resolution in training
        #loss_input = self.subunit_symm_resolve(output_i, loss_input, example["symmetry_resolution"])
        #loss_input = self.residue_symm_resolve(output_i, loss_input, example["automorphisms"])

        loss, loss_dict_batched = self.loss(network_input, output_i, loss_input)
        loss_dict = self.unbatch_losses(loss_dict_batched)

        return loss, loss_dict

    def valid_step(self, inputs, n_cycle, no_grads=True, return_outputs=False):
        gpu = self.model.device
        with torch.no_grad():
            outputs = self.sampler.sample(inputs, n_cycle=n_cycle, use_amp=self.config.training_params.use_amp)
        example = inputs[0]
        network_input = {
            #TODO: make a transform that places unresolved ground truth coordinates on their closest real atomshh
            "X_noisy_L": torch.nan_to_num(example["coord_atom_lvl_to_be_noised"]) + example["noise"],
            "t": example["t"],
            "f": example["feats"],
        } 
        loss_input = {
            "X_gt_L": example["ground_truth"]["coord_atom_lvl"][None].expand(self.config.dataset_params.diffusion_batch_size, -1,-1),
            "crd_mask_L": example["ground_truth"]["mask_atom_lvl"][None].expand(self.config.dataset_params.diffusion_batch_size, -1),
            "X_rep_atoms_I": example["ground_truth"]["coord_token_lvl"],
            "crd_mask_rep_atoms_I": example["ground_truth"]["mask_token_lvl"],
            "interfaces_to_score": example["ground_truth"]["interfaces_to_score"],
            "pn_units_to_score": example["ground_truth"]["pn_units_to_score"],
            "chain_iid_token_lvl": example["ground_truth"]["chain_iid_token_lvl"],
            "example_id": example["example_id"],
        }

        def _inmap(path, x):
            if hasattr(x, 'cpu') and path != ('f','msa_stack'):
                return x.to(gpu) 
            else:
                return x

        network_input = tree.map_structure_with_path(_inmap, network_input)
        loss_input = tree.map_structure(lambda x: x.to(gpu) if hasattr(x, 'cpu') else x, loss_input)

        # symmetry resolution
        loss_input = self.subunit_symm_resolve(outputs, loss_input, example["symmetry_resolution"])
        loss_input = self.residue_symm_resolve(outputs, loss_input, example["automorphisms"])

        metrics_dict = self.metrics(network_input, outputs, loss_input)

        if self.config.eval_params.dump_cif_files or self.config.eval_params.dump_cif_trajectories:
            print (example["example_id"], metrics_dict)

        if self.config.eval_params.dump_cif_files:
            self.write_pdb(example,outputs)

        if self.config.eval_params.dump_cif_trajectories:
            self.write_pdb(example,outputs, dump_traj=True)

        return torch.tensor(0), metrics_dict

    def valid_epoch(self, epoch, rank, world_size):
        # turn off gradients
        self.model.eval()
        valid_loss_dict = []

        for dataset_name, valid_loader in self.valid_loaders.items():
            for valid_idx, inputs in enumerate(valid_loader):
                n_cycle = inputs[0]["feats"]["msa_stack"].shape[0] # use size of MSA stack to set ncycles
                _, loss_dict = self.valid_step(inputs, n_cycle)
                valid_loss_dict.append(loss_dict)

        # synchronize
        all_valid_loss_dict = [None for i in range(world_size)]
        dist.all_gather_object(all_valid_loss_dict, valid_loss_dict)
        all_valid_loss_dict = sum(all_valid_loss_dict, []) # flatten

        if rank==0:
            self.log_validation_losses(epoch, all_valid_loss_dict, self.config.experiment.name)

    def log_validation_losses(self, epoch, loss_dict, tag='valid'):
        outfile = self.output_dir+'/'+tag+'_'+str(epoch)+'.log'
        with open (outfile,'w') as f:
            for line in loss_dict:
                f.write(str(line)+'\n')

    
    def write_pdb(self, example, outputs, dump_traj=False):
        from datahub.utils.io import convert_af3_model_output_to_atom_array_stack
        from datahub.encoding_definitions import AF3SequenceEncoding
        from cifutils.utils.io_utils import to_cif_file
        
        encoding = AF3SequenceEncoding()
        # Collect information needed to write out the CIF file
        atom_to_token_map = example["feats"]["atom_to_token_map"].cpu().numpy()
        decoded_restypes = encoding.decode(torch.argmax(example["feats"]["restype"], dim=-1).cpu())
        pn_unit_iids = example["ground_truth"]["chain_iid_token_lvl"]

        elements = torch.argmax(example["feats"]["ref_element"], -1).cpu().numpy()
        example_id = example["example_id"]

        if dump_traj:
            for i,modelxyz in enumerate(outputs['X_denoised_L_traj']):
                xyz = modelxyz.cpu().numpy()

                # Convert the model output to an atom array
                atom_array_stack = convert_af3_model_output_to_atom_array_stack(
                    atom_to_token_map=atom_to_token_map,
                    pn_unit_iids=pn_unit_iids,
                    decoded_restypes=decoded_restypes,
                    xyz=xyz,
                    elements=elements,
                )

                outfile = self.output_dir + f"/{example_id}_denoised_{i}.cif"
                logger.info(f"Writing {outfile}")
                to_cif_file(atom_array_stack, outfile)

            for i,modelxyz in enumerate(outputs['X_noisy_L_traj']):
                xyz = modelxyz.cpu().numpy()

                # Convert the model output to an atom array
                atom_array_stack = convert_af3_model_output_to_atom_array_stack(
                    atom_to_token_map=atom_to_token_map,
                    pn_unit_iids=pn_unit_iids,
                    decoded_restypes=decoded_restypes,
                    xyz=xyz,
                    elements=elements,
                )

                outfile = self.output_dir + f"/{example_id}_noisy_{i}.cif"
                logger.info(f"Writing {outfile}")
                to_cif_file(atom_array_stack, outfile)

        else:
            xyz = outputs["X_L"].cpu().numpy()

            # Convert the model output to an atom array
            atom_array_stack = convert_af3_model_output_to_atom_array_stack(
                atom_to_token_map=atom_to_token_map,
                pn_unit_iids=pn_unit_iids,
                decoded_restypes=decoded_restypes,
                xyz=xyz,
                elements=elements,
            )

            # Write the atom array to a CIF file
            # NOTE: If the secondary structure does not appear, run `dss` in PyMol 
            # (see: https://biology.stackexchange.com/questions/70143/can-pymol-show-cartoon-secondary-structure-for-a-pdb-of-multiple-frames)
            outfile = self.output_dir + f"/{example_id}.cif"
            logger.info(f"Writing {outfile}")
            to_cif_file(atom_array_stack, outfile)


    def unbatch_losses(self, loss_dict_batched):
        loss_dict = {}
        for key, batched_loss in loss_dict_batched.items():
            if batched_loss.numel() == 1:   
                loss_dict[key] = batched_loss.item()
                continue
            for i, loss in enumerate(batched_loss):
                loss_dict[f"{key}.{i}"] = loss
        return loss_dict
    
    def construct_optimizer(self):
        self.optimizer = getattr(torch.optim, self.config.optimizer.type)(
            self.model.parameters(),
            **self.config.optimizer.params,
        )

class AF3TrainerRollout(AF3Trainer):

    def construct_model(self, device="cpu", inference=False):
        #fd initialize chemical data based on input arguments
        #   this needs to be initialized first
        #initialize chemdata here so that we can use it in the confidence head, including in inference
        init = partial(initialize_chemdata, self.config)
        init()

        model = AF3_structure.Model(**self.config.model).to(device)
        model.device = device
        from rf2aa.model.layers.af3_auxiliary_heads import ConfidenceHead
        confidence = ConfidenceHead(**self.config.confidence_head).to(device)
        self.confidence = confidence
        from rf2aa.flow_matching.sampler import AF3Sampler
        self.sampler = AF3Sampler(self.config, model, confidence=self.confidence)
        from rf2aa.model.af3_with_rollout import AF3_with_rollout 
        self.model = AF3_with_rollout(
            model,
            confidence,
            self.sampler,
            self.config.dataset_params.diffusion_batch_size_rollout
        )

        if self.config.training_params.EMA is not None:
            self.model = EMA(self.model, self.config.training_params.EMA)

        if inference is False:
            self.model = DDP(self.model, device_ids=[device], find_unused_parameters=True, broadcast_buffers=False)
        else:
            from rf2aa.training.EMA import FakeDDPWrapper
            self.model = FakeDDPWrapper(self.model)
        self.sampler.model = self.model.module.shadow.model
        self.sampler.confidence = self.model.module.shadow.confidence
        self.loss = AF3Loss(**self.config.loss)
        self.subunit_symm_resolve = SubunitSymmetryResolution()
        self.residue_symm_resolve = ResidueSymmetryResolution()
        self.confidence_indices = GetConfidenceIndices()
        self.metrics = MetricManager(**self.config.metrics)

    def train_step(self, inputs, n_cycle, no_grads=False, return_outputs=False):
        gpu = self.model.device

        example = inputs[0]
        
        network_input = {
            "X_noisy_L": torch.nan_to_num(example["coord_atom_lvl_to_be_noised"]) + example["noise"],
            "t": example["t"],
            "f": example["feats"],
        } 
        del network_input["f"]["ref_automorphs"]
        del network_input["f"]["ref_automorphs_mask"]

        loss_input = {
            "X_gt_L": example["ground_truth"]["coord_atom_lvl"][None].expand(self.config.dataset_params.diffusion_batch_size, -1, -1),
            "crd_mask_L": example["ground_truth"]["mask_atom_lvl"][None].expand(self.config.dataset_params.diffusion_batch_size, -1),
            "X_rep_atoms_I": example["ground_truth"]["coord_token_lvl"],
            "crd_mask_rep_atoms_I": example["ground_truth"]["mask_token_lvl"],
            "seq": example["confidence_feats"]["rf2aa_seq"],
            "atom_frames": example["confidence_feats"]["atom_frames"],
            "tok_idx": example['feats']['atom_to_token_map'],
            "is_real_atom": example["confidence_feats"]['is_real_atom'],
            "rep_atom_idxs": example['ground_truth']['rep_atom_idxs'],
            "frame_atom_idxs": example["confidence_feats"]['pae_frame_idx_token_lvl_from_atom_lvl'],
            #"terminal_oxygen_idxs": example["confidence_feats"]["terminal_oxygen_idx_atm_lvl"],
        }

        network_input=tree.map_structure(lambda x: x.to(gpu) if hasattr(x, 'cpu') else x, network_input)
        loss_input = tree.map_structure(lambda x: x.to(gpu) if hasattr(x, 'cpu') else x, loss_input)
        logger.debug('network_input:\n' + pretty_describe_dict(network_input))
        logger.debug('loss_input:\n' + pretty_describe_dict(loss_input))
        output_i = self.model(
            network_input,
            n_cycle,
            loss_input["seq"],
            loss_input['rep_atom_idxs'],
            frame_atom_idxs=loss_input['frame_atom_idxs'],
            no_sync=self.model.no_sync,
        )

        # Symmetry resolution
        #Change X_L to the rollout so gt matches rollout batch dimension during the symmetry resolution. This assumes that we are not
        #evaluating non-confidence head losses on the rollout model
        output_i["X_L"] = output_i["X_pred_rollout_L"]
        B = output_i["X_L"].shape[0]
        if loss_input['X_gt_L'].shape[0] == 1:
            loss_input['X_gt_L'] = loss_input['X_gt_L'].expand(B, -1, -1)
            loss_input['crd_mask_L'] = loss_input['crd_mask_L'].expand(B, -1)

        loss_input = self.subunit_symm_resolve(output_i, loss_input, example["symmetry_resolution"])
        loss_input = self.residue_symm_resolve(output_i, loss_input, example["automorphisms"])
        
        loss, loss_dict_batched = self.loss(
            network_input,
            output_i,
            loss_input
        )
        loss_dict = self.unbatch_losses(loss_dict_batched)

        return loss, loss_dict
    

    def valid_step(self, inputs, n_cycle, no_grads=True, return_outputs=False):
        gpu = self.model.device
        
        example = inputs[0]

        network_input = {
            "X_noisy_L": torch.nan_to_num(example["coord_atom_lvl_to_be_noised"]) + example["noise"],
            "t": example["t"],
            "f": example["feats"],
        } 
        loss_input = {
            "X_gt_L": example["ground_truth"]["coord_atom_lvl"][None],
            "crd_mask_L": example["ground_truth"]["mask_atom_lvl"][None],
            "X_rep_atoms_I": example["ground_truth"]["coord_token_lvl"],
            "crd_mask_rep_atoms_I": example["ground_truth"]["mask_token_lvl"],
            "interfaces_to_score": example["ground_truth"]["interfaces_to_score"],
            "pn_units_to_score": example["ground_truth"]["pn_units_to_score"],
            "chain_iid_token_lvl": example["ground_truth"]["chain_iid_token_lvl"],
            "example_id": example["example_id"],
            "is_real_atom": example["confidence_feats"]['is_real_atom'].to(gpu),
            "is_ligand": example["feats"]['is_ligand'].to(gpu),
        }

        outputs = self.sampler.sample(inputs, n_cycle=n_cycle, use_amp=self.config.training_params.use_amp)

        def _inmap(path, x):
            if hasattr(x, 'cpu') and path != ('f','msa_stack'):
                return x.to(gpu) 
            else:
                return x

        network_input = tree.map_structure_with_path(_inmap, network_input)
        loss_input = tree.map_structure(lambda x: x.to(gpu) if hasattr(x, 'cpu') else x, loss_input)

        # symmetry resolution
        loss_input = self.subunit_symm_resolve(outputs, loss_input, example["symmetry_resolution"])
        loss_input = self.residue_symm_resolve(outputs, loss_input, example["automorphisms"])

        loss_input['confidence_loss'] = self.config.loss.confidence_loss
        loss_input = self.confidence_indices(network_input, outputs, loss_input)

        metrics_dict = self.metrics(network_input, outputs, loss_input)

        return torch.tensor(0), metrics_dict

    def load_model(self):
        torch.cuda.empty_cache()
        checkpoint_training_config = self.checkpoint['training_config']
        if "confidence_loss" in checkpoint_training_config["loss"]:
            logger.warning("Loading weights with pretrained confidence head because confidence loss is present in the checkpoint")
            super().load_model()
        else:
            logger.warning("Loading weights from a model that was not trained with a confidence head. Renaming weights to be compatible with confidence head")
            self.model.module.model.model.load_state_dict(self.checkpoint["final_state_dict"], strict=False)
            self.model.module.shadow.model.load_state_dict(self.checkpoint["model_state_dict"], strict=False)
            logger.info("Checkpoint loaded into model")
            logger.warning("Resetting optimizer since model was not trained with confidence head")
            self.config.training_params.reset_optimizer_params = True
        for name, param in self.model.named_parameters():
            if 'confidence' not in name:
                param.requires_grad = False
