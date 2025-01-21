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
from rf2aa.metrics.metric_utils import unbin_rf3_metrics, get_ipae_metrics_from_binned
from rf2aa.debug import pretty_describe_dict

from rf2aa.chemical import ChemicalData as ChemData
import warnings
from rf2aa.training.recycling import recycle_sampling
from functools import partial
import omegaconf
import datetime
from rf2aa.chemical import initialize_chemdata
from contextlib import nullcontext
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
            self.log_validation_losses(epoch, all_valid_loss_dict)

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
            for i,modelxyz in enumerate(outputs['X_noisy_L_traj']):
                if (i%10) != 0:
                    continue

                xyz = modelxyz.cpu().numpy()

                # Convert the model output to an atom array
                atom_array_stack = convert_af3_model_output_to_atom_array_stack(
                    atom_to_token_map=atom_to_token_map,
                    pn_unit_iids=pn_unit_iids,
                    decoded_restypes=decoded_restypes,
                    xyz=xyz,
                    elements=elements,
                )

                outfile = self.output_dir + f"/{example_id}_traj_{i}.cif"
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
        import copy
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
        self.metrics = MetricManager(**self.config.metrics)

    def train_step(self, inputs, n_cycle, no_grads=False, return_outputs=False):
        gpu = self.model.device

        example = inputs[0]
        print('example id', example["example_id"])
        
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
            "terminal_oxygen_idxs": example["confidence_feats"]["terminal_oxygen_idx_atm_lvl"],
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

        # symmetry resolution
        #HACK change X_L to the rollout so gt matches rollout batch dimesnion. This assumes that weights for 
        #non-confidence head losses are set to 0
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
    


    def load_model(self):
        torch.cuda.empty_cache()

        new_model_state = {}
        new_shadow_state = {}
        state_dict = self.model.module.model.state_dict()

        if self.config.training_params.make_fresh_weights_confidence_compatible:
            #get around a loading issue - only need to do this when loading from a weight set trained on something other than the rollout
            # Create a new dictionary to store the modified keys
            new_model_state_dict = {}

            # Iterate over the original dictionary
            for param in self.checkpoint['model_state_dict']:
                # Modify the key
                new_param = 'model.' + param
                # Add the modified key and its value to the new dictionary
                new_model_state_dict[new_param] = self.checkpoint['model_state_dict'][param]

            # Replace the original dictionary with the new one
            self.checkpoint['model_state_dict'] = new_model_state_dict

            #Do the same for the final_state_dict
            new_model_state_dict = {}
            for param in self.checkpoint['final_state_dict']:
                # Modify the key
                new_param = 'model.' + param
                # Add the modified key and its value to the new dictionary
                new_model_state_dict[new_param] = self.checkpoint['final_state_dict'][param]
            self.checkpoint['final_state_dict'] = new_model_state_dict

        for param in state_dict:
            if param not in self.checkpoint['model_state_dict']:
                print ('missing',param)
            #elif ('refinement.atom_decoder.0.se3.final_proj.weight' in param): #fd hack
            #    print ('skipping',param)
            elif (self.checkpoint['model_state_dict'][param].shape == state_dict[param].shape):
                new_model_state[param] = self.checkpoint['final_state_dict'][param]
                new_shadow_state[param] = self.checkpoint['model_state_dict'][param]
            else:
                print (
                    'wrong size',param,
                    self.checkpoint['model_state_dict'][param].shape,
                    state_dict[param].shape )

        self.model.module.model.load_state_dict(new_model_state, strict=False)
        self.model.module.shadow.load_state_dict(new_shadow_state, strict=False)
        print("Checkpoint loaded into model")

    #Recreated here rather than inherited from super so we can ensure grads are set to false for non-confidence parameters
    def train_model(self, rank, world_size):
        """ runs model training on each gpu """ 
        gpu = self.init_process_group(rank, world_size) 
        #rank = gpu
        ic(rank, world_size, gpu)

        #fd initialize chemical data based on input arguments
        #   this needs to be initialized first
        init = partial(initialize_chemdata, self.config)
        init()

        # Define context manager for training run (either nullcontext or W&B)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        if self.config.log_params.use_wandb:    
            import wandb
            wandb.login()
            context_manager = wandb.init(
                    project=self.config.log_params.wandb_project, 
                    config=omegaconf.OmegaConf.to_container(
                        self.config, resolve=True, throw_on_missing=True
                    ),
                    name = f"{self.config.experiment.name}_{timestamp}"
                ) 
        else:
            context_manager = nullcontext() # Does nothing
        

        # Without W&B, context manager does nothing
        with context_manager: 
            train_loader, train_sampler, valid_loaders, valid_samplers = self.construct_dataset(
                init, rank, world_size
            )

            self.train_loader = train_loader
            self.valid_loaders = valid_loaders

            # move global information to device
            self.move_constants_to_device(gpu)

            self.construct_model(device=gpu)
            if rank == 0:
                print(f"Loading model with {count_parameters(self.model)} parameters")

            self.construct_optimizer()
            self.construct_scheduler()
            self.construct_scaler()
            start_epoch = 0
            loaded_checkpoint = self.load_checkpoint(gpu)
            logger.info(f'Loaded checkpoint: {loaded_checkpoint}')
            if loaded_checkpoint:
                start_epoch = self.checkpoint["epoch"] + 1
                self.load_model()
                if not self.config.training_params.reset_optimizer_params:
                    self.load_optimizer()
                    self.load_scheduler()
                    self.load_scaler()
                else:
                    warnings.warn(f"User specified reset_optimizer_params=True. Did not load optimizer values from checkpoint")
            self.checkpoint = None # unload checkpoint dict

            self.recycle_schedule = recycle_sampling["by_batch"](self.config.loader_params.maxcycle, 
                                                                self.config.experiment.n_epoch,
                                                                self.config.dataset_params.n_train,
                                                                world_size)

            #set requires_grad to false for all non confidence parameters to be safe
            for name, param in self.model.named_parameters():
                if 'confidence' not in name:
                    param.requires_grad = False

            for epoch in range(start_epoch,self.config.experiment.n_epoch):
                train_sampler.set_epoch(epoch) #TODO: need to make sure each gpu gets a different example

                print(f'about to go into valid_epoch for epoch {epoch}')
                self.valid_epoch(epoch, rank, world_size)
                print(donevalidating)

                self.train_epoch(epoch, rank, world_size)
                for _, valid_sampler in valid_samplers.items():
                    valid_sampler.set_epoch(epoch)

                if (
                    self.config.dataset_params.validate_every_n_epochs > 0 
                    and epoch % self.config.dataset_params.validate_every_n_epochs==0
                    and (epoch!=start_epoch or self.config.dataset_params.validate_after_first_epoch)
                ):
                    self.valid_epoch(epoch, rank, world_size)

        self.cleanup()

    def valid_step(self, inputs, n_cycle, no_grads=True, return_outputs=False):
        #TODO: get rid of for loops here
        gpu = self.model.device
        
        example = inputs[0]
        print('example id', example["example_id"])
        network_input = {
            #TODO: make a transform that places unresolved ground truth coordinates on their closest real atomshh
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
            "alignment_mask": example["alignment_mask_atm_lvl"],
            
            #for loss calc
            "seq": example["confidence_feats"]["rf2aa_seq"],
            "atom_frames": example["confidence_feats"]["atom_frames"],
            "tok_idx": example['feats']['atom_to_token_map'],
            "is_real_atom": example["confidence_feats"]['is_real_atom'],
            "rep_atom_idxs": example['ground_truth']['rep_atom_idxs'],
            "frame_atom_idxs": example["confidence_feats"]['pae_frame_idx_token_lvl_from_atom_lvl'],
            "terminal_oxygen_idxs": example["confidence_feats"]["terminal_oxygen_idx_atm_lvl"],
        }

        msa_stack = network_input["f"]["msa_stack"]

        #AF3's ranking metrics work more like this, but using ptm instead of ipae:
        ch_label = example["ground_truth"]["chain_iid_token_lvl"]
        unique_chains = np.unique(ch_label)
        
        if len(unique_chains) > 1:
            interface_mask = torch.from_numpy(ch_label[None,:] != ch_label[:,None]).to(dtype=torch.bool, device=gpu)
        else:
            interface_mask = torch.zeros(len(ch_label), len(ch_label), device=gpu, dtype=torch.bool)

        ch_label = example["ground_truth"]["chain_iid_token_lvl"]
        unique_chains = np.unique(ch_label)
        interface_masks = {}
        chain_masks = {}
        single_chain_masks = {}
        interfaces = []
        scored_chains = []
        chains = []
        interface_chains = []
        for k in loss_input['interfaces_to_score']:
            interfaces.append(f'{k[0]}-{k[1]}')
            chains.append(k[0])
            chains.append(k[1])
            interface_chains.append(k[0])
            interface_chains.append(k[1])
        for k in loss_input['pn_units_to_score']:
            chains.append(k[0])
            scored_chains.append(k[0])
        chains = set(chains)
        lig_chains = []
        
        for chain in chains:
            #check if this is a ligand chain; AF3 handles ligand metrics slightly differently
            if torch.all(network_input['f']['is_ligand'][ch_label == chain]):
                lig_chains.append(chain)
            chain_mask = torch.zeros(loss_input["seq"].shape[-1], loss_input["seq"].shape[-1], device=gpu, dtype=torch.bool)
            single_chain_mask = torch.zeros(loss_input["seq"].shape[-1], loss_input["seq"].shape[-1], device=gpu, dtype=torch.bool)
            i_mask = torch.zeros(loss_input["seq"].shape[-1], loss_input["seq"].shape[-1], device=gpu, dtype=torch.bool)
            for i in range(len(ch_label)):
                for j in range(i+1, len(ch_label)):
                    if (ch_label[i] == chain or ch_label[j] == chain):
                        chain_mask[i,j] = True
                        chain_mask[j,i] = True
                        if ch_label[i] == chain and ch_label[j] == chain:
                            single_chain_mask[i,j] = True
                            single_chain_mask[j,i] = True
                        elif ch_label[i] != ch_label[j]:
                            i_mask[i,j] = True
                            i_mask[j,i] = True
            interface_masks[chain] = i_mask
            single_chain_masks[chain] = single_chain_mask
            chain_masks[chain] = chain_mask

        same_chain = torch.zeros(loss_input["seq"].shape[-1], loss_input["seq"].shape[-1], device=gpu, dtype=torch.bool)
        for i in range(len(ch_label)):
            for j in range(i, len(ch_label)):
                if ch_label[i] == ch_label[j]:
                    same_chain[i,j] = True
                    same_chain[j,i] = True
        same_chain = same_chain.unsqueeze(0)

        if True:
            alt_ch_label = example["ground_truth"]["chain_iid_token_lvl"]
            alt_unique_chains = np.unique(ch_label)

            # Interface mask generation
            if len(alt_unique_chains) > 1:
                alt_interface_mask = torch.from_numpy(alt_ch_label[None, :] != alt_ch_label[:, None]).to(dtype=torch.bool, device=gpu)
            else:
                alt_interface_mask = torch.zeros(len(alt_ch_label), len(alt_ch_label), device=gpu, dtype=torch.bool)

            # Chain masks preparation (vectorized)
            alt_ch_tensor = torch.tensor(alt_ch_label, device=gpu)
            alt_seq_len = loss_input["seq"].shape[-1]
            alt_chain_masks = {}
            alt_single_chain_masks = {}
            alt_interface_masks = {}
            alt_lig_chains = []
            
            for chain in set(alt_ch_label):
                mask = alt_ch_tensor == chain
                alt_chain_mask = mask.unsqueeze(0) | mask.unsqueeze(1)
                alt_single_chain_mask = mask.unsqueeze(0) & mask.unsqueeze(1)
                alt_interface_mask = alt_chain_mask & ~alt_single_chain_mask
                
                alt_chain_masks[chain] = alt_chain_mask
                alt_single_chain_masks[chain] = alt_single_chain_mask
                alt_interface_masks[chain] = alt_interface_mask
                
                if torch.all(network_input['f']['is_ligand'][mask]):
                    alt_lig_chains.append(chain)
            
            # Same-chain mask (vectorized)
            alt_same_chain = (alt_ch_tensor.unsqueeze(0) == alt_ch_tensor.unsqueeze(1)).to(dtype=torch.bool).unsqueeze(0)

            print('chain masks match: ', all([torch.all(alt_chain_masks[chain] == chain_masks[chain]) for chain in chains]))
            print('single chain masks match: ', all([torch.all(alt_single_chain_masks[chain] == single_chain_masks[chain]) for chain in chains]))
            print('interface masks match: ', all([torch.all(alt_interface_masks[chain] == interface_masks[chain]) for chain in chains]))
            print('lig chains match: ', alt_lig_chains == lig_chains)
            print('same chain masks match: ', torch.all(alt_same_chain == same_chain))




        pred_err = []
        i_pae_err = []
        interface_err = {}
        for interface in interfaces:
            interface_err[interface] = []
        lig_err = {}
        for lig_chain in lig_chains:
            lig_err[lig_chain] = []
        chain_err = {}
        single_chain_err = {}
        for chain in scored_chains:
            chain_err[chain] = []
            single_chain_err[chain] = []

        # output_stack = {
        #     "X_L": [],
        #     "X_gt_L": [],
        #     "crd_mask_L": [],
        #     'plddt': [],
        #     'pae': [],
        #     'pde': [],
        #     'exp_resolved': [],
        # }

        #set up for sampling the trunk multiple times if desired - doesn't typically matter unless trunk_batch_size_valid > 1
        #get the right msa_stack for this pass
        # network_input["f"]["msa_stack"] = msa_stack[(self.config.dataset_params.trunk_batch_size_valid * i):(self.config.dataset_params.trunk_batch_size_valid * i) + 10]

        outputs = self.sampler.sample(inputs, n_cycle=10, use_amp=self.config.training_params.use_amp)

        # B = outputs["X_L"].shape[0]
        # confidence = self.confidence(
        #     outputs["S_inputs_I"].repeat(B, 1, 1),
        #     outputs["S_I"].repeat(B, 1, 1),
        #     outputs["Z_II"].repeat(B, 1, 1, 1),
        #     outputs["X_L"],
        #     loss_input["seq"],
        #     example['ground_truth']['rep_atom_idxs'].to(outputs["X_L"].device),
        #     frame_atom_idxs=loss_input['frame_atom_idxs'].to(outputs["X_L"].device),
        # )
        # print('confidence.keys', confidence.keys())

        confidence = outputs['confidence']
        
        for i in range(confidence['plddt_logits'].shape[0]):
            plddt_logits = confidence['plddt_logits'][i].unsqueeze(0)
            plddt_logits = plddt_logits.reshape(plddt_logits.shape[0], -1, plddt_logits.shape[1], ChemData().NHEAVY)

            plddt, pae, pde = unbin_rf3_metrics(plddt_logits.float(), confidence['pae_logits'][i].unsqueeze(0).permute(0,3,1,2).float(), confidence['pde_logits'][i].unsqueeze(0).permute(0,3,1,2).float(), loss_input["seq"].to(plddt_logits.device), is_real_atom=loss_input['is_real_atom'].to(gpu))

            pred_err.append({'plddt': plddt, 'pae': pae, 'pde': pde})

            _, i_pae, _ = unbin_rf3_metrics(plddt_logits.float(), confidence['pae_logits'][i].unsqueeze(0).permute(0,3,1,2).float(), confidence['pde_logits'][i].unsqueeze(0).permute(0,3,1,2).float(), loss_input["seq"].to(plddt_logits.device), pae_mask=interface_mask, is_real_atom=loss_input['is_real_atom'].to(gpu))
            i_pae_err.append(i_pae)

            #AF3-style confidence ranking metrics
            for interface in interfaces:
                chain_a = interface.split('-')[0]
                chain_b = interface.split('-')[1]

                #af3 only considers the ligand chain when evaluating interfaces containing a ligand
                if (chain_a in lig_chains or chain_b in lig_chains) and not (chain_a in lig_chains and chain_b in lig_chains):
                    if chain_a in lig_chains:
                        lig_chain = chain_a
                    elif chain_b in lig_chains:
                        lig_chain = chain_b

                    #if a ligand participates in more than 1 interface, we still only want to get calculcate B scores
                    if len(lig_err[lig_chain]) < i+1:
                        _, lig_i_pae, _ = unbin_rf3_metrics(plddt_logits.float(), confidence['pae_logits'][i].unsqueeze(0).permute(0,3,1,2).float(), confidence['pde_logits'][i].unsqueeze(0).permute(0,3,1,2).float(), loss_input["seq"].to(plddt_logits.device), pae_mask=interface_masks[lig_chain], is_real_atom=loss_input['is_real_atom'].to(gpu))
                        lig_err[lig_chain].append(lig_i_pae)
                    
                _, chain_a_i_pae, _ = unbin_rf3_metrics(plddt_logits.float(), confidence['pae_logits'][i].unsqueeze(0).permute(0,3,1,2).float(), confidence['pde_logits'][i].unsqueeze(0).permute(0,3,1,2).float(), loss_input["seq"].to(plddt_logits.device), pae_mask=interface_masks[chain_a], is_real_atom=loss_input['is_real_atom'].to(gpu))
                _, chain_b_i_pae, _ = unbin_rf3_metrics(plddt_logits.float(), confidence['pae_logits'][i].unsqueeze(0).permute(0,3,1,2).float(), confidence['pde_logits'][i].unsqueeze(0).permute(0,3,1,2).float(), loss_input["seq"].to(plddt_logits.device), pae_mask=interface_masks[chain_b], is_real_atom=loss_input['is_real_atom'].to(gpu))
                interface_err[interface].append((chain_a_i_pae + chain_b_i_pae))
            for chain in scored_chains:
                _, chain_pae, _ = unbin_rf3_metrics(plddt_logits.float(), confidence['pae_logits'][i].unsqueeze(0).permute(0,3,1,2).float(), confidence['pde_logits'][i].unsqueeze(0).permute(0,3,1,2).float(), loss_input["seq"].to(plddt_logits.device), pae_mask=chain_masks[chain], is_real_atom=loss_input['is_real_atom'].to(gpu))
                chain_err[chain].append(chain_pae)
                _, single_chain_pae, _ = unbin_rf3_metrics(plddt_logits.float(), confidence['pae_logits'][i].unsqueeze(0).permute(0,3,1,2).float(), confidence['pde_logits'][i].unsqueeze(0).permute(0,3,1,2).float(), loss_input["seq"].to(plddt_logits.device), pae_mask=single_chain_masks[chain], is_real_atom=loss_input['is_real_atom'].to(gpu))
                single_chain_err[chain].append(single_chain_pae)

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

        # output_stack["X_L"].append(outputs["X_L"])
        # output_stack["X_gt_L"].append(loss_input["X_gt_L"])
        # output_stack["crd_mask_L"].append(loss_input["crd_mask_L"])
        # output_stack['plddt'].append(confidence['plddt_logits'])
        # output_stack['pae'].append(confidence['pae_logits'])
        # output_stack['pde'].append(confidence['pde_logits'])
        # output_stack['exp_resolved'].append(confidence['exp_resolved_logits'])


        #Get the index of the lowest complex metric
        best_plddt = -1.0
        best_pde = 100
        best_pae = 100
        plddt_idx = -1
        pde_idx = -1
        pae_idx = -1
        best_i_pae = 100
        ipae_idx = -1
        for i in range(len(pred_err)):
            if pred_err[i]['plddt'] > best_plddt:
                best_plddt = pred_err[i]['plddt']
                plddt_idx = i
            if pred_err[i]['pde'] < best_pde:
                best_pde = pred_err[i]['pde']
                pde_idx = i
            if pred_err[i]['pae'] < best_pae:
                best_pae = pred_err[i]['pae']
                pae_idx = i
            if i_pae_err[i] < best_i_pae:
                best_i_pae = i_pae_err[i]
                ipae_idx = i
        loss_input['pae_idx'] = pae_idx
        loss_input['pde_idx'] = pde_idx
        loss_input['plddt_idx'] = plddt_idx
        loss_input['ipae_idx'] = ipae_idx

        #now the same for AF3-style metrics
        best_interface_idx = {}
        best_lig_ipae_idx = {}
        for k, v in interface_err.items():
            best_pae = 100
            chain_a = k.split('-')[0]
            chain_b = k.split('-')[1]
            chain_a = np.nonzero(unique_chains == chain_a)[0][0]
            chain_b = np.nonzero(unique_chains == chain_b)[0][0]
            for i in range(len(v)):
                if v[i] < best_pae:
                    best_pae = v[i]
                    best_interface_idx[k] = i

            #handle special af3-style lig case
            best_lig_ipae_idx[k] = -1
            chain_1 = k.split('-')[0]
            chain_2 = k.split('-')[1]
            if chain_1 in lig_chains or chain_2 in lig_chains:
                if chain_1 in lig_chains:
                    lig_chain = chain_1
                    lig_num = chain_a
                elif chain_2 in lig_chains:
                    lig_chain = chain_2
                    lig_num = chain_b
                best_pae = 100
                for i in range(len(lig_err[lig_chain])):
                    if lig_err[lig_chain][i] < best_pae:
                        best_pae = lig_err[lig_chain][i]
                        best_lig_ipae_idx[k] = i

        loss_input['best_interface_idx'] = best_interface_idx
        loss_input['best_lig_ipae_idx'] = best_lig_ipae_idx

        best_chain_idx = {}
        best_single_chain_idx = {}
        for chain in scored_chains:
            best_chain_pae = 100
            best_single_chain_pae = 100
            for i in range(len(chain_err[chain])):
                if chain_err[chain][i] < best_chain_pae:
                    best_chain_pae = chain_err[chain][i]
                    best_chain_idx[chain] = i
                if single_chain_err[chain][i] < best_single_chain_pae:
                    best_single_chain_pae = single_chain_err[chain][i]
                    best_single_chain_idx[chain] = i
        loss_input['best_chain_idx'] = best_chain_idx
        loss_input['best_single_chain_idx'] = best_single_chain_idx

        #now cat the outputs that matter for the confidence metrics and loss
        # outputs["X_L"] = torch.cat(output_stack["X_L"], dim=0)
        # loss_input["X_gt_L"] = torch.cat(output_stack["X_gt_L"], dim=0)
        # loss_input["crd_mask_L"] = torch.cat(output_stack["crd_mask_L"], dim=0)
        # outputs['plddt'] = torch.cat(output_stack['plddt'], dim=0)
        # outputs['pae'] = torch.cat(output_stack['pae'], dim=0)
        # outputs['pde'] = torch.cat(output_stack['pde'], dim=0)
        # outputs['exp_resolved'] = torch.cat(output_stack['exp_resolved'], dim=0)
        # outputs['X_pred_rollout_L'] = outputs['X_L']

        #clear up memory
        # del output_stack
        # print('B:', outputs['X_L'].shape[0])


        metrics_dict = self.metrics(network_input, outputs, loss_input)

        return torch.tensor(0), metrics_dict