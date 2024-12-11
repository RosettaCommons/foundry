import re
import torch
import logging
import tree
import time

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from rf2aa.trainer_new import FlowMatchingTrainer
from rf2aa.model import AF3_structure
from rf2aa.data.dataloader_adaptor_af3 import prepare_input_af3
from rf2aa.data.compose_data_datahub_new import NewDatapipeTrainer

from rf2aa.training.EMA import EMA, count_parameters
from rf2aa.flow_matching.sampler import AF3Sampler, AF3PartialSampler
from rf2aa.loss.af3_losses import Loss as AF3Loss
from rf2aa.loss.af3_losses import SubunitSymmetryResolution, ResidueSymmetryResolution
from rf2aa.metrics.metrics_base import MetricManager
from rf2aa.debug import pretty_describe_dict

import rf2aa.util as util
from rf2aa.data.dataloader_adaptor import prepare_input, get_loss_calc_items, prepare_input_fm_allatom
from rf2aa.chemical import ChemicalData as ChemData
import warnings
from rf2aa.training.recycling import recycle_sampling
from functools import partial
import omegaconf
import datetime
from rf2aa.chemical import initialize_chemdata
from contextlib import nullcontext
from icecream import ic


logger = logging.getLogger(__name__)

class AF3Trainer(FlowMatchingTrainer):

    def construct_model(self, device="cpu"):
        self.model = AF3_structure.Model(**self.config.model).to(device)

        print_n_params = False
        if print_n_params:
            logger.info(f'{get_n_params(self.model)=}')
            for k, v in sorted(get_param_sizes(self.model).items(), key=lambda item: item[1]):
                n_param, size = v
                # n_param = np.array(p.size()).prod()
                logger.info(f'{n_param=} {k=} {size=}')

        if self.config.training_params.EMA is not None:
            self.model = EMA(self.model, self.config.training_params.EMA)

        #self.model = DDP(self.model, device_ids=[device], find_unused_parameters=False, broadcast_buffers=False)
        self.model = DDP(self.model, device_ids=None, find_unused_parameters=False, broadcast_buffers=False)
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

    
    def write_pdb(self, X_gt_L, crd_mask_I, seq, name="valid.pdb"):
        I = seq.shape[0]

        is_real_atom = ChemData().heavyatom_mask.to(self.model.device)[seq]
        X_gt_I = torch.zeros(I, ChemData().NTOTAL, 3, device=self.model.device)
        X_gt_I[is_real_atom] = X_gt_L
        import rf2aa
        rf2aa.util.writepdb(filename=name, atoms=X_gt_I, atom_mask=crd_mask_I, seq=seq)

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

    def construct_model(self, device="cpu"):
        model = AF3_structure.Model(**self.config.model).to(device)
        model.device = device
        from rf2aa.model.layers.af3_auxiliary_heads import ConfidenceHead
        confidence = ConfidenceHead(**self.config.confidence_head).to(device)
        self.confidence = confidence
        from rf2aa.flow_matching.sampler import AF3Sampler
        self.sampler = AF3Sampler(self.config, model)
        from rf2aa.model.af3_with_rollout import AF3_with_rollout 
        import copy
        self.model = AF3_with_rollout(
            model,
            confidence,
            # copy.deepcopy(self.sampler)
            self.sampler,
            self.config
        )

        if self.config.training_params.EMA is not None:
            self.model = EMA(self.model, self.config.training_params.EMA)
        def should_ignore(param_name):
            ignore_regexes = [
                re.compile(r'model\.model\.feature_initializer\.input_feature_embedder\.atom_attention_encoder\.process_s_trunk\..*'),
                re.compile(r'model\.model\.feature_initializer\.input_feature_embedder\.atom_attention_encoder\.process_z\..*'),
                re.compile(r'model\.model\.feature_initializer\.input_feature_embedder\.atom_attention_encoder\.process_r\..*'),
                re.compile(r'model\.model\.feature_initializer\.input_feature_embedder\.atom_attention_encoder\.atom_transformer\.diffusion_transformer\.blocks\.\d+\.attention_pair_bias.ln_1\..*'),
                re.compile(r'model\.model\.recycler\.pairformer_stack\.\d+\.attention_pair_bias\.linear_output_project\..*'),
                re.compile(r'model\.model\.recycler\.pairformer_stack\.\d+\.attention_pair_bias\.ada_ln_1\..*'),
                re.compile(r'model\.model\.diffusion_module\.atom_attention_encoder\.atom_transformer\.diffusion_transformer\.blocks\.\d+\.attention_pair_bias\.ln_1\..*'),
                re.compile(r'model\.model\.diffusion_module\.diffusion_transformer\.blocks\.\d+\.attention_pair_bias\.ln_1\..*'),
                re.compile(r'model\.model\.diffusion_module\.atom_attention_decoder\.atom_transformer\.diffusion_transformer\.blocks\.\d+\.attention_pair_bias\.ln_1\..*'),
                re.compile(r'model\.confidence\.pairformer\.\d+\.attention_pair_bias\.linear_output_project\..*'),
                re.compile(r'model\.confidence\.pairformer\.\d+\.attention_pair_bias\.ada_ln_1\..*'),

            ]
            return any(regex.match(param_name) for regex in ignore_regexes)
        params_to_ignore = []
        for param_name, param in self.model.named_parameters():
            if should_ignore(param_name):
                params_to_ignore.append(param_name)
        torch.nn.parallel.DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(
            self.model,
            params_to_ignore
        )
        # assert len(params_to_ignore)

        self.model = DDP(self.model, device_ids=[device], find_unused_parameters=True, broadcast_buffers=False)
        self.sampler.model = self.model.module.shadow.model
        self.loss = AF3Loss(**self.config.loss)
        self.subunit_symm_resolve = SubunitSymmetryResolution()
        self.residue_symm_resolve = ResidueSymmetryResolution()
        self.metrics = MetricManager(**self.config.metrics)

    def train_step(self, inputs, n_cycle, no_grads=False, return_outputs=False):
        gpu = self.model.device

        
        # network_input, loss_input = prepare_input_af3(
        #     inputs[0]['feats'],
        #     **self.config.af3_data_prep,
        # )

        example = inputs[0]
        print('example id', example["example_id"])
        # #Get sequence from restype so we can get rep atoms in ConfidenceHead
        # seq = example['feats']['restype'].to(gpu)
        # #now collapse the one-hot encoding to a single integer
        # seq = torch.argmax(seq, dim=-1)
        # #now offset everything above 20 since restype doesn't include the mask token
        # seq[seq > 20] += 1
        # print('seq', seq, seq.shape)


        # print('example:', example.keys())
        # print('example[encoded]', example['encoded'].keys())
        # print('example[encoded][seq]:', example['encoded']['seq'], example['encoded']['seq'].shape)
        # print('example[feats]:', example['feats'].keys())
        # print('exampe[feats][restype]:', example['feats']['restype'].shape)
        
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
            # "X_gt_I_symm": example["ground_truth"]["coord_token_lvl"][None].expand(self.config.dataset_params.diffusion_batch_size, -1, -1),
            # "crd_mask_I_symm": example["ground_truth"]["mask_token_lvl"][None].expand(self.config.dataset_params.diffusion_batch_size, -1),
            "seq": example["rf2aa_seq"],
            "atom_frames": example["atom_frames"],
            "tok_idx": example['feats']['atom_to_token_map'],
            "is_real_atom": example['is_real_atom'],
            "rep_atom_idxs": example['ground_truth']['rep_atom_idxs'],
            "frame_atom_idxs": example['frame_atom_idxs'],
        }

        # atom_to_token_map = example['feats']['atom_to_token_map']
        # ref_element = torch.argmax(example['feats']['ref_element'], dim=-1)
        # seq = example['rf2aa_seq']
        # print('ref_elements:', ref_element.shape)
        # print('atom_to_token_map:', atom_to_token_map.shape, atom_to_token_map)
        # for i, token in enumerate(example['rf2aa_seq']):
        #     expected_atoms = ChemData().heavyatom_mask[token].sum()
        #     observed_atoms = torch.sum(atom_to_token_map == i)
        #     if expected_atoms != observed_atoms:
        #         print('chain info', example['feats']['asym_id'], example['ground_truth']['chain_iid_token_lvl'])
        #         print(f'mismatch found for position {i} corresponding to token {token}')
        #         print(f'expected_atoms: {expected_atoms}, observed_atoms: {observed_atoms}')
        #         print(f'atom_to_token_map: {atom_to_token_map}')
        #         token_idxs = torch.where(atom_to_token_map == i)
        #         # print('token_idxs:', token_idxs.shape, token_idxs)
        #         token_plus_one_idxs = torch.where(atom_to_token_map == i+1)
        #         token_minus_one_idxs = torch.where(atom_to_token_map == i-1)
        #         token_plus_two_idxs = torch.where(atom_to_token_map == i+2)
        #         print(f'prev token is {seq[i-1]}')
        #         for idx in token_minus_one_idxs:
        #             print(f'atom in previous token {idx} is {ref_element[idx]}')
        #         print(f'this token is {seq[i]}')
        #         for idx in token_idxs:
        #             print(f'atom {idx} is {ref_element[idx]}')
        #         print(f'next token is {seq[i+1]}')
        #         for idx in token_plus_one_idxs:
        #             print(f'atom in next token {idx} is {ref_element[idx]}')
        #         print(f'next next token is {seq[i+2]}')
        #         for idx in token_plus_two_idxs:
        #             print(f'atom in next next token {idx} is {ref_element[idx]}')
        #         return torch.tensor(0.0, requires_grad=True), {}




        network_input=tree.map_structure(lambda x: x.to(gpu) if hasattr(x, 'cpu') else x, network_input)
        loss_input = tree.map_structure(lambda x: x.to(gpu) if hasattr(x, 'cpu') else x, loss_input)
        logger.debug('network_input:\n' + pretty_describe_dict(network_input))
        logger.debug('loss_input:\n' + pretty_describe_dict(loss_input))
        output_i = self.model(
            network_input,
            n_cycle,
            loss_input["seq"],
            loss_input['rep_atom_idxs'],
            # loss_input["X_gt_I_symm"].to(gpu),
            # loss_input["crd_mask_I_symm"].to(gpu),
            # loss_input["seq"].to(gpu),
            frame_atom_idxs=loss_input['frame_atom_idxs'],
            no_sync=self.model.no_sync,
        )

        # xyz = output_i["X_pred_rollout_L"]
        # xyz = convert_allatom_coords_to_residue_coords(xyz, loss_input["seq"])
        # util.writepdb('test_pdbs/rollout.pdb', xyz[0].unsqueeze(0), loss_input["seq"][None])
        # xyz = loss_input["X_gt_L"]
        # xyz = convert_allatom_coords_to_residue_coords(xyz, loss_input["seq"])
        # util.writepdb('test_pdbs/gt.pdb', xyz[0].unsqueeze(0), loss_input["seq"][None])
        # xyz = output_i["X_L"]
        # xyz = convert_allatom_coords_to_residue_coords(xyz, loss_input["seq"])
        # util.writepdb('test_pdbs/X_L.pdb', xyz[0].unsqueeze(0), loss_input["seq"][None])
        # print('wrote pdbs')
        # sys.exit(0)

        # symmetry resolution
        #HACK change X_L to the rollout so gt matches rollout batch dimesnion. This assumes that weights for 
        #non-confidence head losses are set to 0
        output_i["X_L"] = output_i["X_pred_rollout_L"]
        B = output_i["X_L"].shape[0]
        if loss_input['X_gt_L'].shape[0] == 1:
            loss_input['X_gt_L'] = loss_input['X_gt_L'].expand(B, -1, -1)
            loss_input['crd_mask_L'] = loss_input['crd_mask_L'].expand(B, -1)

        
        #getting some bugs in this
        try:
            loss_input = self.subunit_symm_resolve(output_i, loss_input, example["symmetry_resolution"])
            loss_input = self.residue_symm_resolve(output_i, loss_input, example["automorphisms"])
        except Exception as e:
            print('error in symmetry resolution', e)
            print('example id', example["example_id"])
            torch.save((output_i, loss_input, example["symmetry_resolution"],example["automorphisms"]), 'sym_error_data.pkl')
            print('continuing after saving in sym_error_data.pkl')


        loss, loss_dict_batched = self.loss(
            network_input,
            output_i,
            loss_input
        )
        loss_dict = self.unbatch_losses(loss_dict_batched)

        # output = {"X_L": output_i["X_L"]} | network_input | loss_input
        # from rf2aa.callbacks import lddt_metrics

        # lddt, _ = lddt_metrics(None, output)
        # sigma_data = self.config.af3_data_prep.sigma_data
        # t = network_input['t']
        # X_noisy_L = network_input['X_noisy_L']
        # null_pred = (sigma_data**2 / (sigma_data**2 + t**2))[...,None,None] * X_noisy_L
        # lddt_null, _ = lddt_metrics(None, {"X_L": null_pred} | network_input | loss_input)
        # for key, val in lddt_null.items():
        #     lddt[f"{key}_null_pred"] = val
        # loss_dict_batched["noise_std_dev"]  = torch.std(X_noisy_L, dim=(-1,-2))
        # loss_dict_batched = loss_dict_batched | lddt 
        # loss_dict_batched["t"] = network_input['t']
        # loss_dict.update(self.unbatch_losses(loss_dict_batched))
        return loss, loss_dict
    


    def load_model(self):
        torch.cuda.empty_cache()

        new_model_state = {}
        new_shadow_state = {}
        state_dict = self.model.module.model.state_dict()

        # def merge_torch_weights(first_file_path, second_file_path, output_file_path):
        #     """
        #     Merge PyTorch weight files with parameter renaming and filtering.
            
        #     Args:
        #     first_file_path (str): Path to the first .pt weight file
        #     second_file_path (str): Path to the second .pt weight file
        #     output_file_path (str): Path to save the merged weight file
        #     """
        #     # Load the first checkpoint
        #     first_checkpoint = torch.load(first_file_path)
            
        #     # Load the second checkpoint
        #     second_checkpoint = torch.load(second_file_path)
            
        #     # Create a new state dict
        #     merged_state_dict = {}
        #     merged_final_state_dict = {}
            
        #     # Rename parameters from the first checkpoint and add 'model.' prefix
        #     for key, value in first_checkpoint['model_state_dict'].items():
        #         print(f'Renaming {key} to model.{key}')
        #         merged_state_dict[f'model.{key}'] = value
        #     for key, value in first_checkpoint['final_state_dict'].items():
        #         print(f'Renaming {key} to model.{key}')
        #         merged_final_state_dict[f'model.{key}'] = value
            
        #     # Add parameters from the second checkpoint that contain 'confidence'
        #     for key, value in second_checkpoint['model_state_dict'].items():
        #         if 'confidence' in key:
        #             print(f'Adding {key}')
        #             merged_state_dict[key] = value
        #     for key, value in second_checkpoint['final_state_dict'].items():
        #         if 'confidence' in key:
        #             print(f'Adding {key}')
        #             merged_final_state_dict[key] = value

            
        #     # overwrite state dicts with merged ones
        #     first_checkpoint['model_state_dict'] = merged_state_dict
        #     first_checkpoint['final_state_dict'] = merged_final_state_dict

        #     #overwrite the optimizer with the second checkpoint's optimizer
        #     # first_checkpoint['optimizer_state_dict'] = second_checkpoint['optimizer_state_dict']
        #     # first_checkpoint['scheduler_state_dict'] = second_checkpoint['scheduler_state_dict']
        #     # first_checkpoint['scaler_state_dict'] = second_checkpoint['scaler_state_dict']
            
        #     # Save the merged checkpoint
        #     torch.save(first_checkpoint, output_file_path)
            
        #     print(f"Merged weights saved to {output_file_path}")

        # first = '/home/tuscant/weights/rf2aa-af3-repro2_240.pt'
        # second = '/home/tuscant/code/af3_pae/RF2-allatom/rf2aa/output/rf2aa-af3-repro-rollout_last.pt'
        # product = '/home/tuscant/weights/240_merged_confidence_5.pt'
        # merge_torch_weights(first, second, product)
        # print('merged weights')
        # print(donemerging)

        # if self.config.training_params.reset_optimizer_params:
        #     #get around a loading issue - I'll only need to do this when loading from a weight set trained on something other than the rollout
        #     # Create a new dictionary to store the modified keys
        #     new_model_state_dict = {}

        #     # Iterate over the original dictionary
        #     for param in self.checkpoint['model_state_dict']:
        #         # Modify the key
        #         new_param = 'model.' + param
        #         # Add the modified key and its value to the new dictionary
        #         new_model_state_dict[new_param] = self.checkpoint['model_state_dict'][param]

        #     # Replace the original dictionary with the new one
        #     self.checkpoint['model_state_dict'] = new_model_state_dict

        #     #Do the same for the final_state_dict
        #     new_model_state_dict = {}
        #     for param in self.checkpoint['final_state_dict']:
        #         # Modify the key
        #         new_param = 'model.' + param
        #         # Add the modified key and its value to the new dictionary
        #         new_model_state_dict[new_param] = self.checkpoint['final_state_dict'][param]
        #     self.checkpoint['final_state_dict'] = new_model_state_dict

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

            #set requires_grad to false for all non confidence parameters
            for name, param in self.model.named_parameters():
                if 'confidence' not in name:
                    param.requires_grad = False
                else:
                    print (f'keeping grads for {name}')

            print(f"Starting training from epoch {start_epoch}")
            #self.valid_epoch(start_epoch-1, rank, world_size)
            for epoch in range(start_epoch,self.config.experiment.n_epoch):
                train_sampler.set_epoch(epoch) #TODO: need to make sure each gpu gets a different example

                # print(f'about to go into valid_epoch')
                # self.valid_epoch(epoch, rank, world_size)
                # print(donevalidating)

                print('about to go into train_epoch for epoch', epoch)
                self.train_epoch(epoch, rank, world_size)
                for _, valid_sampler in valid_samplers.items():
                    valid_sampler.set_epoch(epoch)

                if (
                    self.config.dataset_params.validate_every_n_epochs > 0 
                    and epoch % self.config.dataset_params.validate_every_n_epochs==0
                    and (epoch!=start_epoch or self.config.dataset_params.validate_after_first_epoch)
                ):
                    print('about to go into valid_epoch for epoch', epoch)
                    self.valid_epoch(epoch, rank, world_size)

        self.cleanup()

    def valid_step(self, inputs, n_cycle, no_grads=True, return_outputs=False):
        gpu = self.model.device
        
        example = inputs[0]
        # input_noise_stack = torch.nan_to_num(example["coord_atom_lvl_to_be_noised"]) + example["noise"]
        network_input = {
            #TODO: make a transform that places unresolved ground truth coordinates on their closest real atomshh
            "X_noisy_L": torch.nan_to_num(example["coord_atom_lvl_to_be_noised"]) + example["noise"],
            "t": example["t"],
            "f": example["feats"],
        } 
        loss_input = {
            "X_gt_L": example["ground_truth"]["coord_atom_lvl"][None], #.expand(self.config.dataset_params.diffusion_batch_size, -1,-1),
            "crd_mask_L": example["ground_truth"]["mask_atom_lvl"][None], #.expand(self.config.dataset_params.diffusion_batch_size, -1),
            "X_rep_atoms_I": example["ground_truth"]["coord_token_lvl"],
            "crd_mask_rep_atoms_I": example["ground_truth"]["mask_token_lvl"],
            "interfaces_to_score": example["ground_truth"]["interfaces_to_score"],
            "pn_units_to_score": example["ground_truth"]["pn_units_to_score"],
            "chain_iid_token_lvl": example["ground_truth"]["chain_iid_token_lvl"],
            "example_id": example["example_id"],
            "alignment_mask": example["alignment_mask_atm_lvl"],
            
            #for loss calc
            "seq": example["rf2aa_seq"],
            "atom_frames": example["atom_frames"],
            "tok_idx": example['feats']['atom_to_token_map'],
            "is_real_atom": example['is_real_atom'],
            "rep_atom_idxs": example['ground_truth']['rep_atom_idxs'],
            "frame_atom_idxs": example['frame_atom_idxs'],
        }

        msa_stack = network_input["f"]["msa_stack"]

        interface_mask = torch.zeros(example["rf2aa_seq"].shape[-1], example["rf2aa_seq"].shape[-1], device=gpu, dtype=torch.bool)
        ch_label = example["ground_truth"]["chain_iid_token_lvl"]
        print('example id', example["example_id"])
        for i in range(len(ch_label)):
            for j in range(i+1, len(ch_label)):
                if ch_label[i] != ch_label[j]:
                    interface_mask[i,j] = True
                    interface_mask[j,i] = True
        pred_err = []
        i_pae_err = []

        output_stack = {
            "X_L": [],
            "X_gt_L": [],
            "crd_mask_L": [],
            'plddt': [],
            'pae': [],
            'pde': [],
            'exp_resolved': [],
        }

        for i in range(self.config.dataset_params.trunk_batch_size_valid):
            #get the right msa_stack for this pass
            network_input["f"]["msa_stack"] = msa_stack[(self.config.dataset_params.trunk_batch_size_valid * i):(self.config.dataset_params.trunk_batch_size_valid * i) + 10]

            outputs = self.sampler.sample(inputs, n_cycle=10, use_amp=self.config.training_params.use_amp)

            B = outputs["X_L"].shape[0]
            confidence = self.confidence(
                outputs["S_inputs_I"].repeat(B, 1, 1),
                outputs["S_I"].repeat(B, 1, 1),
                outputs["Z_II"].repeat(B, 1, 1, 1),
                outputs["X_L"],
                # input["f"],
                example["rf2aa_seq"],
                example['ground_truth']['rep_atom_idxs'].to(outputs["X_L"].device),
            )

            
            for i in range(confidence['plddt_logits'].shape[0]):
                plddt_logits = confidence['plddt_logits'][i].unsqueeze(0)
                plddt_logits = plddt_logits.reshape(plddt_logits.shape[0], plddt_logits.shape[1], -1, ChemData().NHEAVY)
                plddt_logits = plddt_logits.permute(0,2,1,3)
                # print('plddt_logits in valid step', plddt_logits.shape)
                # pred_err.append(err)
                # print('err', err)
                plddt, pae, pde = util.unbin_rf3_metrics(plddt_logits.float(), confidence['pae_logits'][i].unsqueeze(0).permute(0,3,1,2).float(), confidence['pde_logits'][i].unsqueeze(0).permute(0,3,1,2).float(), example["rf2aa_seq"].to(plddt_logits.device), is_real_atom=example['is_real_atom'].to(gpu))
                # print('plddt', plddt)
                # print('pae', pae)
                # print('pde', pde)
                pred_err.append({'plddt': plddt, 'pae': pae, 'pde': pde})


                _, i_pae, _ = plddt, pae, pde = util.unbin_rf3_metrics(plddt_logits.float(), confidence['pae_logits'][i].unsqueeze(0).permute(0,3,1,2).float(), confidence['pde_logits'][i].unsqueeze(0).permute(0,3,1,2).float(), example["rf2aa_seq"].to(plddt_logits.device), pae_mask=interface_mask, is_real_atom=example['is_real_atom'].to(gpu))
                i_pae_err.append(i_pae)

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

            output_stack["X_L"].append(outputs["X_L"])
            output_stack["X_gt_L"].append(loss_input["X_gt_L"])
            output_stack["crd_mask_L"].append(loss_input["crd_mask_L"])
            output_stack['plddt'].append(confidence['plddt_logits'])
            output_stack['pae'].append(confidence['pae_logits'])
            output_stack['pde'].append(confidence['pde_logits'])
            output_stack['exp_resolved'].append(confidence['exp_resolved_logits'])


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

        #now cat the outputs that matter for the confidence metrics and loss
        outputs["X_L"] = torch.cat(output_stack["X_L"], dim=0)
        loss_input["X_gt_L"] = torch.cat(output_stack["X_gt_L"], dim=0)
        loss_input["crd_mask_L"] = torch.cat(output_stack["crd_mask_L"], dim=0)
        outputs['plddt'] = torch.cat(output_stack['plddt'], dim=0)
        outputs['pae'] = torch.cat(output_stack['pae'], dim=0)
        outputs['pde'] = torch.cat(output_stack['pde'], dim=0)
        outputs['exp_resolved'] = torch.cat(output_stack['exp_resolved'], dim=0)
        outputs['X_pred_rollout_L'] = outputs['X_L']

        #clear up memory
        del output_stack
        print('B:', outputs['X_L'].shape[0])


        metrics_dict = self.metrics(network_input, outputs, loss_input)
        print(metrics_dict)
        # loss, loss_dict_batched = self.loss(
        #     network_input,
        #     outputs,
        #     loss_input
        # )
        # print('confidence losses', loss_dict_batched)

        return torch.tensor(0), metrics_dict


def convert_allatom_coords_to_residue_coords(X_pred_L, seq):
    """
    Reverse of convert_residue_coords_to_allatom_coords.

    X_pred_L: (B, N_real_atoms, 3) - Coordinates for only real atoms (heavy atoms)
    seq: (B, I) - Sequence tensor indicating residue types

    Returns:
        X_pred: (B, I, natoms, 3) - Full tensor with coordinates for all atoms,
                                    with zeroed entries for non-heavy atoms.
    """
    # Get the mask for real atoms based on the sequenceChemData()
    is_real_atom = ChemData().heavyatom_mask.to(seq.device)[seq]  # (B, I, natoms)
    # print('is_real_atom:', is_real_atom.shape, is_real_atom)

    # Initialize the full tensor with zeros
    B = X_pred_L.shape[0]
    I = seq.shape[-1]
    natoms = is_real_atom.shape[-1]
    X_pred = torch.zeros((B, I, natoms, 3), device=seq.device)

    # Fill in only the "real atom" positions from X_pred_L
    X_pred[:,is_real_atom] = X_pred_L

    return X_pred