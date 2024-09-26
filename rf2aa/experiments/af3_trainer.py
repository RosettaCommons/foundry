import re
import torch
import logging
import tree

from torch.nn.parallel import DistributedDataParallel as DDP
from rf2aa.trainer_new import FlowMatchingTrainer
from rf2aa.model import AF3_structure
from rf2aa.data.dataloader_adaptor_af3 import prepare_input_af3
from rf2aa.data.compose_dataset_datahub import NewDatapipeTrainer
from rf2aa.training.EMA import EMA
from rf2aa.flow_matching.sampler import AF3Sampler, AF3PartialSampler
from rf2aa.loss.af3_losses import Loss as AF3Loss
from rf2aa.debug import pretty_describe_dict


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

        def should_ignore(param_name):
            ignore_regexes = [
                re.compile(r'model\.feature_initializer\.input_feature_embedder\.atom_attention_encoder\.process_s_trunk\..*'),
                re.compile(r'model\.feature_initializer\.input_feature_embedder\.atom_attention_encoder\.process_z\..*'),
                re.compile(r'model\.feature_initializer\.input_feature_embedder\.atom_attention_encoder\.process_r\..*'),
                re.compile(r'model\.feature_initializer\.input_feature_embedder\.atom_attention_encoder\.atom_transformer\.diffusion_transformer\.blocks\.\d+\.attention_pair_bias.ln_1\..*'),
                re.compile(r'model\.recycler\.pairformer_stack\.\d+\.attention_pair_bias\.linear_output_project\..*'),
                re.compile(r'model\.recycler\.pairformer_stack\.\d+\.attention_pair_bias\.ada_ln_1\..*'),
                re.compile(r'model\.diffusion_module\.atom_attention_encoder\.atom_transformer\.diffusion_transformer\.blocks\.\d+\.attention_pair_bias\.ln_1\..*'),
                re.compile(r'model\.diffusion_module\.diffusion_transformer\.blocks\.\d+\.attention_pair_bias\.ln_1\..*'),
                re.compile(r'model\.diffusion_module\.atom_attention_decoder\.atom_transformer\.diffusion_transformer\.blocks\.\d+\.attention_pair_bias\.ln_1\..*'),
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
        assert len(params_to_ignore)

        device_ids = [device]
        if device == 'cpu':
            device_ids = None
        self.model = DDP(self.model, device_ids=device_ids, find_unused_parameters=False, broadcast_buffers=False)
        if "partial_t" in self.config.af3_data_prep:
            self.sampler = AF3PartialSampler(self.config, self.model.module.shadow)
        else:
            self.sampler = AF3Sampler(self.config, self.model.module.shadow)
        self.loss = AF3Loss(**self.config.loss)

    def train_step(self, inputs, n_cycle, no_grads=False, return_outputs=False):
        gpu = self.model.device

        network_input, loss_input = prepare_input_af3(
            inputs,
            **self.config.af3_data_prep,
        )

        network_input=tree.map_structure(lambda x: x.to(gpu) if hasattr(x, 'cpu') else x, network_input)
        loss_input = tree.map_structure(lambda x: x.to(gpu) if hasattr(x, 'cpu') else x, loss_input)
        logger.debug('network_input:\n' + pretty_describe_dict(network_input))
        logger.debug('loss_input:\n' + pretty_describe_dict(loss_input))

        output_i = self.model(
            network_input,
            n_cycle,
            no_sync=self.model.no_sync,
        )

        loss, loss_dict_batched = self.loss(
            network_input,
            output_i,
            loss_input
        )
        loss_dict = {}
        output = {"X_L": output_i["X_L"]} | network_input | loss_input
        from rf2aa.callbacks import lddt_metrics

        lddt, _ = lddt_metrics(None, output)
        sigma_data = 16
        t = network_input['t']
        X_noisy_L = network_input['X_noisy_L']
        null_pred = (sigma_data**2 / (sigma_data**2 + t**2))[...,None,None] * X_noisy_L
        lddt_null, _ = lddt_metrics(None, {"X_L": null_pred} | network_input | loss_input)
        for key, val in lddt_null.items():
            lddt[f"{key}_null_pred"] = val
        loss_dict_batched["noise_std_dev"]  = torch.std(X_noisy_L, dim=(-1,-2))
        loss_dict_batched = loss_dict_batched | lddt 
        loss_dict_batched["t"] = network_input['t']
        loss_dict.update(self.unbatch_losses(loss_dict_batched))

        #self.write_pdb(
            #loss_input["X_gt_L"][0], 
            #loss_input["crd_mask_I"], 
            #loss_input["seq"],
            #name=f"train_{inputs['item']['CHAINID']}.pdb"
        #)

        return loss, loss_dict

    def valid_step(self, inputs, n_cycle, no_grads=True, return_outputs=False):
        gpu = self.model.device
        n_cycle = 4
        outputs = self.sampler.sample(inputs, n_cycle=n_cycle, use_amp=self.config.training_params.use_amp)
        
        X_L = outputs['X_L']

        network_input, loss_input = prepare_input_af3(
            inputs,
            **self.config.af3_data_prep,
        )
        t = self.sampler.construct_noise_schedule(200, 0, 1)[0] if "partial_t" not in self.config.af3_data_prep else self.config.af3_data_prep.partial_t
        network_input['t'] = torch.tensor(t).tile(self.config.af3_data_prep.D).to(gpu)
        network_input=tree.map_structure(lambda x: x.to(gpu) if hasattr(x, 'cpu') else x, network_input)
        loss_input=tree.map_structure(lambda x: x.to(gpu) if hasattr(x, 'cpu') else x, loss_input)

        loss, loss_dict_batched = self.loss(
            network_input,
            outputs,
            loss_input
        )
        output = {"X_L": X_L} | network_input | loss_input
        
        loss_dict = {}
        lddt, _ = lddt_metrics(None, output)
        agg_lddt = agg_lddt | lddt
        loss_dict_batched = loss_dict_batched | agg_lddt 
        loss_dict.update(self.unbatch_losses(loss_dict_batched))
        loss_dict = tree.map_structure(lambda x: torch.tensor(x) if not torch.is_tensor(x) else x, loss_dict)
        loss_dict = tree.map_structure(lambda x: x.to(gpu), loss_dict)
        import sys
        self.log_validation_losses(inputs["item"]["CHAINID"], loss_dict)
        sys.stdout.flush()
        return loss, loss_dict
    
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
        from rf2aa.flow_matching.sampler import AF3Sampler
        self.sampler = AF3Sampler(self.config, model)
        from rf2aa.model.af3_with_rollout import AF3_with_rollout 
        import copy
        self.model = AF3_with_rollout(
            model,
            confidence,
            copy.deepcopy(self.sampler)
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
        assert len(params_to_ignore)

        self.model = DDP(self.model, device_ids=[device], find_unused_parameters=False, broadcast_buffers=False)
        self.sampler.model = self.model.module.shadow.model
        self.loss = AF3Loss(**self.config.loss)

    def train_step(self, inputs, n_cycle, no_grads=False, return_outputs=False):
        gpu = self.model.device

        network_input, loss_input = prepare_input_af3(
            inputs,
            **self.config.af3_data_prep,
        )

        network_input=tree.map_structure(lambda x: x.to(gpu) if hasattr(x, 'cpu') else x, network_input)
        loss_input = tree.map_structure(lambda x: x.to(gpu) if hasattr(x, 'cpu') else x, loss_input)
        logger.debug('network_input:\n' + pretty_describe_dict(network_input))
        logger.debug('loss_input:\n' + pretty_describe_dict(loss_input))
        output_i = self.model(
            network_input,
            n_cycle,
            loss_input["X_gt_I_symm"].to(gpu),
            loss_input["crd_mask_I_symm"].to(gpu),
            loss_input["seq"].to(gpu),
            no_sync=self.model.no_sync,
        )

        loss, loss_dict_batched = self.loss(
            network_input,
            output_i,
            loss_input
        )
        loss_dict = {}
        output = {"X_L": output_i["X_L"]} | network_input | loss_input
        from rf2aa.callbacks import lddt_metrics

        lddt, _ = lddt_metrics(None, output)
        sigma_data = self.config.af3_data_prep.sigma_data
        t = network_input['t']
        X_noisy_L = network_input['X_noisy_L']
        null_pred = (sigma_data**2 / (sigma_data**2 + t**2))[...,None,None] * X_noisy_L
        lddt_null, _ = lddt_metrics(None, {"X_L": null_pred} | network_input | loss_input)
        for key, val in lddt_null.items():
            lddt[f"{key}_null_pred"] = val
        loss_dict_batched["noise_std_dev"]  = torch.std(X_noisy_L, dim=(-1,-2))
        loss_dict_batched = loss_dict_batched | lddt 
        loss_dict_batched["t"] = network_input['t']
        loss_dict.update(self.unbatch_losses(loss_dict_batched))
        return loss, loss_dict
