import datetime
import logging
import os
import subprocess
import time
import warnings
from contextlib import nullcontext
from datetime import timedelta
from functools import partial

import numpy as np
import omegaconf
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
from hydra.utils import instantiate
from icecream import ic
from torch.nn.parallel import DistributedDataParallel as DDP

from rf2aa.chemical import ChemicalData as ChemData
from rf2aa.chemical import initialize_chemdata
from rf2aa.debug import (
    debug_grads,
)
from rf2aa.flow_matching.interpolant import Interpolant
from rf2aa.flow_matching.sampler import AllAtomSampler
from rf2aa.loss.loss_factory import get_loss_and_misc
from rf2aa.model.network import RosettaFold
from rf2aa.model.RoseTTAFoldModel import LegacyRoseTTAFoldModule
from rf2aa.training.EMA import EMA, count_parameters
from rf2aa.training.optimizer import add_weight_decay
from rf2aa.training.recycling import (
    recycle_sampling,
    recycle_step_gen,
    recycle_step_legacy,
    recycle_step_packed,
)
from rf2aa.training.scheduler import get_stepwise_decay_schedule_with_warmup
from rf2aa.util_module import XYZConverter

logger = logging.getLogger(__name__)

# TODO: control environment variables from config
# limit thread counts
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:512"
# Update environment variable with correct path (needed for W&B upload)
# os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
## To reproduce errors

torch.set_num_threads(4)
# torch.autograd.set_detect_anomaly(True)


class Trainer:
    def __init__(self, config) -> None:
        self.config = config

        assert self.config.ddp_params.batch_size == 1, "batch size is assumed to be 1"
        if self.config.experiment.output_dir is not None:
            self.output_dir = self.config.experiment.output_dir
        else:
            self.output_dir = "models/"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # commit, diff = self.record_git_commit()
        commit, diff = None, None
        self.commit, self.diff = commit, diff

        self.dataset_constructor = instantiate(self.config.dataset_params.constructor)
        print(f"Using dataset constructor: {self.dataset_constructor}")

    def construct_model(self, device):
        raise NotImplementedError()

    def construct_optimizer(self):
        if self.config.training_params.weight_decay is not None:
            opt_params = add_weight_decay(
                self.model, self.config.training_params.weight_decay
            )
        else:
            opt_params = self.model.parameters()
        self.optimizer = torch.optim.AdamW(
            opt_params, lr=self.config.training_params.learning_rate
        )

    def construct_scheduler(self):
        self.scheduler = get_stepwise_decay_schedule_with_warmup(
            self.optimizer, **self.config.training_params.learning_rate_schedule
        )

    def construct_scaler(self):
        self.scaler = torch.cuda.amp.GradScaler(
            "cuda", enabled=self.config.training_params.use_amp
        )

    def load_checkpoint(self, rank):
        # if self.config.training_params.from_scratch:
        #    return False
        checkpoint_path = f"{self.output_dir}/{self.config.experiment.name}_last.pt"
        # 'checkpoint_path' takes priority ...
        if self.config.eval_params.checkpoint_path:
            checkpoint_path = self.config.eval_params.checkpoint_path
        # ... followed by 'resume_from_checkpoint_path'
        elif self.config.training_params.resume_from_checkpoint_path:
            checkpoint_path = self.config.training_params.resume_from_checkpoint_path

        # check if checkpoint path is real
        if not os.path.exists(checkpoint_path):
            warnings.warn(
                f"{checkpoint_path} not found, continuing with random parameters"
            )
            return False
        map_location = f"cuda:{rank}"

        print(f"Loading checkpoint from {checkpoint_path} on rank:{rank}")
        self.checkpoint = torch.load(
            checkpoint_path, map_location=map_location, weights_only=False
        )
        return True

    def load_model(self):
        torch.cuda.empty_cache()

        new_model_state = {}
        new_shadow_state = {}
        state_dict = self.model.module.model.state_dict()
        for param in state_dict:
            if param not in self.checkpoint["model_state_dict"]:
                print("missing", param)
            elif (
                self.checkpoint["model_state_dict"][param].shape
                == state_dict[param].shape
            ):
                new_model_state[param] = self.checkpoint["final_state_dict"][param]
                new_shadow_state[param] = self.checkpoint["model_state_dict"][param]
            else:
                print(
                    "wrong size",
                    param,
                    self.checkpoint["model_state_dict"][param].shape,
                    state_dict[param].shape,
                )

        self.model.module.model.load_state_dict(new_model_state, strict=False)
        self.model.module.shadow.load_state_dict(new_shadow_state, strict=False)
        print("Checkpoint loaded into model")

    def load_optimizer(self):
        self.optimizer.load_state_dict(self.checkpoint["optimizer_state_dict"])

    def load_scheduler(self):
        self.scheduler.load_state_dict(self.checkpoint["scheduler_state_dict"])

    def load_scaler(self):
        self.scaler.load_state_dict(self.checkpoint["scaler_state_dict"])

    def construct_dataset(self, init_db, rank, world_size):
        return self.dataset_constructor(
            init_db,
            self.config.dataset_params,
            self.config.loader_params,
            rank,
            world_size,
        )
        # return compose_dataset(
        # init_db, self.config.dataset_params, self.config.loader_params, rank, world_size
        # )

    def construct_loss_function(self):
        raise NotImplementedError()

    def move_constants_to_device(self, gpu):
        self.fi_dev = ChemData().frame_indices.to(gpu)
        self.xyz_converter = XYZConverter().to(gpu)

        self.l2a = ChemData().long2alt.to(gpu)
        self.aamask = ChemData().allatom_mask.to(gpu)
        self.num_bonds = ChemData().num_bonds.to(gpu)
        self.atom_type_index = ChemData().atom_type_index.to(gpu)
        self.ljlk_parameters = ChemData().ljlk_parameters.to(gpu)
        self.lj_correction_parameters = ChemData().lj_correction_parameters.to(gpu)
        self.hbtypes = ChemData().hbtypes.to(gpu)
        self.hbbaseatoms = ChemData().hbbaseatoms.to(gpu)
        self.hbpolys = ChemData().hbpolys.to(gpu)
        self.cb_len = ChemData().cb_length_t.to(gpu)
        self.cb_ang = ChemData().cb_angle_t.to(gpu)
        self.cb_tor = ChemData().cb_torsion_t.to(gpu)

    def checkpoint_model(self, epoch, metadata={}):
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": self.model.module.shadow.state_dict(),
            "final_state_dict": self.model.module.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "training_config": dict(self.config),
            "commit": self.commit,
            "diff": self.diff,
        }
        checkpoint_data.update(metadata)

        if epoch < 0:
            torch.save(
                checkpoint_data,
                f"{self.output_dir}/{self.config.experiment.name}_error.pt",
            )
        else:
            torch.save(
                checkpoint_data,
                f"{self.output_dir}/{self.config.experiment.name}_last.pt",
            )
            if epoch % self.config.log_params.checkpoint_every_n_epochs == 0:
                torch.save(
                    checkpoint_data,
                    f"{self.output_dir}/{self.config.experiment.name}_{epoch}.pt",
                )

    def launch_distributed_training(self):
        world_size = torch.cuda.device_count()
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = (
                "127.0.0.1"  # multinode requires this set in submit script
            )
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "%d" % self.config.ddp_params.port

        if "SLURM_NTASKS" in os.environ and "SLURM_PROCID" in os.environ:
            world_size = int(os.environ["SLURM_NTASKS"])
            rank = int(os.environ["SLURM_PROCID"])
            print("Launched from slurm", rank, world_size)

            self.train_model(rank, world_size)

        else:
            print("Launched from interactive")
            world_size = torch.cuda.device_count()

            if world_size == 0:
                print("Error! No GPUs found!")
            elif world_size == 1:
                # No need for multiple processes with 1 GPU
                self.train_model(0, world_size)
            else:
                mp.spawn(
                    self.train_model, args=(world_size,), nprocs=world_size, join=True
                )

    def record_git_commit(self, path=None):
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # git hash of current commit
        try:
            commit = (
                subprocess.check_output(
                    f"git --git-dir {script_dir}/../.git rev-parse HEAD", shell=True
                )
                .decode()
                .strip()
            )
        except subprocess.CalledProcessError:
            print("WARNING: Failed to determine git commit hash.")
            commit = "unknown"

        # save git diff from last commit
        git_diff = subprocess.Popen(
            ["git diff"],
            cwd=os.getcwd(),
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        out, err = git_diff.communicate()

        return commit, out

    def init_process_group(self, rank, world_size):
        gpu = rank % torch.cuda.device_count()
        dist.init_process_group(
            backend=self.config.training_params.ddp_backend,
            timeout=timedelta(seconds=3600),
            world_size=world_size,
            rank=rank,
        )
        torch.cuda.set_device("cuda:%d" % gpu)
        # device = 'cpu'
        # if torch.cuda.device_count():
        #    device = "cuda:%d"%gpu
        #    torch.cuda.set_device(device)
        return gpu

    def cleanup(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    def train_model(self, rank, world_size):
        """runs model training on each gpu"""
        gpu = self.init_process_group(rank, world_size)
        # rank = gpu
        ic(rank, world_size, gpu)

        # fd initialize chemical data based on input arguments
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
                name=f"{self.config.experiment.name}_{timestamp}",
            )
        else:
            context_manager = nullcontext()  # Does nothing

        # Without W&B, context manager does nothing
        with context_manager:
            train_loader, train_sampler, valid_loaders, valid_samplers = (
                self.construct_dataset(init, rank, world_size)
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
            logger.info(f"Loaded checkpoint: {loaded_checkpoint}")
            if loaded_checkpoint:
                start_epoch = self.checkpoint["epoch"]
                self.load_model()
                if not self.config.training_params.reset_optimizer_params:
                    self.load_optimizer()
                    self.load_scheduler()
                    self.load_scaler()
                else:
                    warnings.warn(
                        "User specified reset_optimizer_params=False. Did not load optimizer values from checkpoint"
                    )
            self.checkpoint = None  # unload checkpoint dict

            self.recycle_schedule = recycle_sampling["by_batch"](
                self.config.loader_params.maxcycle,
                self.config.experiment.n_epoch,
                self.config.dataset_params.n_train,
                world_size,
            )

            print(f"Starting training from epoch {start_epoch}")

            if self.config.experiment.prevalidate:
                logger.info(
                    "Prevalidating checkpoint, if you want to directly start training set config.experiment.prevalidate=False"
                )
                self.valid_epoch(start_epoch - 1, rank, world_size)

            for epoch in range(start_epoch, self.config.experiment.n_epoch):
                train_sampler.set_epoch(
                    epoch
                )  # TODO: need to make sure each gpu gets a different example
                self.train_epoch(epoch, rank, world_size)
                for _, valid_sampler in valid_samplers.items():
                    valid_sampler.set_epoch(epoch)

                if (
                    self.config.dataset_params.validate_every_n_epochs > 0
                    and epoch % self.config.dataset_params.validate_every_n_epochs == 0
                    and (
                        epoch != start_epoch
                        or self.config.dataset_params.validate_after_first_epoch
                    )
                ):
                    self.valid_epoch(epoch, rank, world_size)

        self.cleanup()

    def train_epoch(self, epoch, rank, world_size):
        """train model"""
        # turn on gradients
        self.model.train()

        # clear gradients
        self.optimizer.zero_grad()
        start_time = time.time()
        if len(self.train_loader) == 0:
            return

        for train_idx, inputs in enumerate(self.train_loader):
            n_cycle = self.recycle_schedule[epoch, train_idx]  # number of recycling
            # run forward pass and compute loss
            loss, loss_dict = self.train_step(inputs, n_cycle)
            # aggregate loss and update parameters
            loss = loss / self.config.ddp_params.accum

            try:
                self.scaler.scale(loss).backward()
            except Exception as e:
                print("Backwards error in", inputs[0]["example_id"])
                raise e

            find_no_grad_parameters = False
            if find_no_grad_parameters:
                no_grad_parameters = []
                for n, p in self.model.module.model.named_parameters():
                    if p.grad is None:
                        no_grad_parameters.append(n)

                if no_grad_parameters:
                    print("Parameters with grad == None:")
                    for n in no_grad_parameters:
                        print(n)
                    print(
                        f"Fraction with grad == None: {len(no_grad_parameters)}/{len(list(self.model.module.model.named_parameters()))}"
                    )

            train_time = time.time() - start_time
            if (train_idx) % self.config.ddp_params.accum == 0:
                self.update_parameters()
                torch.cuda.empty_cache()

            if (
                train_idx % self.config.log_params.log_every_n_examples == 0
                and rank == 0
            ):
                train_time = time.time() - start_time
                self.log_intermediate_losses(
                    inputs,
                    loss_dict,
                    n_cycle,
                    (train_idx + 1) * world_size,
                    len(self.train_loader) * world_size,
                    train_time,
                )

                # If using W&B, log the intermediate losses (note: this is only done for rank = 0)
                if self.config.log_params.use_wandb:
                    wandb.log(loss_dict)

        if rank == 0:
            self.checkpoint_model(epoch)

    def valid_epoch(self, epoch, rank, world_size):
        """validate model"""
        # turn on gradients
        self.model.eval()

        for dataset_name, valid_loader in self.valid_loaders.items():
            valid_loss_dict = None
            for valid_idx, inputs in enumerate(valid_loader):
                n_cycle = self.config.loader_params.maxcycle

                # fd We could make this a separate function call?
                loss, loss_dict = self.train_step(inputs, n_cycle, nograds=True)

                if valid_loss_dict is None:
                    valid_loss_dict = torch.zeros_like(
                        torch.stack(list(loss_dict.values()))
                    )
                valid_loss_dict += torch.stack(list(loss_dict.values()))

            if len(valid_loader) == 0:
                continue

            valid_loss_dict /= float(len(valid_loader) * world_size)
            dist.all_reduce(valid_loss_dict, op=dist.ReduceOp.SUM)

            # reconstruct loss dictionary
            dict_keys = list(loss_dict.keys())
            valid_loss_dict = {
                dict_keys[i]: valid_loss_dict[i]
                for i in range(valid_loss_dict.shape[0])
            }

            if rank == 0:
                self.log_validation_losses(dataset_name, valid_loss_dict)
                # If using W&B, log the validation losses (note: this is only done for rank = 0)
                if self.config.log_params.use_wandb:
                    wandb.log(valid_loss_dict)

    def train_step(self, inputs, n_cycle):
        """take an input from dataloader, run the model and compute a loss"""
        raise NotImplementedError()

    def valid_step(self, inputs, n_cycle):
        """take an input from dataloader, run the model and compute a loss.  No grads/checkpointing"""
        raise NotImplementedError()

    def update_parameters(self):
        """scale, clip gradients and update parameters"""
        # gradient clipping
        if self.config.debug_params.debug_grads:
            debug_grads(self.model)
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.training_params.grad_clip
        )
        self.scaler.step(self.optimizer)
        scale = self.scaler.get_scale()
        self.scaler.update()
        skip_lr_sched = scale != self.scaler.get_scale()
        self.optimizer.zero_grad()
        if not skip_lr_sched:
            self.scheduler.step()
        self.model.module.update()  # apply EMA

    def log_intermediate_losses(self, inputs, loss_dict, n_cycle, Nex, Nepoch, runtime):
        if type(inputs) == tuple:
            item = inputs[-1]
        elif type(inputs) == list:
            item = inputs[0]["example_id"]
        else:
            item = inputs["item"]
        max_mem = torch.cuda.max_memory_allocated() / 1e9
        print(
            f"Models: {Nex} of: {Nepoch} Max_Memory: {max_mem:.4f}Gb Runtime: {runtime:.4f}"
        )
        print(
            f"Example: {item} Recycle:{n_cycle}\n"
            + "\t".join([f"{k}: {v:.4f}" for k, v in loss_dict.items()])
        )
        import sys

        sys.stdout.flush()
        # print(f"Models: {Nex} Example: {item['CHAINID']}"+" ".join([f"{k}: {v:.4f}" for k,v in loss_dict.items()]))
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def log_validation_losses(self, dataset_name, loss_dict):
        print(
            f"Dataset: {dataset_name} "
            + "\t".join([f"{k}:{v:.4f}" for k, v in loss_dict.items()])
        )


class LegacyTrainer(Trainer):
    """trains Legacy versions of RFAA"""

    def __init__(self, config) -> None:
        super().__init__(config)

    def construct_model(self, device="cpu"):
        self.model = LegacyRoseTTAFoldModule(
            **self.config.legacy_model_param,
            aamask=ChemData().allatom_mask.to(device),
            atom_type_index=ChemData().atom_type_index.to(device),
            ljlk_parameters=ChemData().ljlk_parameters.to(device),
            lj_correction_parameters=ChemData().lj_correction_parameters.to(device),
            num_bonds=ChemData().num_bonds.to(device),
            cb_len=ChemData().cb_length_t.to(device),
            cb_ang=ChemData().cb_angle_t.to(device),
            cb_tor=ChemData().cb_torsion_t.to(device),
        ).to(device)
        device_ids = [device]
        if device == "cpu":
            device_ids = None
        if self.config.training_params.EMA is not None:
            self.model = EMA(self.model, self.config.training_params.EMA)
        self.model = DDP(
            self.model,
            device_ids=device_ids,
            find_unused_parameters=False,
            broadcast_buffers=False,
        )

    def train_step(self, inputs, n_cycle, nograds=False, return_outputs=False):
        """take an input from dataloader, run the model and compute a loss"""
        gpu = self.model.device
        # HACK: certain features are constructed during the train step
        # in the future this should only promote the constructed features onto gpu
        (
            task,
            item,
            network_input,
            true_crds,
            atom_mask,
            msa,
            mask_msa,
            unclamp,
            negative,
            symmRs,
            Lasu,
            ch_label,
        ) = prepare_input(inputs, self.xyz_converter, gpu)

        output_i = recycle_step_legacy(
            self.model,
            network_input,
            n_cycle,
            self.config.training_params.use_amp,
            nograds=nograds,
        )
        seq, same_chain, idx_pdb, bond_feats, dist_matrix, atom_frames, _, _ = (
            get_loss_calc_items(inputs, device=gpu)
        )

        # HACK: indexing into msa and mask msa recycle dimension in arguments of this function
        # HACK: need to promote some inputs to gpu for loss calculation, all promotions should happen together
        msa = msa.to(gpu)
        mask_msa = mask_msa.to(gpu)
        loss, loss_dict = get_loss_and_misc(
            self,  # avoid reloading constants to device
            output_i,
            true_crds,
            atom_mask,
            same_chain,
            seq,
            msa[:, n_cycle - 1],
            mask_msa[:, n_cycle - 1],
            idx_pdb,
            bond_feats,
            dist_matrix,
            atom_frames,
            None,
            None,
            unclamp,
            negative,
            task,
            item,
            symmRs,
            Lasu,
            ch_label,
            self.config.loss_param,
        )
        if return_outputs:
            return loss, loss_dict, output_i
        else:
            return loss, loss_dict


class ComposedTrainer(Trainer):
    """trains composed versions of RFAA"""

    def __init__(self, config) -> None:
        super().__init__(config)

    def construct_model(self, device="cpu"):
        self.model = RosettaFold(self.config).to(device)
        if self.config.training_params.EMA is not None:
            self.model = EMA(self.model, self.config.training_params.EMA)
        self.model = DDP(
            self.model,
            device_ids=[device],
            find_unused_parameters=False,
            broadcast_buffers=False,
        )

    def train_step(self, inputs, n_cycle, nograds=False, return_outputs=False):
        """take an input from dataloader, run the model and compute a loss"""
        gpu = self.model.device
        # HACK: certain features are constructed during the train step
        # in the future this should only promote the constructed features onto gpu
        (
            task,
            item,
            network_input,
            true_crds,
            atom_mask,
            msa,
            mask_msa,
            unclamp,
            negative,
            symmRs,
            Lasu,
            ch_label,
        ) = prepare_input(inputs, self.xyz_converter, gpu)

        output_i = recycle_step_packed(
            self.model,
            network_input,
            n_cycle,
            self.config.training_params.use_amp,
            nograds=nograds,
        )
        (
            seq,
            same_chain,
            idx_pdb,
            bond_feats,
            dist_matrix,
            atom_frames,
            true_crds,
            mask_crds,
        ) = get_loss_calc_items(inputs, device=gpu)

        # HACK: indexing into msa and mask msa recycle dimension in arguments of this function
        # HACK: need to promote some inputs to gpu for loss calculation, all promotions should happen together
        msa = msa.to(gpu)
        mask_msa = mask_msa.to(gpu)

        loss, loss_dict = get_loss_and_misc(
            self,  # avoid reloading constants to device
            output_i,
            true_crds,
            atom_mask,
            same_chain,
            seq,
            msa[:, n_cycle - 1],
            mask_msa[:, n_cycle - 1],
            idx_pdb,
            bond_feats,
            dist_matrix,
            atom_frames,
            None,
            None,
            unclamp,
            negative,
            task,
            item,
            symmRs,
            Lasu,
            ch_label,
            self.config.loss_param,
        )

        if return_outputs:
            return loss, loss_dict, output_i
        else:
            return loss, loss_dict


class FlowMatchingTrainer(Trainer):
    def construct_model(self, device="cpu"):
        self.model = RosettaFold(self.config).to(device)
        if self.config.training_params.EMA is not None:
            self.model = EMA(self.model, self.config.training_params.EMA)
        self.model = DDP(
            self.model,
            device_ids=[device],
            find_unused_parameters=False,
            broadcast_buffers=False,
        )
        self.sampler = AllAtomSampler(
            self.model,
            self.config.interpolant.sampling.num_timesteps,
            self.config.interpolant.min_t,
            self.interpolant,
            self.xyz_converter,
            is_training=True,
        )

    def move_constants_to_device(self, gpu):
        self.interpolant = Interpolant(self.config.interpolant)
        self.interpolant.set_device(gpu)
        super().move_constants_to_device(gpu)

    def train_step(self, inputs, n_cycle, no_grads=False, return_outputs=False):
        gpu = self.model.device
        # try:
        (
            task,
            item,
            network_input,
            true_crds,
            atom_mask,
            msa,
            mask_msa,
            unclamp,
            negative,
            symmRs,
            Lasu,
            ch_label,
            r3_t,
            trans_1,
            mask_allatom,
        ) = prepare_input_fm_allatom(inputs, self.interpolant, self.xyz_converter, gpu)

        output_i = recycle_step_gen(
            self.model,
            network_input,
            n_cycle,
            self.config.training_params.use_amp,
            nograds=no_grads,
        )
        (
            seq,
            same_chain,
            idx_pdb,
            bond_feats,
            dist_matrix,
            atom_frames,
            true_crds,
            atom_mask,
        ) = get_loss_calc_items(inputs, device=gpu)
        (
            logit_s,
            logit_aa_s,
            logit_pae,
            logit_pde,
            p_bind,
            pred_crds,
            alphas,
            pred_allatom,
            pred_lddts,
            _,
            _,
            _,
        ) = output_i
        # loss = (pred_allatom - true_crds).mean()
        # loss_dict = {"loss": loss.mean()}

        # HACK: indexing into msa and mask msa recycle dimension in arguments of this function
        # HACK: need to promote some inputs to gpu for loss calculation, all promotions should happen together
        msa = msa.to(gpu)
        mask_msa = mask_msa.to(gpu)

        loss, loss_dict = get_loss_and_misc(
            self,  # avoid reloading constants to device
            output_i,
            true_crds,
            atom_mask,
            same_chain,
            seq,
            msa[:, n_cycle - 1],
            mask_msa[:, n_cycle - 1],
            idx_pdb,
            bond_feats,
            dist_matrix,
            atom_frames,
            trans_1,
            r3_t,
            unclamp,
            negative,
            task,
            item,
            symmRs,
            Lasu,
            ch_label,
            self.config.loss_param,
        )

        if return_outputs:
            return loss, loss_dict, output_i
        else:
            return loss, loss_dict

    def valid_step(self, inputs, n_cycle, no_grads=True, return_outputs=False):
        gpu = self.model.device
        # try:
        (
            task,
            item,
            network_input,
            true_crds,
            atom_mask,
            msa,
            mask_msa,
            unclamp,
            negative,
            symmRs,
            Lasu,
            ch_label,
            r3_t,
            trans_1,
            mask_allatom,
        ) = prepare_input_fm_allatom(inputs, self.interpolant, self.xyz_converter, gpu)

        # output_i = recycle_step_gen(self.model, network_input, n_cycle, self.config.training_params.use_amp, nograds=no_grads)
        with torch.no_grad():
            output_i = self.sampler.sample(
                inputs, n_cycle=n_cycle, use_amp=self.config.training_params.use_amp
            )
        (
            seq,
            same_chain,
            idx_pdb,
            bond_feats,
            dist_matrix,
            atom_frames,
            true_crds,
            atom_mask,
        ) = get_loss_calc_items(inputs, device=gpu)

        # HACK: indexing into msa and mask msa recycle dimension in arguments of this function
        # HACK: need to promote some inputs to gpu for loss calculation, all promotions should happen together
        msa = msa.to(gpu)
        mask_msa = mask_msa.to(gpu)

        loss, loss_dict = get_loss_and_misc(
            self,  # avoid reloading constants to device
            output_i,
            true_crds,
            atom_mask,
            same_chain,
            seq,
            msa[:, n_cycle - 1],
            mask_msa[:, n_cycle - 1],
            idx_pdb,
            bond_feats,
            dist_matrix,
            atom_frames,
            trans_1,
            r3_t,
            unclamp,
            negative,
            task,
            item,
            symmRs,
            Lasu,
            ch_label,
            self.config.loss_param,
        )

        # fd last layer l0 are unused in grads
        # fd to do: fix this in refinement module
        loss += 0.0 * output_i[-1].sum()

        if return_outputs:
            return loss, loss_dict, output_i
        else:
            return loss, loss_dict

    def valid_epoch(self, epoch, rank, world_size):
        """validate model"""
        # turn on gradients
        self.model.eval()
        for dataset_name, valid_loader in self.valid_loaders.items():
            valid_loss_dict = None
            for valid_idx, inputs in enumerate(valid_loader):
                n_cycle = self.config.loader_params.maxcycle

                loss, loss_dict = self.valid_step(inputs, n_cycle)
                if valid_loss_dict is None:
                    valid_loss_dict = torch.zeros_like(
                        torch.stack(list(loss_dict.values()))
                    )
                valid_loss_dict += torch.stack(list(loss_dict.values()))

            if len(valid_loader) == 0:
                continue

            valid_loss_dict /= float(len(valid_loader) * world_size)
            dist.all_reduce(valid_loss_dict, op=dist.ReduceOp.SUM)

            # reconstruct loss dictionary
            dict_keys = list(loss_dict.keys())
            valid_loss_dict = {
                dict_keys[i]: valid_loss_dict[i]
                for i in range(valid_loss_dict.shape[0])
            }

            if rank == 0:
                self.log_validation_losses(dataset_name, valid_loss_dict)
                # If using W&B, log the validation losses (note: this is only done for rank = 0)
                if self.config.log_params.use_wandb:
                    wandb.log(valid_loss_dict)


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def get_param_sizes(model):
    o = {}
    for k, p in model.named_parameters():
        o[k] = (np.array(p.size()).prod(), p.size())
    return o


if __name__ == "__main__":
    main()
