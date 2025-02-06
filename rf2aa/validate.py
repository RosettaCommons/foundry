import os
from functools import partial

import hydra
import pandas as pd
import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP

from rf2aa.chemical import initialize_chemdata
from rf2aa.data.compose_dataset import compose_posebusters
from rf2aa.data.dataloader_adaptor import get_loss_calc_items
from rf2aa.trainer_new import trainer_factory
from rf2aa.util import writepdb


class PoseBustersBenchmark:
    def __init__(self, config):
        # config file logic for validation, low->high prio:
        # 1) use default parameters in config/train/base.yml
        # 2) use parameters saved in model
        # 3) use specific params in config/inference
        default_config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "config/train/base.yaml"
        )
        base_config = OmegaConf.load(default_config_path)
        tmp_data = torch.load(config.eval_params.checkpoint_path, mmap=True)
        if "training_config" in tmp_data:
            train_config = tmp_data["training_config"]
            self.config = OmegaConf.merge(base_config, train_config, config)
        else:
            self.config = OmegaConf.merge(base_config, config)
        tmp_data = None

        assert self.config.ddp_params.batch_size == 1, "batch size is assumed to be 1"
        if self.config.experiment.output_dir is not None:
            self.output_dir = self.config.experiment.output_dir
        else:
            self.output_dir = "output/"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.trainer = trainer_factory[self.config.experiment.trainer](
            config=self.config
        )

    def construct_dataset(self, rank, world_size):
        # fd initialize chemical data based on input arguments
        #   this needs to be initialized first
        init = partial(initialize_chemdata, self.config)
        init()

        return compose_posebusters(init, self.config.loader_params, rank, world_size)

    def launch_distributed_eval(self):
        world_size = torch.cuda.device_count()
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = (
                "127.0.0.1"  # multinode requires this set in submit script
            )
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "%d" % self.config.ddp_params.port

        world_size = torch.cuda.device_count()

        if world_size == 0:
            print("Error! No GPUs found!")
        elif world_size == 1:
            # No need for multiple processes with 1 GPU
            self.evaluate_model(0, world_size)
        else:
            mp.spawn(
                self.evaluate_model, args=(world_size,), nprocs=world_size, join=True
            )

    def evaluate_model(self, rank, world_size):
        gpu = self.trainer.init_process_group(rank, world_size)
        benchmark_loader = self.construct_dataset(rank, world_size)

        # move global information to device
        self.trainer.move_constants_to_device(gpu)

        self.trainer.construct_model(device=gpu)
        self.trainer.model = DDP(
            self.trainer.model,
            device_ids=[gpu],
            find_unused_parameters=False,
            broadcast_buffers=False,
        )

        self.trainer.load_checkpoint(rank)
        self.trainer.load_model()
        self.trainer.model.eval()
        records = []
        for inputs in benchmark_loader:
            item = inputs[-1]
            with torch.no_grad():
                loss, loss_dict, outputs = self.trainer.train_step(
                    inputs,
                    self.config.loader_params.maxcycle,
                    nograds=True,
                    return_outputs=True,
                )
            loss_dict["CHAINID"] = item["CHAINID"][0]
            for k, v in loss_dict.items():
                if torch.is_tensor(v):
                    loss_dict[k] = v.item()
            records.append(loss_dict)
            df = pd.DataFrame(records)
            df.to_csv(
                f"{self.output_dir}/{self.config.experiment.name}_{rank}_posebusters.csv"
            )
            torch.cuda.empty_cache()

            true_crds = inputs[5]
            seq, _, idx_pdb, bond_feats, _, _ = get_loss_calc_items(inputs, device=gpu)
            pred_crds, alphas, pred_lddts = outputs[5], outputs[6], outputs[8]
            _, pred_allatom = self.trainer.xyz_converter.compute_all_atom(
                seq[:, 0], pred_crds[-1], alphas[-1]
            )

            writepdb(
                f"{self.output_dir}/{item['CHAINID'][0]}_nat.pdb",
                true_crds[:, 0],
                seq[:, 0].long(),
                bond_feats=bond_feats,
            )
            writepdb(
                f"{self.output_dir}/{item['CHAINID'][0]}_pred.pdb",
                pred_allatom[0],
                seq[:, 0].long(),
                bond_feats=bond_feats,
            )


@hydra.main(version_base=None, config_path="config/inference")
def main(config):
    benchmarker = PoseBustersBenchmark(config=config)
    benchmarker.launch_distributed_eval()


if __name__ == "__main__":
    main()
