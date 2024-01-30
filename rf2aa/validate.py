import torch
import os
import hydra
import pandas as pd
from torch.nn.parallel import DistributedDataParallel as DDP

from rf2aa.trainer_new import LegacyTrainer
from rf2aa.data.compose_dataset import compose_posebusters

class Validator(LegacyTrainer):

    def evaluate_model(self, rank, world_size):
        raise NotImplementedError()
    
    def compose_dataset(self):
        raise NotImplementedError()
    
    def valid_step(self, inputs, n_cycle):
        pass


class PoseBustersBenchmark(Validator):

    def construct_dataset(self, rank, world_size):
        return compose_posebusters(self.config.loader_params, rank, world_size)

    def evaluate_model(self, rank, world_size):
        world_size = torch.cuda.device_count()
        if ('MASTER_ADDR' not in os.environ):
            os.environ['MASTER_ADDR'] = '127.0.0.1' # multinode requires this set in submit script
        if ('MASTER_PORT' not in os.environ):
            os.environ['MASTER_PORT'] = '%d'%self.config.ddp_params.port

        gpu = self.init_process_group(rank, world_size) 
        benchmark_loader = self.construct_dataset(rank, world_size)

        # move global information to device
        self.move_constants_to_device(gpu)

        self.construct_model(device=gpu)
        self.model = DDP(self.model, device_ids=[gpu], find_unused_parameters=False, broadcast_buffers=False)

        self.load_checkpoint(rank)
        self.load_model()
        self.model.eval()
        records = []
        for inputs in benchmark_loader:
            item = inputs[-1]
            with torch.no_grad():
                loss, loss_dict = self.train_step(inputs, self.config.loader_params.maxcycle) 
            loss_dict["CHAINID"] = item["CHAINID"][0]
            for k, v in loss_dict.items():
                if torch.is_tensor(v):
                    loss_dict[k] = v.item()
            records.append(loss_dict)
            df = pd.DataFrame(records)
            df.to_csv(f"{self.output_dir}/{self.config.experiment.name}_posebusters.csv")
            torch.cuda.empty_cache()


@hydra.main(version_base=None, config_path='config/train')
def main(config):
    benchmarker = PoseBustersBenchmark(config=config)
    benchmarker.evaluate_model(0, 1)

if __name__ == "__main__":
    main()
    
    