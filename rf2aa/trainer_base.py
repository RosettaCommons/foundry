import hydra

from rf2aa.set_seed import seed_all

from rf2aa.trainer_new import ComposedTrainer, LegacyTrainer, FlowMatchingTrainer
from rf2aa.experiments.msa_module_trainer import MsaModuleTrainer
from rf2aa.experiments.af3_trainer import AF3Trainer, AF3TrainerRollout
from rf2aa.manual_dependency import append_package_path



@hydra.main(version_base=None, config_path='config/train')
def main(config):
    seed_all(config.training_params.seed)
    #for package_path in config.dependencies.package_paths:
    #    append_package_path(package_path)
    trainer = trainer_factory[config.experiment.trainer](config=config)

    # Wrap the training in a try-except block to ensure SLURM cleanup post-interrupt (otherwise, we'd need to change the SLURM id each run)
    try:
        trainer.launch_distributed_training()
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    except Exception as e:
        print("Training interrupted by exception:", e)
        raise e
    finally:
        print("Cleaning up...") 
        trainer.cleanup()

trainer_factory = {
    "legacy": LegacyTrainer,
    "composed": ComposedTrainer,
    "flow_matching": FlowMatchingTrainer,
    "af3_repro": AF3Trainer,
    "af3_rollout": AF3TrainerRollout,
    "msa_module": MsaModuleTrainer,
}

if __name__ == "__main__":
    main()
