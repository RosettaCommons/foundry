import hydra
import logging
import torch
from icecream import ic

from rf2aa.trainer_lightning import LitDataModule, LitAF3Repro
import lightning as L
from rf2aa.debug import pretty_describe_dict
from rf2aa import pymol
from rf2aa import pymol_tools

pymol.init('http://chesaw.dhcp.ipd:9123')
logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path='config/train')
def main(config):
    pymol_tools.clear()
    model = LitAF3Repro.load_from_checkpoint(config.inference.checkpoint, config=config)
    datamodule = LitDataModule(config)
    model.eval()
    with torch.no_grad():
        for batch in datamodule.predict_dataloader():
            predicted = model(batch)
            # Uncomment to sanity check the training step
            # predicted = model.training_step(batch, 0)
            logger.info('predicted:\n' + pretty_describe_dict(predicted))
            ic(
                predicted['loss']
            )
            pymol_tools.show_pymol(
                pymol_tools.to_atom37(predicted['X_gt_L'][0], predicted['crd_mask_I']),
                predicted['seq'],
                predicted['bond_feats'],
                label=f'gt'
            )
            for i, X_L in enumerate(predicted['X_L']):
                pymol_tools.show_pymol(
                    pymol_tools.to_atom37(X_L, predicted['crd_mask_I']),
                    predicted['seq'],
                    predicted['bond_feats'],
                    label=f'pred_{i}'
                )
            break

if __name__ == "__main__":
    main()