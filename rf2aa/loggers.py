import logging

from icecream import ic
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.loggers.csv_logs import CSVLogger
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities import rank_zero_only

logger = logging.getLogger(__name__)


def mean_over(df, grouper, metrics):
    out = (
        df[[grouper] + metrics]
        .groupby(grouper)
        .mean(numeric_only=True)
        .stack()
        .to_dict()
    )
    out = {
        f"{grouper}={grouper_v}.{metric}": v for (grouper_v, metric), v in out.items()
    }
    return out


class LitLogger(Logger):
    def __init__(self, save_dir, use_wandb, sublogger):
        self.use_wandb = use_wandb
        if self.use_wandb:
            self.sublogger = WandbLogger(**sublogger)
        else:
            self.sublogger = CSVLogger(**sublogger)
        super().__init__()

    def log_df(self, df, stratifications=None):
        global_step = df["global_step"].iloc[0]
        mean_over_step = df.groupby("global_step").mean(numeric_only=True)
        assert len(mean_over_step) == 1
        mean_over_step = mean_over_step.iloc[0]
        mean_over_step = mean_over_step.to_dict()

        stratified = {}
        for groupers, values in stratifications.items():
            # TODO: enable stratification over multiple keys
            assert len(groupers) == 1
            stratified.update(mean_over(df, groupers[0], values))

        stratified = mean_over_step | stratified
        self.sublogger.log_metrics(stratified, step=global_step)

    @property
    def name(self):
        return "MyLogger"

    @property
    def version(self):
        # Return the experiment version, int or str.
        return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        self.sublogger.log_hyperparams(params)

    @rank_zero_only
    def log_metrics(self, metrics, step):
        ic(step)
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        self.sublogger.log_metrics(metrics, step)

    @rank_zero_only
    def save(self):
        # Optional. Any code necessary to save logger data goes here
        self.sublogger.save()

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        self.sublogger.finalize(status)
