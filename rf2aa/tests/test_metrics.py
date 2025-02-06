import pytest
import torch
from hydra import compose, initialize

from rf2aa.metrics.metrics_factory import MetricManager, metrics_factory
from rf2aa.tests.test_conditions import configs, model_pickle_path, setup_array
from rf2aa.tests.test_model import setup_test

test_conditions = setup_array(
    ["pdb"], [config for config in configs if "legacy" not in config]
)
gpu = "cuda:0" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize("example,model", test_conditions)
def test_metrics(example, model):
    # seting up the test
    dataset_name, dataset_inputs, model_name, model = setup_test(example, model)
    model_pickle = model_pickle_path(dataset_name, model_name)
    rf_outputs = torch.load(model_pickle, map_location=gpu)["outputs"]
    loss_calc_items = None
    for metric_name, metric in metrics_factory.items():
        # calling the function
        try:
            metric(rf_outputs, loss_calc_items)
        except Exception as e:
            raise ValueError(
                f"{metric_name} fails with following exception: {e}"
            ) from e


def test_metric_config():
    config = "base"
    cfg_overrides = []
    with initialize(version_base=None, config_path="../config/train"):
        cfg = compose(config_name=config, overrides=cfg_overrides)

    metrics_manager = MetricManager(cfg)

    assert len(metrics_manager.metrics) == 2
    assert "mean_pae" in metrics_manager.metrics
    assert "mean_plddt" in metrics_manager.metrics
