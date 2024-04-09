from typing import Any, Dict
from rf2aa.metrics.predicted_error import PAE, PLDDT

class MetricManager:

    def __init__(self, config) -> None:
        self.config = config
        self.metrics = {metric:metrics_factory[metric] for metric in config.metrics}

    def __call__(self, rf_outputs, loss_calc_items) -> Dict:
        metrics_dict = {}
        for metric_name, metric in self.metrics:
            metric_value = metric(rf_outputs, loss_calc_items)
            metrics_dict[metric_name] = metric_value
        return metrics_dict

metrics_factory = {
    "mean_pae": PAE(),
    "mean_plddt": PLDDT()
}