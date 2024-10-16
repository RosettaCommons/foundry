#class Metric:
    #def __call__(self, rf_output, loss_calc_items) -> float:
        #raise NotImplementedError("base class")

class MetricManager(nn.Module):
    """
    Similar syntax to LossManager, but for metrics
    """

    def __init__(self, **metrics):
        super().__init__()
        self.to_compute = []
        for metric_name, metric in metrics.items():
            metric_fn = hydra.utils.instantiate(metric)
            print(f"Adding metric {metric_name} to the loss function")
            self.to_compute.append(metric_fn)
        
    def forward(
        self,
        network_input,
        network_output,
        loss_input,
    ):
        loss_dict = {}
        for loss_fn in self.to_compute:
            loss_, loss_dict_ = loss_fn(network_input, network_output, loss_input)
            loss_dict.update(loss_dict_)
        return loss_dict


class Metric:

    def __call__(self, network_input, network_output, loss_input) -> float:
        raise NotImplementedError("base class")
