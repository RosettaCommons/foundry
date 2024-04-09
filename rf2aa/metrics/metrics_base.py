class Metric:
    def __call__(self, rf_output, loss_calc_items) -> float:
        raise NotImplementedError("base class")

