from typing import Dict

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


REGRESSION_METRICS = {
    'R2': r2_score,
    'MAE': mean_absolute_error,
    'MSE': mean_squared_error,
    'MAPE': mean_absolute_percentage_error
}


def compute_metrics(y_true, y_hat) -> Dict[str, float]:
    outs = {}
    for metric_name, metric in REGRESSION_METRICS.items():
        metric_value = metric(y_true, y_hat)
        outs[metric_name] = metric_value
    return outs
