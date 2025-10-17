"""Evaluation metrics for time series forecasting."""


import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)

from ..utils.logger import get_logger

logger = get_logger(__name__)


def mean_absolute_error_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""
    return float(mean_absolute_error(y_true, y_pred))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mean_absolute_scaled_error(
    y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, seasonality: int = 1
) -> float:
    """
    Calculate Mean Absolute Scaled Error (MASE).

    Args:
        y_true: True values
        y_pred: Predicted values
        y_train: Training data for scaling
        seasonality: Seasonal period for naive forecast

    Returns:
        MASE value
    """
    # Calculate MAE of predictions
    mae_pred = mean_absolute_error(y_true, y_pred)

    # Calculate MAE of naive seasonal forecast on training data
    if len(y_train) <= seasonality:
        # Use simple naive forecast if not enough data for seasonal
        naive_errors = np.abs(np.diff(y_train))
    else:
        naive_forecast = y_train[:-seasonality]
        naive_actual = y_train[seasonality:]
        naive_errors = np.abs(naive_actual - naive_forecast)

    mae_naive = float(np.mean(naive_errors))

    if mae_naive == 0:
        return float(np.inf) if mae_pred > 0 else 0.0

    return float(mae_pred / mae_naive)


def symmetric_mean_absolute_percentage_error(
    y_true: np.ndarray, y_pred: np.ndarray
) -> float:
    """Calculate Symmetric Mean Absolute Percentage Error (sMAPE)."""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2

    # Avoid division by zero
    mask = denominator != 0
    if not np.any(mask):
        return 0.0

    return float(np.mean(numerator[mask] / denominator[mask]) * 100)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate directional accuracy (percentage of correct direction predictions).

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Directional accuracy as percentage
    """
    if len(y_true) <= 1:
        return 0.0

    # Calculate direction changes
    true_directions = np.sign(np.diff(y_true))
    pred_directions = np.sign(np.diff(y_pred))

    # Calculate accuracy
    correct_directions = (true_directions == pred_directions).sum()
    total_directions = len(true_directions)

    return float(correct_directions / total_directions * 100)


def continuous_ranked_probability_score(
    y_true: np.ndarray, y_pred_quantiles: np.ndarray, quantile_levels: np.ndarray
) -> float:
    """
    Calculate Continuous Ranked Probability Score (CRPS) for probabilistic forecasts.

    Args:
        y_true: True values
        y_pred_quantiles: Predicted quantiles (shape: [n_samples, n_quantiles])
        quantile_levels: Quantile levels (e.g., [0.1, 0.2, ..., 0.9])

    Returns:
        CRPS value
    """
    if y_pred_quantiles.ndim == 1:
        # Single prediction, convert to 2D
        y_pred_quantiles = y_pred_quantiles.reshape(1, -1)

    n_samples, n_quantiles = y_pred_quantiles.shape

    if len(quantile_levels) != n_quantiles:
        raise ValueError(
            "Number of quantile levels must match number of quantile predictions"
        )

    crps_values = []

    for i in range(n_samples):
        true_val = y_true[i]
        pred_quantiles = y_pred_quantiles[i]

        # Calculate CRPS for this sample
        crps = 0.0
        for j, q_level in enumerate(quantile_levels):
            q_pred = pred_quantiles[j]
            indicator = 1.0 if true_val <= q_pred else 0.0
            crps += (indicator - q_level) * (q_pred - true_val)

        crps_values.append(crps)

    return float(np.mean(crps_values))


def quantile_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """
    Calculate quantile loss for a specific quantile.

    Args:
        y_true: True values
        y_pred: Predicted quantile values
        quantile: Quantile level (e.g., 0.5 for median)

    Returns:
        Quantile loss
    """
    errors = y_true - y_pred
    loss = np.maximum(quantile * errors, (quantile - 1) * errors)
    return float(np.mean(loss))


def interval_coverage(
    y_true: np.ndarray, y_pred_lower: np.ndarray, y_pred_upper: np.ndarray
) -> float:
    """
    Calculate prediction interval coverage.

    Args:
        y_true: True values
        y_pred_lower: Lower bound predictions
        y_pred_upper: Upper bound predictions

    Returns:
        Coverage percentage
    """
    within_interval = (y_true >= y_pred_lower) & (y_true <= y_pred_upper)
    return float(np.mean(within_interval) * 100)


def interval_width(y_pred_lower: np.ndarray, y_pred_upper: np.ndarray) -> float:
    """Calculate average prediction interval width."""
    return float(np.mean(y_pred_upper - y_pred_lower))


class ForecastEvaluator:
    """Comprehensive forecast evaluation class."""

    def __init__(self, metrics: list[str] | None = None) -> None:
        """
        Initialize the evaluator.

        Args:
            metrics: List of metrics to compute
        """
        if metrics is None:
            metrics = ["mae", "rmse", "mase", "smape", "directional_accuracy"]

        self.metrics = metrics
        self.available_metrics = {
            "mae": mean_absolute_error_metric,
            "rmse": root_mean_squared_error,
            "mase": mean_absolute_scaled_error,
            "smape": symmetric_mean_absolute_percentage_error,
            "directional_accuracy": directional_accuracy,
            "crps": continuous_ranked_probability_score,
            "quantile_loss": quantile_loss,
            "interval_coverage": interval_coverage,
            "interval_width": interval_width,
        }

    def evaluate_point_forecast(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_train: np.ndarray | None = None,
    ) -> dict[str, float]:
        """
        Evaluate point forecasts.

        Args:
            y_true: True values
            y_pred: Predicted values
            y_train: Training data (needed for MASE)

        Returns:
            Dictionary of metric values
        """
        results = {}

        for metric_name in self.metrics:
            if metric_name not in self.available_metrics:
                logger.warning(f"Unknown metric: {metric_name}")
                continue

            try:
                if metric_name == "mase" and y_train is not None:
                    value = mean_absolute_scaled_error(y_true, y_pred, y_train)
                elif metric_name in ["mae", "rmse", "smape", "directional_accuracy"]:
                    metric_func = self.available_metrics[metric_name]
                    if metric_func is not None and callable(metric_func):
                        value = metric_func(y_true, y_pred)
                    else:
                        continue
                else:
                    continue

                results[metric_name] = value

            except Exception as e:
                logger.error(f"Error calculating {metric_name}: {e}")
                results[metric_name] = np.nan

        return results

    def evaluate_probabilistic_forecast(
        self,
        y_true: np.ndarray,
        y_pred_quantiles: np.ndarray,
        quantile_levels: np.ndarray,
        confidence_level: float = 0.9,
    ) -> dict[str, float]:
        """
        Evaluate probabilistic forecasts.

        Args:
            y_true: True values
            y_pred_quantiles: Predicted quantiles
            quantile_levels: Quantile levels
            confidence_level: Confidence level for interval evaluation

        Returns:
            Dictionary of metric values
        """
        results = {}

        # CRPS
        if "crps" in self.metrics:
            try:
                crps = continuous_ranked_probability_score(
                    y_true, y_pred_quantiles, quantile_levels
                )
                results["crps"] = crps
            except Exception as e:
                logger.error(f"Error calculating CRPS: {e}")
                results["crps"] = np.nan

        # Quantile losses
        for i, q_level in enumerate(quantile_levels):
            metric_name = f"quantile_loss_{q_level:.1f}"
            if metric_name in self.metrics or "quantile_loss" in self.metrics:
                try:
                    q_loss = quantile_loss(y_true, y_pred_quantiles[:, i], q_level)
                    results[metric_name] = q_loss
                except Exception as e:
                    logger.error(f"Error calculating {metric_name}: {e}")
                    results[metric_name] = np.nan

        # Interval coverage and width
        alpha = 1 - confidence_level
        lower_q = alpha / 2
        upper_q = 1 - alpha / 2

        try:
            lower_idx = np.argmin(np.abs(quantile_levels - lower_q))
            upper_idx = np.argmin(np.abs(quantile_levels - upper_q))

            y_pred_lower = y_pred_quantiles[:, lower_idx]
            y_pred_upper = y_pred_quantiles[:, upper_idx]

            if "interval_coverage" in self.metrics:
                coverage = interval_coverage(y_true, y_pred_lower, y_pred_upper)
                results["interval_coverage"] = coverage

            if "interval_width" in self.metrics:
                width = interval_width(y_pred_lower, y_pred_upper)
                results["interval_width"] = width

        except Exception as e:
            logger.error(f"Error calculating interval metrics: {e}")
            if "interval_coverage" in self.metrics:
                results["interval_coverage"] = np.nan
            if "interval_width" in self.metrics:
                results["interval_width"] = np.nan

        return results

    def evaluate_classification(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict[str, float]:
        """
        Evaluate classification forecasts (e.g., up/down direction).

        Args:
            y_true: True class labels
            y_pred: Predicted class labels

        Returns:
            Dictionary of metric values
        """
        results = {}

        try:
            results["accuracy"] = float(accuracy_score(y_true, y_pred))
            results["f1_score"] = float(f1_score(y_true, y_pred, average="weighted"))
        except Exception as e:
            logger.error(f"Error calculating classification metrics: {e}")
            results["accuracy"] = np.nan
            results["f1_score"] = np.nan

        return results


def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray | None = None,
    y_pred_quantiles: np.ndarray | None = None,
    quantile_levels: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Calculate all available metrics for a forecast.

    Args:
        y_true: True values
        y_pred: Point predictions
        y_train: Training data
        y_pred_quantiles: Quantile predictions (optional)
        quantile_levels: Quantile levels (optional)

    Returns:
        Dictionary of all metric values
    """
    evaluator = ForecastEvaluator()

    # Point forecast metrics
    results = evaluator.evaluate_point_forecast(y_true, y_pred, y_train)

    # Probabilistic metrics (if available)
    if y_pred_quantiles is not None and quantile_levels is not None:
        prob_results = evaluator.evaluate_probabilistic_forecast(
            y_true, y_pred_quantiles, quantile_levels
        )
        results.update(prob_results)

    return results
