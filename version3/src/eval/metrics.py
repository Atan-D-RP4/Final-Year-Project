"""Evaluation metrics for forecasting models."""

import numpy as np
import pandas as pd


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error.

    Args:
        actual: Actual values
        predicted: Predicted values

    Returns:
        MAE value
    """
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error.

    Args:
        actual: Actual values
        predicted: Predicted values

    Returns:
        RMSE value
    """
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mase(
    actual: np.ndarray,
    predicted: np.ndarray,
    seasonal_period: int = 1,
) -> float:
    """Mean Absolute Scaled Error.

    Args:
        actual: Actual values
        predicted: Predicted values
        seasonal_period: Seasonal period for naive forecast

    Returns:
        MASE value
    """
    # Compute naive forecast error
    if len(actual) < seasonal_period + 1:
        return float(np.mean(np.abs(actual - predicted)))

    naive_errors = np.abs(actual[seasonal_period:] - actual[:-seasonal_period])
    naive_mean_error = np.mean(naive_errors)

    if naive_mean_error == 0:
        return float(np.mean(np.abs(actual - predicted)))

    return float(np.mean(np.abs(actual - predicted)) / naive_mean_error)


def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error.

    Args:
        actual: Actual values
        predicted: Predicted values

    Returns:
        sMAPE value (0-1 scale)
    """
    denominator = (np.abs(actual) + np.abs(predicted)) / 2.0
    diff = np.abs(actual - predicted) / denominator
    diff[denominator == 0] = 0
    return float(np.mean(diff))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Percentage Error.

    Args:
        actual: Actual values
        predicted: Predicted values

    Returns:
        MAPE value (0-1 scale)
    """
    # Avoid division by zero
    mask = actual != 0
    if not np.any(mask):
        return float(np.mean(np.abs(actual - predicted)))

    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])))


def directional_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Directional accuracy (percentage of correct direction changes).

    Args:
        actual: Actual values
        predicted: Predicted values

    Returns:
        Directional accuracy (0-1 scale)
    """
    if len(actual) < 2:
        return 0.0

    actual_direction = np.sign(np.diff(actual))
    predicted_direction = np.sign(np.diff(predicted))

    accuracy = np.mean(actual_direction == predicted_direction)
    return float(accuracy)


def crps(
    actual: float,
    predicted_samples: np.ndarray,
) -> float:
    """Continuous Ranked Probability Score.

    Args:
        actual: Actual value
        predicted_samples: Array of predicted samples

    Returns:
        CRPS value
    """
    sorted_pred = np.sort(predicted_samples)
    n = len(sorted_pred)

    # Find where actual falls in sorted predictions
    idx = np.searchsorted(sorted_pred, actual)

    # Calculate CRPS
    crps_val = 0.0
    for i, pred in enumerate(sorted_pred):
        if i < idx:
            crps_val += pred - actual
        else:
            crps_val += actual - pred

    crps_val += actual - sorted_pred[0]
    crps_val += sorted_pred[-1] - actual

    return crps_val / n


def quantile_loss(
    actual: np.ndarray,
    predicted: np.ndarray,
    q: float = 0.5,
) -> float:
    """Quantile loss for probabilistic forecasting.

    Args:
        actual: Actual values
        predicted: Predicted values (quantile point)
        q: Quantile level (0-1)

    Returns:
        Quantile loss
    """
    residual = actual - predicted
    return float(np.mean(np.where(residual >= 0, q * residual, (q - 1) * residual)))


def interval_coverage(
    actual: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
) -> float:
    """Percentage of actual values within prediction interval.

    Args:
        actual: Actual values
        lower_bound: Lower bound of interval
        upper_bound: Upper bound of interval

    Returns:
        Coverage probability (0-1)
    """
    covered = (actual >= lower_bound) & (actual <= upper_bound)
    return float(np.mean(covered))


def interval_width(lower_bound: np.ndarray, upper_bound: np.ndarray) -> float:
    """Average width of prediction intervals.

    Args:
        lower_bound: Lower bound of interval
        upper_bound: Upper bound of interval

    Returns:
        Average interval width
    """
    return float(np.mean(upper_bound - lower_bound))


def calibration_error(
    actual: np.ndarray,
    quantile_forecasts: dict[float, np.ndarray],
) -> float:
    """Calibration error for quantile forecasts.

    Args:
        actual: Actual values
        quantile_forecasts: Dictionary mapping quantile levels to forecasts

    Returns:
        Average absolute calibration error
    """
    errors = []

    for q, forecast in quantile_forecasts.items():
        # Empirical coverage
        coverage = float(np.mean(actual <= forecast))
        # Target coverage
        error = np.abs(coverage - q)
        errors.append(error)

    if not errors:
        return 0.0
    return float(np.mean(errors))


def pinball_loss(
    actual: np.ndarray,
    predicted: np.ndarray,
    q: float = 0.5,
) -> float:
    """Pinball loss for quantile regression.

    Args:
        actual: Actual values
        predicted: Predicted quantiles
        q: Quantile level

    Returns:
        Pinball loss
    """
    residual = actual - predicted
    return float(np.mean(np.where(residual >= 0, q * residual, (q - 1) * residual)))


class ForecastEvaluator:
    """Comprehensive forecast evaluator."""

    def __init__(self, seasonal_period: int = 252):
        """Initialize evaluator.

        Args:
            seasonal_period: Seasonal period for metrics like MASE
        """
        self.seasonal_period = seasonal_period
        self.results = {}

    def evaluate(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        quantile_forecasts: dict[float, np.ndarray] | None = None,
        metrics: list[str] | None = None,
    ) -> dict:
        """Evaluate forecasts with multiple metrics.

        Args:
            actual: Actual values
            predicted: Predicted values
            quantile_forecasts: Optional quantile forecasts for probabilistic
                evaluation
            metrics: List of metrics to compute (None for all)

        Returns:
            Dictionary of metric results
        """
        if metrics is None:
            metrics = [
                "mae",
                "rmse",
                "mase",
                "mape",
                "smape",
                "directional_accuracy",
            ]

        results = {}

        # Compute point forecast metrics
        results.update(self._compute_point_metrics(actual, predicted, metrics))

        # Compute probabilistic metrics
        if quantile_forecasts is not None:
            results.update(self._compute_probabilistic_metrics(quantile_forecasts, actual, metrics))

        self.results = results
        return results

    def _compute_point_metrics(
        self, actual: np.ndarray, predicted: np.ndarray, metrics: list[str]
    ) -> dict:
        """Compute point forecast metrics."""
        results = {}

        # Point forecast metrics
        if "mae" in metrics:
            results["mae"] = float(mae(actual, predicted))

        if "rmse" in metrics:
            results["rmse"] = float(rmse(actual, predicted))

        if "mase" in metrics:
            results["mase"] = float(mase(actual, predicted, self.seasonal_period))

        if "mape" in metrics:
            results["mape"] = float(mape(actual, predicted))

        if "smape" in metrics:
            results["smape"] = float(smape(actual, predicted))

        if "directional_accuracy" in metrics:
            results["directional_accuracy"] = float(directional_accuracy(actual, predicted))

        return results

    def _compute_probabilistic_metrics(
        self, quantile_forecasts: dict[float, np.ndarray], actual: np.ndarray, metrics: list[str]
    ) -> dict:
        """Compute probabilistic metrics."""
        results = {}

        if "calibration_error" in metrics:
            results["calibration_error"] = float(calibration_error(actual, quantile_forecasts))

        if "interval_coverage_90" in metrics and (
            0.05 in quantile_forecasts and 0.95 in quantile_forecasts
        ):
            coverage = interval_coverage(
                actual,
                quantile_forecasts[0.05],
                quantile_forecasts[0.95],
            )
            results["interval_coverage_90"] = float(coverage)

        if "interval_width_90" in metrics and (
            0.05 in quantile_forecasts and 0.95 in quantile_forecasts
        ):
            width = interval_width(
                quantile_forecasts[0.05],
                quantile_forecasts[0.95],
            )
            results["interval_width_90"] = float(width)

        return results

    def summary(self) -> pd.DataFrame:
        """Get summary of evaluation results.

        Returns:
            DataFrame with metric results
        """
        return pd.DataFrame([self.results])

    def compare_models(
        self,
        models_predictions: dict[str, np.ndarray],
        actual: np.ndarray,
        metrics: list[str] | None = None,
    ) -> pd.DataFrame:
        """Compare multiple model predictions.

        Args:
            models_predictions: Dictionary mapping model names to predictions
            actual: Actual values
            metrics: List of metrics to compute

        Returns:
            DataFrame comparing models
        """
        comparisons = {}

        for model_name, predicted in models_predictions.items():
            comparisons[model_name] = self.evaluate(actual, predicted, metrics=metrics)

        return pd.DataFrame(comparisons).T

    def directional_accuracy_breakdown(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
    ) -> dict:
        """Breakdown of directional accuracy by regime.

        Args:
            actual: Actual values
            predicted: Predicted values

        Returns:
            Dictionary with breakdown by upward, downward, flat
        """
        actual_direction = np.sign(np.diff(actual))
        predicted_direction = np.sign(np.diff(predicted))

        upward_actual = actual_direction == 1
        downward_actual = actual_direction == -1
        flat_actual = actual_direction == 0

        results = {}

        if np.sum(upward_actual) > 0:
            upward_acc = np.mean(
                predicted_direction[upward_actual] == actual_direction[upward_actual]
            )
            results["upward_accuracy"] = float(upward_acc)

        if np.sum(downward_actual) > 0:
            downward_acc = np.mean(
                predicted_direction[downward_actual] == actual_direction[downward_actual]
            )
            results["downward_accuracy"] = float(downward_acc)

        if np.sum(flat_actual) > 0:
            flat_acc = np.mean(predicted_direction[flat_actual] == actual_direction[flat_actual])
            results["flat_accuracy"] = float(flat_acc)

        return results

    def error_statistics(self, actual: np.ndarray, predicted: np.ndarray) -> dict:
        """Compute error statistics.

        Args:
            actual: Actual values
            predicted: Predicted values

        Returns:
            Dictionary with error statistics
        """
        errors = actual - predicted

        return {
            "mean_error": float(np.mean(errors)),
            "std_error": float(np.std(errors)),
            "min_error": float(np.min(errors)),
            "max_error": float(np.max(errors)),
            "median_error": float(np.median(errors)),
        }


def calculate_all_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    metrics: list[str] | None = None,
    quantile_forecasts: dict[float, np.ndarray] | None = None,
) -> dict:
    """Calculate all requested metrics.

    Args:
        actual: Actual values
        predicted: Predicted values
        metrics: List of metrics to compute
        quantile_forecasts: Optional quantile forecasts

    Returns:
        Dictionary of metric results
    """
    evaluator = ForecastEvaluator()
    return evaluator.evaluate(actual, predicted, quantile_forecasts, metrics)
