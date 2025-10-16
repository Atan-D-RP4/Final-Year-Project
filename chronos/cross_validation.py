"""
Cross-Validation Framework for Time Series Forecasting
Implements time-series specific validation methods and comprehensive evaluation metrics
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


class TimeSeriesCrossValidator:
    """
    Time series cross-validation with multiple strategies
    """

    def __init__(
        self,
        strategy: str = "expanding",
        n_splits: int = 5,
        test_size: int = 5,
        gap: int = 0,
        min_train_size: int = 20,
    ):
        """
        Initialize time series cross-validator

        Args:
            strategy: 'expanding', 'sliding', or 'blocked'
            n_splits: Number of cross-validation splits
            test_size: Size of test set in each split
            gap: Gap between train and test sets
            min_train_size: Minimum training set size
        """
        self.strategy = strategy
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.min_train_size = min_train_size

    def split(self, data: List[float]) -> List[Tuple[List[int], List[int]]]:
        """
        Generate train/test splits for time series data

        Args:
            data: Time series data

        Returns:
            List of (train_indices, test_indices) tuples
        """
        n_samples = len(data)
        splits = []

        if self.strategy == "expanding":
            # Expanding window: training set grows with each split
            for i in range(self.n_splits):
                test_start = self.min_train_size + i * self.test_size + self.gap
                test_end = test_start + self.test_size

                if test_end > n_samples:
                    break

                train_indices = list(range(test_start - self.gap))
                test_indices = list(range(test_start, test_end))

                if len(train_indices) >= self.min_train_size:
                    splits.append((train_indices, test_indices))

        elif self.strategy == "sliding":
            # Sliding window: training set size remains constant
            train_size = max(
                self.min_train_size,
                (n_samples - self.n_splits * self.test_size - self.gap)
                // self.n_splits,
            )

            for i in range(self.n_splits):
                test_start = self.min_train_size + i * self.test_size + self.gap
                test_end = test_start + self.test_size

                if test_end > n_samples:
                    break

                train_start = max(0, test_start - self.gap - train_size)
                train_end = test_start - self.gap

                train_indices = list(range(train_start, train_end))
                test_indices = list(range(test_start, test_end))

                if len(train_indices) >= self.min_train_size:
                    splits.append((train_indices, test_indices))

        elif self.strategy == "blocked":
            # Blocked cross-validation: non-overlapping blocks
            block_size = n_samples // (self.n_splits * 2)  # Alternate train/test blocks

            for i in range(self.n_splits):
                train_start = i * 2 * block_size
                train_end = train_start + block_size
                test_start = train_end + self.gap
                test_end = test_start + min(block_size, self.test_size)

                if test_end > n_samples:
                    break

                train_indices = list(range(train_start, train_end))
                test_indices = list(range(test_start, test_end))

                if len(train_indices) >= self.min_train_size:
                    splits.append((train_indices, test_indices))

        return splits

    def visualize_splits(
        self, data: List[float], splits: List[Tuple[List[int], List[int]]]
    ):
        """
        Visualize the cross-validation splits

        Args:
            data: Time series data
            splits: List of (train_indices, test_indices) tuples
        """
        fig, axes = plt.subplots(len(splits), 1, figsize=(12, 2 * len(splits)))
        if len(splits) == 1:
            axes = [axes]

        for i, (train_idx, test_idx) in enumerate(splits):
            ax = axes[i]

            # Plot full data in light gray
            ax.plot(range(len(data)), data, color="lightgray", alpha=0.5)

            # Plot training data in blue
            train_data = [data[j] for j in train_idx]
            ax.plot(train_idx, train_data, color="blue", label="Train")

            # Plot test data in red
            test_data = [data[j] for j in test_idx]
            ax.plot(test_idx, test_data, color="red", label="Test")

            ax.set_title(f"Split {i + 1}: Train={len(train_idx)}, Test={len(test_idx)}")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


class ForecastingMetrics:
    """
    Comprehensive forecasting evaluation metrics
    """

    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Squared Error"""
        return mean_squared_error(y_true, y_pred)

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error"""
        return np.sqrt(mean_squared_error(y_true, y_pred))

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error"""
        return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error"""
        return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    @staticmethod
    def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Symmetric Mean Absolute Percentage Error"""
        return (
            np.mean(
                2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
            )
            * 100
        )

    @staticmethod
    def mase(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_train: np.ndarray,
        seasonality: int = 1,
    ) -> float:
        """Mean Absolute Scaled Error"""
        # Calculate naive forecast error on training set
        if len(y_train) <= seasonality:
            naive_error = np.mean(np.abs(np.diff(y_train)))
        else:
            naive_forecast = y_train[:-seasonality]
            naive_actual = y_train[seasonality:]
            naive_error = np.mean(np.abs(naive_actual - naive_forecast))

        if naive_error == 0:
            return np.inf if np.any(y_true != y_pred) else 0

        return np.mean(np.abs(y_true - y_pred)) / naive_error

    @staticmethod
    def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Directional Accuracy (percentage of correct direction predictions)"""
        if len(y_true) < 2:
            return 0.0

        true_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))

        return np.mean(true_direction == pred_direction) * 100

    @staticmethod
    def forecast_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Forecast Bias (mean of prediction errors)"""
        return np.mean(y_pred - y_true)

    @staticmethod
    def prediction_interval_coverage(
        y_true: np.ndarray, lower_bound: np.ndarray, upper_bound: np.ndarray
    ) -> float:
        """Prediction Interval Coverage Probability"""
        coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
        return float(coverage * 100)

    @staticmethod
    def theil_u_statistic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Theil's U Statistic"""
        mse_pred = np.mean((y_true - y_pred) ** 2)
        mse_naive = np.mean((y_true[1:] - y_true[:-1]) ** 2)

        if mse_naive == 0:
            return np.inf if mse_pred > 0 else 0

        return np.sqrt(mse_pred / mse_naive)


class BacktestingFramework:
    """Comprehensive backtesting framework for time series forecasting"""

    def __init__(
        self,
        forecaster,
        cv_strategy: str = "expanding",
        n_splits: int = 5,
        prediction_length: int = 5,
        window_size: int = 10,
    ):
        """Initialize backtesting framework

        Args:
            forecaster: Forecasting model with fit_transform and forecast methods
            cv_strategy: Cross-validation strategy
            n_splits: Number of CV splits
            prediction_length: Forecast horizon
            window_size: Input window size
        """
        self.forecaster = forecaster
        self.cv_strategy = cv_strategy
        self.n_splits = n_splits
        self.prediction_length = prediction_length
        self.window_size = window_size

        self.cv = TimeSeriesCrossValidator(
            strategy=cv_strategy,
            n_splits=n_splits,
            test_size=prediction_length,
            min_train_size=window_size * 2,
        )

        self.metrics = ForecastingMetrics()
        self.results = {}

    def run_backtest(
        self,
        data: List[float],
        show_progress: bool = True,
        return_predictions: bool = False,
    ) -> Dict[str, Any]:
        """
        Run comprehensive backtest

        Args:
            data: Time series data
            show_progress: Whether to show progress
            return_predictions: Whether to return individual predictions

        Returns:
            Dictionary with backtest results
        """
        if show_progress:
            print(
                f"Running backtest with {self.cv_strategy} CV, {self.n_splits} splits..."
            )

        splits = self.cv.split(data)

        if not splits:
            raise ValueError(
                "No valid splits generated. Check data size and parameters."
            )

        all_predictions = []
        all_actuals = []
        all_train_data = []
        split_results = []

        for i, (train_idx, test_idx) in enumerate(splits):
            if show_progress:
                print(f"  Processing split {i + 1}/{len(splits)}...")

            # Extract train and test data
            train_data = [data[j] for j in train_idx]
            test_data = [data[j] for j in test_idx]

            try:
                # Prepare training data
                tokenized_data, tokenizer = self.forecaster.prepare_data(
                    train_data, self.window_size
                )

                # Make forecast
                forecast_result = self.forecaster.forecast_zero_shot(
                    tokenized_data, len(test_data)
                )

                predictions = forecast_result["mean"][0].numpy()

                # Store results
                all_predictions.extend(predictions)
                all_actuals.extend(test_data)
                all_train_data.append(train_data)

                # Calculate metrics for this split
                split_metrics = self._calculate_metrics(
                    np.array(test_data), np.array(predictions), np.array(train_data)
                )

                split_results.append(
                    {
                        "split": i + 1,
                        "train_size": len(train_data),
                        "test_size": len(test_data),
                        "metrics": split_metrics,
                        "predictions": predictions.tolist()
                        if return_predictions
                        else None,
                        "actuals": test_data if return_predictions else None,
                    }
                )

            except Exception as e:
                if show_progress:
                    print(f"    Error in split {i + 1}: {e}")
                continue

        if not all_predictions:
            raise ValueError("No successful predictions made")

        # Calculate overall metrics
        overall_metrics = self._calculate_metrics(
            np.array(all_actuals),
            np.array(all_predictions),
            np.concatenate(all_train_data) if all_train_data else np.array([]),
        )

        # Calculate metric statistics across splits
        metric_stats = self._calculate_metric_statistics(split_results)

        self.results = {
            "overall_metrics": overall_metrics,
            "metric_statistics": metric_stats,
            "split_results": split_results,
            "cv_strategy": self.cv_strategy,
            "n_splits": len(splits),
            "total_predictions": len(all_predictions),
            "all_predictions": all_predictions if return_predictions else None,
            "all_actuals": all_actuals if return_predictions else None,
        }

        if show_progress:
            print("  Backtest completed!")

        return self.results

    def _calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray
    ) -> Dict[str, float]:
        """Calculate all metrics for given predictions"""
        metrics = {
            "mse": self.metrics.mse(y_true, y_pred),
            "rmse": self.metrics.rmse(y_true, y_pred),
            "mae": self.metrics.mae(y_true, y_pred),
            "mape": self.metrics.mape(y_true, y_pred),
            "smape": self.metrics.smape(y_true, y_pred),
            "directional_accuracy": self.metrics.directional_accuracy(y_true, y_pred),
            "forecast_bias": self.metrics.forecast_bias(y_true, y_pred),
            "theil_u": self.metrics.theil_u_statistic(y_true, y_pred),
        }

        # Add MASE if training data is available
        if len(y_train) > 1:
            metrics["mase"] = self.metrics.mase(y_true, y_pred, y_train)

        return metrics

    def _calculate_metric_statistics(
        self, split_results: List[Dict]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate statistics across splits for each metric"""
        metric_names = list(split_results[0]["metrics"].keys())
        stats = {}

        for metric in metric_names:
            values = [
                split["metrics"][metric]
                for split in split_results
                if not np.isnan(split["metrics"][metric])
                and not np.isinf(split["metrics"][metric])
            ]

            if values:
                stats[metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "median": float(np.median(values)),
                }
            else:
                stats[metric] = {
                    "mean": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                    "median": np.nan,
                }

        return stats

    def print_results(self):
        """Print formatted backtest results"""
        if not self.results:
            print("No results to display. Run backtest first.")
            return

        print("" + "=" * 70)
        print("TIME SERIES FORECASTING BACKTEST RESULTS")
        print("=" * 70)

        print("Backtest Configuration:")
        print(f"  CV Strategy: {self.results['cv_strategy']}")
        print(f"  Number of Splits: {self.results['n_splits']}")
        print(f"  Total Predictions: {self.results['total_predictions']}")
        print(f"  Prediction Length: {self.prediction_length}")

        print("Overall Performance:")
        overall = self.results["overall_metrics"]
        for metric, value in overall.items():
            print(f"  {metric.upper()}: {value:.4f}")

        print("Cross-Validation Statistics:")
        stats = self.results["metric_statistics"]
        print(f"{'Metric':<20} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
        print("-" * 70)

        for metric, stat_dict in stats.items():
            print(
                f"{metric.upper():<20} {stat_dict['mean']:<10.4f} {stat_dict['std']:<10.4f} "
                f"{stat_dict['min']:<10.4f} {stat_dict['max']:<10.4f}"
            )

    def plot_results(self):
        """Plot backtest results"""
        if not self.results or not self.results.get("all_predictions"):
            print(
                "No prediction data to plot. Run backtest with return_predictions=True."
            )
            return

        predictions = self.results["all_predictions"]
        actuals = self.results["all_actuals"]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Time series plot
        axes[0, 0].plot(actuals, label="Actual", alpha=0.7)
        axes[0, 0].plot(predictions, label="Predicted", alpha=0.7)
        axes[0, 0].set_title("Predictions vs Actuals")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Scatter plot
        axes[0, 1].scatter(actuals, predictions, alpha=0.6)
        min_val, max_val = (
            min(min(actuals), min(predictions)),
            max(max(actuals), max(predictions)),
        )
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
        axes[0, 1].set_xlabel("Actual")
        axes[0, 1].set_ylabel("Predicted")
        axes[0, 1].set_title("Prediction Scatter Plot")
        axes[0, 1].grid(True, alpha=0.3)

        # Residuals
        residuals = np.array(actuals) - np.array(predictions)
        axes[1, 0].plot(residuals)
        axes[1, 0].axhline(y=0, color="r", linestyle="--")
        axes[1, 0].set_title("Residuals Over Time")
        axes[1, 0].set_ylabel("Residual")
        axes[1, 0].grid(True, alpha=0.3)

        # Residual histogram
        axes[1, 1].hist(residuals, bins=20, alpha=0.7, edgecolor="black")
        axes[1, 1].axvline(x=0, color="r", linestyle="--")
        axes[1, 1].set_title("Residual Distribution")
        axes[1, 1].set_xlabel("Residual")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def compare_strategies(
        self, data: List[float], strategies: List[str]
    ) -> Dict[str, Dict]:
        """Compare different CV strategies"""
        comparison_results = {}

        original_strategy = self.cv_strategy

        for strategy in strategies:
            print(f"Testing {strategy} strategy...")
            self.cv_strategy = strategy
            self.cv.strategy = strategy

            try:
                results = self.run_backtest(data, show_progress=False)
                comparison_results[strategy] = results["overall_metrics"]
            except Exception as e:
                print(f"Error with {strategy}: {e}")
                comparison_results[strategy] = None

        # Restore original strategy
        self.cv_strategy = original_strategy
        self.cv.strategy = original_strategy

        # Print comparison
        print("" + "=" * 80)
        print("CROSS-VALIDATION STRATEGY COMPARISON")
        print("=" * 80)

        if any(result is not None for result in comparison_results.values()):
            # Get metric names from first successful result
            metric_names = None
            for result in comparison_results.values():
                if result is not None:
                    metric_names = list(result.keys())
                    break

            if metric_names:
                print(f"{'Strategy':<15} ", end="")
                for metric in metric_names:
                    print(f"{metric.upper():<10} ", end="")
                print()
                print("-" * (15 + len(metric_names) * 11))

                for strategy, results in comparison_results.items():
                    print(f"{strategy:<15} ", end="")
                    if results:
                        for metric in metric_names:
                            print(f"{results[metric]:<10.4f} ", end="")
                    else:
                        print("FAILED" + " " * (len(metric_names) * 11 - 6), end="")
                    print()

        return comparison_results


def test_cross_validation():
    """
    Test the cross-validation framework
    """
    print("Testing Cross-Validation Framework...")

    # Generate synthetic time series data
    np.random.seed(42)
    n_samples = 100
    trend = np.linspace(0, 10, n_samples)
    seasonal = 3 * np.sin(2 * np.pi * np.arange(n_samples) / 12)
    noise = np.random.normal(0, 1, n_samples)
    data = (trend + seasonal + noise + 50).tolist()

    print(f"Generated synthetic data with {len(data)} points")

    # Test different CV strategies
    strategies = ["expanding", "sliding", "blocked"]

    for strategy in strategies:
        print(f"Testing {strategy} strategy:")
        cv = TimeSeriesCrossValidator(
            strategy=strategy, n_splits=5, test_size=5, min_train_size=20
        )

        splits = cv.split(data)
        print(f"  Generated {len(splits)} splits")

        for i, (train_idx, test_idx) in enumerate(splits[:2]):  # Show first 2 splits
            print(f"    Split {i + 1}: Train={len(train_idx)}, Test={len(test_idx)}")

    # Test metrics
    print("Testing metrics:")
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
    y_train = np.array([0.5, 0.8, 1.2, 1.5])

    metrics = ForecastingMetrics()
    print(f"  MSE: {metrics.mse(y_true, y_pred):.4f}")
    print(f"  MAE: {metrics.mae(y_true, y_pred):.4f}")
    print(f"  MAPE: {metrics.mape(y_true, y_pred):.4f}")
    print(f"  MASE: {metrics.mase(y_true, y_pred, y_train):.4f}")
    print(
        f"  Directional Accuracy: {metrics.directional_accuracy(y_true, y_pred):.2f}%"
    )

    print("Cross-validation framework test completed!")
    return data


if __name__ == "__main__":
    # Run tests
    test_data = test_cross_validation()
