"""Phase 3: Zero-shot forecasting with Chronos vs Baseline models."""

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data.cleaning import DataCleaner, create_features
from src.data.fetchers import DataFetcher
from src.eval.metrics import ForecastEvaluator
from src.models.baselines import (
    ARIMAForecaster,
    EnsembleForecaster,
    ExponentialSmoothingForecaster,
    LinearRegressionForecaster,
    LSTMForecaster,
    MeanForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
    VARForecaster,
)
from src.models.chronos_wrapper import ChronosFinancialForecaster
from src.utils.config import DataConfig, EvalConfig, PreprocessingConfig
from src.utils.logger import setup_logger


class ZeroShotExperiment:
    """Zero-shot forecasting experiment."""

    def __init__(
        self,
        results_dir: str = "results/phase3",
        log_dir: str = "results/phase3/logs",
    ):
        """Initialize experiment.

        Args:
            results_dir: Directory to save results
            log_dir: Directory for logs
        """
        self.results_dir = Path(results_dir)
        self.log_dir = Path(log_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = setup_logger(
            "phase3",
            str(self.log_dir / "experiment.log"),
        )

        self.logger.info("Initialized Phase 3 Zero-Shot Experiment")

    def setup_data(
        self,
        data_config: Optional[DataConfig] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Setup and fetch data.

        Args:
            data_config: Data configuration

        Returns:
            Tuple of (full_data, train_data, test_data)
        """
        if data_config is None:
            data_config = DataConfig()

        self.logger.info("Fetching financial data...")

        # Fetch data
        fetcher = DataFetcher()
        
        # Format dates
        start_date = (
            data_config.start_date
            if isinstance(data_config.start_date, str)
            else data_config.start_date.strftime("%Y-%m-%d")
        )
        end_date = (
            data_config.end_date
            if isinstance(data_config.end_date, str)
            else data_config.end_date.strftime("%Y-%m-%d")
        )
        
        data_dict = fetcher.fetch_all_data(
            market_symbols=data_config.market_symbols,
            fred_series=data_config.fred_series,
            start_date=start_date,
            end_date=end_date,
        )
        
        # Combine market and economic data
        market_data = data_dict.get("market")
        if market_data is None:
            self.logger.error("No market data fetched")
            raise ValueError("Failed to fetch market data")
        
        data = market_data

        self.logger.info(f"Fetched data shape: {data.shape}")

        # Clean data
        cleaner = DataCleaner()
        data = cleaner.clean(data)
        data = create_features(data)

        self.logger.info(f"Cleaned data shape: {data.shape}")

        # Split into train/test
        split_idx = int(len(data) * 0.8)
        train_data = data.iloc[:split_idx].copy()
        test_data = data.iloc[split_idx:].copy()

        self.logger.info(
            f"Train shape: {train_data.shape}, Test shape: {test_data.shape}"
        )

        return data, train_data, test_data

    def run_zero_shot_comparison(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        target_col: str = "Close",
        prediction_length: int = 20,
        eval_config: Optional[EvalConfig] = None,
    ) -> dict:
        """Run zero-shot comparison between Chronos and baselines.

        Args:
            train_data: Training data
            test_data: Test data
            target_col: Target column name
            prediction_length: Number of periods to forecast
            eval_config: Evaluation configuration

        Returns:
            Dictionary with results
        """
        if eval_config is None:
            eval_config = EvalConfig()

        self.logger.info("=" * 80)
        self.logger.info("ZERO-SHOT FORECASTING COMPARISON")
        self.logger.info("=" * 80)

        results = {
            "timestamp": datetime.now().isoformat(),
            "target_col": target_col,
            "prediction_length": prediction_length,
            "model_results": {},
        }

        # Initialize models
        models = {
            "Naive": NaiveForecaster(),
            "Seasonal Naive": SeasonalNaiveForecaster(),
            "Mean": MeanForecaster(),
            "Exponential Smoothing": ExponentialSmoothingForecaster(),
            "ARIMA": ARIMAForecaster(),
            "Linear Regression": LinearRegressionForecaster(
                lags=20, horizon=1
            ),
            "VAR": VARForecaster(),
            "LSTM": LSTMForecaster(sequence_length=20, device="cpu"),
            "Ensemble": EnsembleForecaster(),
            "Chronos (zero-shot)": ChronosFinancialForecaster(
                prediction_length=prediction_length,
                device="cpu",
            ),
        }

        evaluator = ForecastEvaluator()

        # Fit models on training data
        self.logger.info("Fitting models...")
        for model_name, model in models.items():
            try:
                self.logger.info(f"  Fitting {model_name}...")
                if model_name == "Chronos (zero-shot)":
                    model.fit(train_data, target_col)
                elif model_name == "LSTM":
                    model.fit(
                        train_data,
                        target_col,
                        epochs=10,
                        batch_size=16,
                    )
                else:
                    model.fit(train_data, target_col)
                self.logger.info(f"  ✓ {model_name} fitted successfully")
            except Exception as e:
                self.logger.error(f"  ✗ {model_name} fit failed: {e}")

        # Generate forecasts on test data
        self.logger.info("Generating forecasts...")

        forecasts = {}
        for model_name, model in models.items():
            try:
                self.logger.info(f"  Forecasting with {model_name}...")

                if model_name == "Chronos (zero-shot)":
                    # Chronos returns dict with multiple outputs
                    pred_dict = model.forecast_zero_shot(
                        test_data,
                        target_col,
                        num_samples=100,
                    )
                    pred = pred_dict["median"][: len(test_data)]
                else:
                    pred = model.forecast(
                        test_data,
                        target_col,
                        prediction_length,
                    )

                # Ensure correct length
                if len(pred) > len(test_data):
                    pred = pred[: len(test_data)]
                elif len(pred) < len(test_data):
                    # Pad with last value
                    pred = np.pad(
                        pred,
                        (0, len(test_data) - len(pred)),
                        mode="edge",
                    )

                forecasts[model_name] = pred
                self.logger.info(f"  ✓ {model_name} forecast generated")

            except Exception as e:
                self.logger.error(f"  ✗ {model_name} forecast failed: {e}")

        # Evaluate forecasts
        self.logger.info("Evaluating forecasts...")

        actual = test_data[target_col].values

        evaluation_results = {}
        for model_name, pred in forecasts.items():
            try:
                metrics = evaluator.evaluate(
                    actual,
                    pred,
                    metrics=[
                        "mae",
                        "rmse",
                        "mase",
                        "mape",
                        "smape",
                        "directional_accuracy",
                    ],
                )
                evaluation_results[model_name] = metrics

                self.logger.info(f"  ✓ {model_name} evaluated")
                self.logger.info(f"    MAE: {metrics['mae']:.4f}")
                self.logger.info(f"    RMSE: {metrics['rmse']:.4f}")
                self.logger.info(
                    f"    Directional Accuracy: {metrics['directional_accuracy']:.4f}"
                )

            except Exception as e:
                self.logger.error(f"  ✗ {model_name} evaluation failed: {e}")

        results["model_results"] = evaluation_results
        results["forecasts"] = forecasts
        results["actual"] = actual

        return results

    def save_results(self, results: dict, experiment_name: str) -> None:
        """Save experiment results.

        Args:
            results: Results dictionary
            experiment_name: Name of experiment
        """
        # Save as JSON
        import json

        json_path = self.results_dir / f"{experiment_name}_results.json"

        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif isinstance(value, dict):
                json_results[key] = {
                    k: (
                        v.tolist() if isinstance(v, np.ndarray) else v
                    )
                    for k, v in value.items()
                }
            else:
                json_results[key] = value

        with open(json_path, "w") as f:
            json.dump(json_results, f, indent=2)

        self.logger.info(f"Results saved to {json_path}")

        # Save as CSV for easy viewing
        if "model_results" in results:
            csv_path = self.results_dir / f"{experiment_name}_metrics.csv"
            df = pd.DataFrame(results["model_results"]).T
            df.to_csv(csv_path)
            self.logger.info(f"Metrics saved to {csv_path}")

        # Save forecasts comparison
        if "forecasts" in results:
            forecast_path = self.results_dir / f"{experiment_name}_forecasts.csv"
            forecast_df = pd.DataFrame(
                {
                    "actual": results["actual"],
                    **results["forecasts"],
                }
            )
            forecast_df.to_csv(forecast_path, index=False)
            self.logger.info(f"Forecasts saved to {forecast_path}")

    def plot_results(
        self,
        results: dict,
        experiment_name: str,
    ) -> None:
        """Plot experiment results.

        Args:
            results: Results dictionary
            experiment_name: Name of experiment
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.logger.warning("matplotlib not available, skipping plots")
            return

        # Plot 1: Forecast comparison
        fig, ax = plt.subplots(figsize=(14, 6))

        actual = results["actual"]
        ax.plot(actual, "k-", linewidth=2, label="Actual", alpha=0.7)

        for model_name, forecast in results["forecasts"].items():
            ax.plot(forecast, label=model_name, alpha=0.7)

        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.set_title(f"Forecast Comparison - {experiment_name}")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        plot_path = self.results_dir / f"{experiment_name}_forecasts.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        self.logger.info(f"Plot saved to {plot_path}")
        plt.close()

        # Plot 2: Metrics comparison
        if "model_results" in results:
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            axes = axes.flatten()

            metrics_to_plot = [
                "mae",
                "rmse",
                "mase",
                "mape",
                "smape",
                "directional_accuracy",
            ]

            for idx, metric in enumerate(metrics_to_plot):
                if idx < len(axes):
                    ax = axes[idx]

                    # Extract metric values for each model
                    model_names = []
                    metric_values = []

                    for (
                        model_name,
                        metrics,
                    ) in results["model_results"].items():
                        if metric in metrics:
                            model_names.append(model_name)
                            metric_values.append(metrics[metric])

                    if metric_values:
                        ax.bar(range(len(model_names)), metric_values)
                        ax.set_xticks(range(len(model_names)))
                        ax.set_xticklabels(
                            model_names, rotation=45, ha="right"
                        )
                        ax.set_ylabel(metric.upper())
                        ax.set_title(metric.upper())
                        ax.grid(True, alpha=0.3, axis="y")

            plt.tight_layout()
            metrics_plot_path = self.results_dir / (
                f"{experiment_name}_metrics.png"
            )
            plt.savefig(metrics_plot_path, dpi=150, bbox_inches="tight")
            self.logger.info(f"Metrics plot saved to {metrics_plot_path}")
            plt.close()

    def run(
        self,
        data_config: Optional[DataConfig] = None,
        eval_config: Optional[EvalConfig] = None,
        target_col: str = "Close",
        prediction_length: int = 20,
        experiment_name: str = "zero_shot",
    ) -> dict:
        """Run complete zero-shot experiment.

        Args:
            data_config: Data configuration
            eval_config: Evaluation configuration
            target_col: Target column name
            prediction_length: Forecast horizon
            experiment_name: Name for experiment results

        Returns:
            Results dictionary
        """
        self.logger.info(
            f"Starting zero-shot experiment: {experiment_name}"
        )

        # Setup data
        _, train_data, test_data = self.setup_data(data_config)

        # Run comparison
        results = self.run_zero_shot_comparison(
            train_data,
            test_data,
            target_col=target_col,
            prediction_length=prediction_length,
            eval_config=eval_config,
        )

        # Save results
        self.save_results(results, experiment_name)

        # Plot results
        self.plot_results(results, experiment_name)

        self.logger.info("Experiment completed successfully!")
        self.logger.info("=" * 80)

        return results


def main():
    """Run Phase 3 zero-shot experiment."""
    experiment = ZeroShotExperiment(
        results_dir="results/phase3",
        log_dir="results/phase3/logs",
    )

    # Run with default configurations
    results = experiment.run(
        target_col="Close",
        prediction_length=20,
        experiment_name="zero_shot_default",
    )

    print("\n" + "=" * 80)
    print("PHASE 3 ZERO-SHOT EXPERIMENT COMPLETED")
    print("=" * 80)
    print(f"Results saved to: {experiment.results_dir}")


if __name__ == "__main__":
    main()
