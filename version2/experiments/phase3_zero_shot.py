"""Phase 3: Baselines & Zero-Shot Experiments."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.cleaning import DataCleaner, create_features
from src.data.fetchers import DataFetcher, get_sample_data
from src.eval.metrics import ForecastEvaluator, calculate_all_metrics
from src.models.baselines import (
    ARIMAForecaster,
    LinearForecaster,
    LSTMForecaster,
    NaiveForecaster,
    VARForecaster,
)
from src.models.chronos_wrapper import (
    ChronosFinancialForecaster,
    check_chronos_availability,
)
from src.preprocessing.tokenizer import AdvancedTokenizer, FinancialDataTokenizer
from src.utils.config import Config, load_config
from src.utils.logger import setup_logger

# Setup logging
logger = setup_logger("phase3_experiments")


class Phase3Experiments:
    """Phase 3 experiment runner for baselines and zero-shot forecasting."""

    def __init__(self, config: Config | None = None) -> None:
        """
        Initialize the experiment runner.

        Args:
            config: Configuration object
        """
        self.config = config or load_config()
        self.results: dict = {}

        # Initialize components
        self.data_fetcher = DataFetcher(self.config.data.fred_api_key)
        self.data_cleaner = DataCleaner(self.config.data.max_missing_ratio)
        self.evaluator = ForecastEvaluator(self.config.experiment.metrics)

        # Create results directory
        self.results_dir = self.config.results_dir / "phase3"
        self.results_dir.mkdir(exist_ok=True)

        logger.info("Phase 3 experiment runner initialized")

    def run_all_experiments(self) -> dict:
        """Run all Phase 3 experiments."""
        logger.info("Starting Phase 3 experiments")

        # Step 1: Data collection and preparation
        logger.info("Step 1: Data collection and preparation")
        data = self._prepare_data()

        if data.empty:
            logger.error("No data available for experiments")
            return {}

        # Step 2: Run baseline experiments
        logger.info("Step 2: Running baseline experiments")
        baseline_results = self._run_baseline_experiments(data)

        # Step 3: Run tokenization experiments
        logger.info("Step 3: Running tokenization experiments")
        tokenization_results = self._run_tokenization_experiments(data)

        # Step 4: Run zero-shot Chronos experiments
        logger.info("Step 4: Running zero-shot Chronos experiments")
        chronos_results = self._run_chronos_experiments(data)

        # Step 5: Compare results
        logger.info("Step 5: Comparing results")
        comparison_results = self._compare_results(
            baseline_results, tokenization_results, chronos_results
        )

        # Step 6: Generate reports
        logger.info("Step 6: Generating reports")
        self._generate_reports(comparison_results)

        # Combine all results
        self.results = {
            "baselines": baseline_results,
            "tokenization": tokenization_results,
            "chronos": chronos_results,
            "comparison": comparison_results,
        }

        logger.info("Phase 3 experiments completed")
        return self.results

    def _prepare_data(self) -> pd.DataFrame:
        """Prepare data for experiments."""
        logger.info("Preparing data for experiments")

        try:
            # Try to fetch real data first
            if self.config.data.fred_api_key:
                logger.info("Fetching real financial data")
                data_dict = self.data_fetcher.fetch_all_data(
                    self.config.data.market_symbols,
                    self.config.data.fred_series,
                    self.config.data.start_date,
                    self.config.data.end_date,
                )
            else:
                logger.info("FRED API key not available, using sample data")
                data_dict = {}

            # Use sample data if real data is not available
            if not data_dict:
                logger.info("Using sample data for experiments")
                data_dict = get_sample_data()

            # Clean and combine data
            cleaned_data = {}
            for source, df in data_dict.items():
                if source == "market":
                    cleaned_data[source] = self.data_cleaner.clean_market_data(df)
                elif source == "economic":
                    cleaned_data[source] = self.data_cleaner.clean_economic_data(df)

            # Combine datasets
            combined_data = self.data_cleaner.create_combined_dataset(
                cleaned_data, self.config.data.frequency
            )

            # Create additional features
            enhanced_data = create_features(combined_data)

            logger.info(f"Prepared data with shape: {enhanced_data.shape}")
            return enhanced_data

        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return pd.DataFrame()

    def _run_baseline_experiments(self, data: pd.DataFrame) -> dict:
        """Run baseline model experiments."""
        logger.info("Running baseline experiments")

        results: dict[str, dict] = {}

        # Define target columns to forecast
        target_cols = [
            col
            for col in data.columns
            if any(target in col for target in ["Close", "^GSPC", "^VIX"])
        ]

        if not target_cols:
            # Use first numeric column as target
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            target_cols = [numeric_cols[0]] if len(numeric_cols) > 0 else []

        if not target_cols:
            logger.warning("No suitable target columns found")
            return results

        # Use first target for experiments
        target_col = target_cols[0]
        logger.info(f"Using target column: {target_col}")

        # Split data
        train_size = int(len(data) * (1 - self.config.model.test_size))
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]

        if len(test_data) < self.config.model.prediction_length:
            logger.warning("Test data too small, adjusting split")
            train_data = data.iloc[: -self.config.model.prediction_length]
            test_data = data.iloc[-self.config.model.prediction_length :]

        # Define baseline models
        models = {
            "naive_last": NaiveForecaster(
                prediction_length=self.config.model.prediction_length, method="last"
            ),
            "naive_mean": NaiveForecaster(
                prediction_length=self.config.model.prediction_length, method="mean"
            ),
            "linear": LinearForecaster(
                prediction_length=self.config.model.prediction_length
            ),
        }

        # Add ARIMA if we have enough data
        if len(train_data) >= 50:
            models["arima"] = ARIMAForecaster(
                prediction_length=self.config.model.prediction_length
            )

        # Add VAR if we have multiple variables
        if len(train_data.select_dtypes(include=[np.number]).columns) >= 2:
            models["var"] = VARForecaster(
                prediction_length=self.config.model.prediction_length
            )

        # Add LSTM if we have enough data and PyTorch is available
        if len(train_data) >= 200:
            try:
                models["lstm"] = LSTMForecaster(
                    prediction_length=self.config.model.prediction_length,
                    epochs=10,  # Reduced for faster experiments
                )
            except Exception as e:
                logger.warning(f"Could not initialize LSTM: {e}")

        # Run experiments for each model
        for model_name, model in models.items():
            logger.info(f"Running {model_name} experiment")

            try:
                # Fit and predict
                model.fit(train_data, target_col)
                predictions = model.predict(test_data)

                # Get true values
                true_values = (
                    test_data[target_col].dropna().iloc[: len(predictions)].values
                )

                if len(true_values) != len(predictions):
                    min_len = min(len(true_values), len(predictions))
                    true_values = true_values[:min_len]
                    predictions = predictions[:min_len]

                # Calculate metrics
                metrics = calculate_all_metrics(
                    true_values, predictions, train_data[target_col].dropna().values
                )

                results[model_name] = {
                    "predictions": predictions,
                    "true_values": true_values,
                    "metrics": metrics,
                }

                logger.info(f"{model_name} - MAE: {metrics.get('mae', 'N/A'):.4f}")

            except Exception as e:
                logger.error(f"Error running {model_name}: {e}")
                results[model_name] = {"error": str(e)}

        return results

    def _run_tokenization_experiments(self, data: pd.DataFrame) -> dict:
        """Run tokenization strategy experiments."""
        logger.info("Running tokenization experiments")

        results: dict[str, dict] = {}

        # Define tokenization strategies
        tokenizers = {
            "uniform": FinancialDataTokenizer(
                num_bins=self.config.model.num_bins,
                method="uniform",
                context_length=self.config.model.context_length,
            ),
            "quantile": FinancialDataTokenizer(
                num_bins=self.config.model.num_bins,
                method="quantile",
                context_length=self.config.model.context_length,
            ),
            "advanced": AdvancedTokenizer(
                num_bins=self.config.model.num_bins,
                method="quantile",
                context_length=self.config.model.context_length,
                include_technical_indicators=True,
                include_time_features=True,
            ),
        }

        # Add k-means tokenizer if sklearn is available
        try:
            tokenizers["kmeans"] = FinancialDataTokenizer(
                num_bins=self.config.model.num_bins,
                method="kmeans",
                context_length=self.config.model.context_length,
            )
        except Exception as e:
            logger.warning(f"Could not initialize k-means tokenizer: {e}")

        # Split data
        train_size = int(len(data) * (1 - self.config.model.test_size))
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]

        # Run experiments for each tokenizer
        for tokenizer_name, tokenizer in tokenizers.items():
            logger.info(f"Testing {tokenizer_name} tokenization")

            try:
                # Fit tokenizer
                tokenizer.fit(train_data)

                # Transform data
                train_tokens = tokenizer.transform(train_data)
                test_tokens = tokenizer.transform(test_data)

                # Analyze tokenization quality
                analysis = self._analyze_tokenization(
                    tokenizer, train_data, train_tokens, test_tokens
                )

                results[tokenizer_name] = {
                    "tokenizer": tokenizer,
                    "train_tokens": train_tokens,
                    "test_tokens": test_tokens,
                    "analysis": analysis,
                }

                logger.info(f"{tokenizer_name} tokenization completed")

            except Exception as e:
                logger.error(f"Error with {tokenizer_name} tokenization: {e}")
                results[tokenizer_name] = {"error": str(e)}

        return results

    def _run_chronos_experiments(self, data: pd.DataFrame) -> dict:
        """Run Chronos zero-shot experiments."""
        logger.info("Running Chronos zero-shot experiments")

        results = {}

        # Check if Chronos is available
        chronos_available = check_chronos_availability()
        logger.info(f"Chronos availability: {chronos_available}")

        # Define target column
        target_cols = [
            col
            for col in data.columns
            if any(target in col for target in ["Close", "^GSPC", "^VIX"])
        ]

        if not target_cols:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            target_cols = [numeric_cols[0]] if len(numeric_cols) > 0 else []

        if not target_cols:
            logger.warning("No suitable target columns found for Chronos")
            return results

        target_col = target_cols[0]

        # Split data
        train_size = int(len(data) * (1 - self.config.model.test_size))
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]

        # Test different Chronos configurations
        chronos_configs = [
            {
                "name": "chronos_small",
                "model_name": "amazon/chronos-t5-small",
                "tokenizer_config": {"method": "quantile", "num_bins": 1024},
            }
        ]

        # Run experiments for each configuration
        for config in chronos_configs:
            config_name = config["name"]
            logger.info(f"Testing {config_name}")

            try:
                # Initialize Chronos forecaster
                forecaster = ChronosFinancialForecaster(
                    model_name=config["model_name"],
                    prediction_length=self.config.model.prediction_length,
                    context_length=self.config.model.context_length,
                )

                # Fit (mainly for tokenizer)
                forecaster.fit(train_data, target_col, config["tokenizer_config"])

                # Generate zero-shot forecast
                predictions = forecaster.forecast_zero_shot(test_data)

                # Get true values
                true_values = (
                    test_data[target_col].dropna().iloc[: len(predictions)].values
                )

                if len(true_values) != len(predictions):
                    min_len = min(len(true_values), len(predictions))
                    true_values = true_values[:min_len]
                    predictions = predictions[:min_len]

                # Calculate metrics
                metrics = calculate_all_metrics(
                    true_values, predictions, train_data[target_col].dropna().values
                )

                results[config_name] = {
                    "forecaster": forecaster,
                    "predictions": predictions,
                    "true_values": true_values,
                    "metrics": metrics,
                    "config": config,
                }

                logger.info(f"{config_name} - MAE: {metrics.get('mae', 'N/A'):.4f}")

            except Exception as e:
                logger.error(f"Error with {config_name}: {e}")
                results[config_name] = {"error": str(e)}

        return results

    def _analyze_tokenization(
        self,
        tokenizer: FinancialDataTokenizer,
        data: pd.DataFrame,
        train_tokens: dict,
        test_tokens: dict,
    ) -> dict:
        """Analyze tokenization quality."""
        analysis = {}

        try:
            # Vocabulary usage
            combined_tokens = train_tokens.get("combined", np.array([]))
            if len(combined_tokens) > 0:
                unique_tokens = len(np.unique(combined_tokens))
                total_tokens = len(combined_tokens)
                vocab_usage = unique_tokens / tokenizer.num_bins

                analysis["vocab_usage"] = vocab_usage
                analysis["unique_tokens"] = unique_tokens
                analysis["total_tokens"] = total_tokens

            # Reconstruction quality (for first feature)
            if tokenizer.feature_names:
                feature_name = tokenizer.feature_names[0]
                if feature_name in train_tokens["individual"]:
                    tokens = train_tokens["individual"][feature_name]
                    reconstructed = tokenizer.inverse_transform(tokens, feature_name)
                    original = (
                        data[feature_name].dropna().iloc[: len(reconstructed)].values
                    )

                    if len(original) == len(reconstructed):
                        reconstruction_error = np.mean(np.abs(original - reconstructed))
                        analysis["reconstruction_error"] = reconstruction_error

        except Exception as e:
            logger.warning(f"Error in tokenization analysis: {e}")
            analysis["error"] = str(e)

        return analysis

    def _compare_results(
        self, baseline_results: dict, tokenization_results: dict, chronos_results: dict
    ) -> dict:
        """Compare results across all experiments."""
        logger.info("Comparing experimental results")

        comparison: Dict[str, Dict] = {
            "summary": {},
            "best_models": {},
            "metric_comparison": {},
        }

        # Collect all model results
        all_results = {}

        # Add baseline results
        for model_name, result in baseline_results.items():
            if "metrics" in result:
                all_results[f"baseline_{model_name}"] = result["metrics"]

        # Add Chronos results
        for model_name, result in chronos_results.items():
            if "metrics" in result:
                all_results[f"chronos_{model_name}"] = result["metrics"]

        if not all_results:
            logger.warning("No valid results to compare")
            return comparison

        # Find best models for each metric
        metrics = ["mae", "rmse", "mase", "smape", "directional_accuracy"]

        for metric in metrics:
            metric_values = {}
            for model_name, model_metrics in all_results.items():
                if metric in model_metrics and not np.isnan(model_metrics[metric]):
                    metric_values[model_name] = model_metrics[metric]

            if metric_values:
                if metric == "directional_accuracy":
                    # Higher is better
                    best_model = max(
                        metric_values.keys(), key=lambda k: metric_values[k]
                    )
                else:
                    # Lower is better
                    best_model = min(
                        metric_values.keys(), key=lambda k: metric_values[k]
                    )

                comparison["best_models"][metric] = {
                    "model": best_model,
                    "value": metric_values[best_model],
                }
                comparison["metric_comparison"][metric] = metric_values

        # Summary statistics
        comparison["summary"] = {
            "total_models_tested": len(all_results),
            "baseline_models": len(baseline_results),
            "chronos_models": len(chronos_results),
            "tokenization_strategies": len(tokenization_results),
        }

        return comparison

    def _generate_reports(self, comparison_results: dict) -> None:
        """Generate experiment reports and visualizations."""
        logger.info("Generating experiment reports")

        try:
            # Create summary report
            self._create_summary_report(comparison_results)

            # Create visualizations
            self._create_visualizations(comparison_results)

            logger.info(f"Reports saved to {self.results_dir}")

        except Exception as e:
            logger.error(f"Error generating reports: {e}")

    def _create_summary_report(self, comparison_results: dict) -> None:
        """Create a summary report of the experiments."""
        report_path = self.results_dir / "phase3_summary_report.txt"

        with open(report_path, "w") as f:
            f.write("Phase 3 Experiments Summary Report\n")
            f.write("=" * 50 + "\n\n")

            # Summary statistics
            summary = comparison_results.get("summary", {})
            f.write("Summary Statistics:\n")
            f.write("-" * 20 + "\n")
            f.writelines(f"{key}: {value}\n" for key, value in summary.items())
            f.write("\n")

            # Best models per metric
            best_models = comparison_results.get("best_models", {})
            f.write("Best Models by Metric:\n")
            f.write("-" * 25 + "\n")
            f.writelines(
                f"{metric}: {info['model']} ({info['value']:.4f})\n"
                for metric, info in best_models.items()
            )
            f.write("\n")

            # Metric comparison
            metric_comparison = comparison_results.get("metric_comparison", {})
            f.write("Detailed Metric Comparison:\n")
            f.write("-" * 30 + "\n")
            for metric, values in metric_comparison.items():
                f.write(f"\n{metric.upper()}:\n")
                sorted_models = sorted(values.items(), key=lambda x: x[1])
                f.writelines(
                    f"  {model}: {value:.4f}\n" for model, value in sorted_models
                )

    def _create_visualizations(self, comparison_results: dict) -> None:
        """Create visualizations of the results."""
        try:
            # Metric comparison plot
            metric_comparison = comparison_results.get("metric_comparison", {})

            if metric_comparison:
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                axes = axes.flatten()

                for i, (metric, values) in enumerate(metric_comparison.items()):
                    if i >= len(axes):
                        break

                    models = list(values.keys())
                    scores = list(values.values())

                    axes[i].bar(models, scores)
                    axes[i].set_title(f"{metric.upper()}")
                    axes[i].tick_params(axis="x", rotation=45)

                # Remove empty subplots
                for j in range(i + 1, len(axes)):
                    fig.delaxes(axes[j])

                plt.tight_layout()
                plt.savefig(
                    self.results_dir / "metric_comparison.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

        except Exception as e:
            logger.warning(f"Could not create visualizations: {e}")


def main() -> None:
    """Main function to run Phase 3 experiments."""
    logger.info("Starting Phase 3 experiments")

    # Load configuration
    config = load_config()

    # Initialize and run experiments
    experiments = Phase3Experiments(config)
    results = experiments.run_all_experiments()

    # Print summary
    if results:
        logger.info("Phase 3 experiments completed successfully")

        # Print best models
        comparison = results.get("comparison", {})
        best_models = comparison.get("best_models", {})

        if best_models:
            logger.info("Best models by metric:")
            for metric, info in best_models.items():
                logger.info(f"  {metric}: {info['model']} ({info['value']:.4f})")
    else:
        logger.error("Phase 3 experiments failed")


if __name__ == "__main__":
    main()
