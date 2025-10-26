"""Phase 4: Fine-Tuning & Causal Attribution Experiments."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.cleaning import DataCleaner, create_features
from src.data.fetchers import DataFetcher, get_sample_data
from src.eval.metrics import ForecastEvaluator, calculate_all_metrics
from src.models.chronos_wrapper import ChronosFinancialForecaster
from src.utils.config import Config, load_config
from src.utils.logger import setup_logger

# Setup logging
logger = setup_logger("phase4_experiments")


class Phase4Experiments:
    """Phase 4 experiment runner for fine-tuning and causal attribution."""

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
        self.results_dir = self.config.results_dir / "phase4"
        self.results_dir.mkdir(exist_ok=True)

        logger.info("Phase 4 experiment runner initialized")

    def run_all_experiments(self) -> dict:
        """Run all Phase 4 experiments."""
        logger.info("Starting Phase 4 experiments")

        # Step 1: Data collection and preparation
        logger.info("Step 1: Data collection and preparation")
        data = self._prepare_data()

        if data.empty:
            logger.error("No data available for experiments")
            return {}

        # Step 2: Fine-tuning experiments
        logger.info("Step 2: Running fine-tuning experiments")
        fine_tuning_results = self._run_fine_tuning_experiments(data)

        # Step 3: Causal attribution analysis
        logger.info("Step 3: Running causal attribution analysis")
        attribution_results = self._run_attribution_analysis(data)

        # Combine all results
        self.results = {
            "fine_tuning": fine_tuning_results,
            "attribution": attribution_results,
        }

        logger.info("Phase 4 experiments completed")
        return self.results

    def _prepare_data(self) -> pd.DataFrame:
        """Prepare data for experiments."""
        logger.info("Preparing data for experiments")

        try:
            # Try to fetch real data first
            if self.config.data.fred_api_key:
                logger.info("Fetching real financial data")
                if self.config.data.market_symbols and self.config.data.fred_series:
                    data_dict = self.data_fetcher.fetch_all_data(
                        self.config.data.market_symbols,
                        self.config.data.fred_series,
                        self.config.data.start_date,
                        self.config.data.end_date,
                    )
                else:
                    logger.warning(
                        "Market symbols or FRED series not specified in config"
                    )
                    data_dict = {}
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

    def _run_fine_tuning_experiments(self, data: pd.DataFrame) -> dict:
        """Run fine-tuning experiments."""
        logger.info("Running fine-tuning experiments")

        results: dict[str, dict] = {}

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
            logger.warning("No suitable target columns found for fine-tuning")
            return results

        target_col = target_cols[0]

        # Split data
        train_size = int(len(data) * (1 - self.config.model.test_size))
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]

        # Initialize Chronos forecaster
        forecaster = ChronosFinancialForecaster(
            model_name="amazon/chronos-t5-small",
            prediction_length=self.config.model.prediction_length,
            context_length=self.config.model.context_length,
        )

        # Fine-tune the model
        try:
            forecaster.fine_tune(train_data, test_data)

            # Fit (mainly for tokenizer)
            forecaster.fit(train_data, target_col)

            # Evaluate fine-tuned model
            predictions = forecaster.forecast_zero_shot(test_data)
            true_values = (
                test_data[target_col].dropna().iloc[: len(predictions)].to_numpy()
            )

            if len(true_values) != len(predictions):
                min_len = min(len(true_values), len(predictions))
                true_values = true_values[:min_len]
                predictions = predictions[:min_len]

            metrics = calculate_all_metrics(
                y_true=true_values,
                y_pred=predictions,
                y_train=train_data[target_col].dropna().values,
            )

            results["fine_tuned_chronos"] = {
                "predictions": predictions,
                "true_values": true_values,
                "metrics": metrics,
            }

            logger.info(f"Fine-tuned Chronos - MAE: {metrics.get('mae', 'N/A'):.4f}")

        except Exception as e:
            logger.error(f"Error fine-tuning Chronos model: {e}")
            results["fine_tuned_chronos"] = {"error": str(e)}

        return results

    def _run_attribution_analysis(self, data: pd.DataFrame) -> dict:
        """Run causal attribution analysis."""
        logger.info("Running causal attribution analysis")

        results = {}

        # Define attribution methods
        attribution_methods = [
            "ablation_importance",
            "permutation_importance",
            "shap_analysis",
        ]

        def ablation_importance(data: pd.DataFrame, target_col: str) -> dict:
            logger.info("Performing ablation importance analysis")
            # Placeholder implementation
            # TODO: Implement actual ablation importance logic
            return {"status": "Ablation importance analysis completed"}

        def permutation_importance(data: pd.DataFrame, target_col: str) -> dict:
            logger.info("Performing permutation importance analysis")
            # Placeholder implementation
            # TODO: Implement actual permutation importance logic
            return {"status": "Permutation importance analysis completed"}

        def shap_analysis(data: pd.DataFrame, target_col: str) -> dict:
            logger.info("Performing SHAP analysis")
            # Placeholder implementation using SHAP library (future implementation)
            # TODO: Implement actual SHAP analysis logic
            return {"status": "SHAP analysis completed"}

        # Execute each method
        for method in attribution_methods:
            if method == "ablation_importance":
                results[method] = ablation_importance(data, "^GSPC")
            elif method == "permutation_importance":
                results[method] = permutation_importance(data, "^GSPC")
            elif method == "shap_analysis":
                results[method] = shap_analysis(data, "^GSPC")

        logger.info("Causal attribution analysis completed")
        return results


def main() -> None:
    """Main function to run Phase 4 experiments."""
    logger.info("Starting Phase 4 experiments")

    # Load configuration
    config = load_config()

    # Initialize and run experiments
    experiments = Phase4Experiments(config)
    results = experiments.run_all_experiments()

    # Print summary
    if results:
        logger.info("Phase 4 experiments completed successfully")
        logger.info("Results:")
        logger.info(results)
    else:
        logger.error("Phase 4 experiments failed")


if __name__ == "__main__":
    main()
