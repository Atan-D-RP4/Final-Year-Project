"""
Comprehensive Demo of Enhanced Chronos Forecasting Framework
Demonstrates all major features including IMF data, advanced preprocessing, cross-validation, and visualization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
import warnings
import os

# Import our enhanced modules
from enhanced_data_preparation import EnhancedBehavioralDataLoader
from advanced_tokenizer import AdvancedTokenizer
from cross_validation import BacktestingFramework, TimeSeriesCrossValidator
from baseline_models import BaselineComparison
from visualization import ForecastVisualizer, create_forecast_report
from chronos_behavioral_framework import ChronosBehavioralForecaster, BenchmarkRunner

warnings.filterwarnings("ignore")


class ComprehensiveForecaster:
    """
    Enhanced forecaster with all advanced features
    """

    def __init__(
        self,
        model_name: str = "amazon/chronos-bolt-small",
        device: str = "cpu",
        use_advanced_tokenizer: bool = True,
        tokenizer_config: Dict = None,
    ):
        """
        Initialize comprehensive forecaster

        Args:
            model_name: Chronos model name
            device: Device to run on
            use_advanced_tokenizer: Whether to use advanced tokenizer
            tokenizer_config: Configuration for advanced tokenizer
        """
        self.model_name = model_name
        self.device = device
        self.use_advanced_tokenizer = use_advanced_tokenizer

        # Initialize base forecaster
        self.base_forecaster = ChronosBehavioralForecaster(model_name, device)

        # Initialize advanced tokenizer if requested
        if use_advanced_tokenizer:
            default_config = {
                "window_size": 10,
                "quantization_levels": 1000,
                "scaling_method": "standard",
                "feature_selection": True,
                "max_features": 15,
                "economic_features": True,
            }
            if tokenizer_config:
                default_config.update(tokenizer_config)

            self.advanced_tokenizer = AdvancedTokenizer(**default_config)
        else:
            self.advanced_tokenizer = None

    def prepare_data(self, data, window_size: int = 10):
        """Prepare data using advanced or basic tokenizer"""
        if self.use_advanced_tokenizer and self.advanced_tokenizer:
            # Use advanced tokenizer
            tokenized_data = self.advanced_tokenizer.fit_transform(data)
            return tokenized_data, self.advanced_tokenizer
        else:
            # Use basic tokenizer
            return self.base_forecaster.prepare_data(data, window_size)

    def forecast_zero_shot(self, context_data, prediction_length: int = 5):
        """Make zero-shot forecast"""
        return self.base_forecaster.forecast_zero_shot(context_data, prediction_length)


def run_imf_data_demo():
    """
    Demonstrate IMF data loading and forecasting
    """
    print("=" * 80)
    print("PHASE 1: IMF DATA INTEGRATION DEMO")
    print("=" * 80)

    # Initialize enhanced data loader
    loader = EnhancedBehavioralDataLoader()

    # Load different types of IMF data
    print("\\n1. Loading IMF Economic Data...")

    datasets = {}

    # US Inflation data
    try:
        inflation_data = loader.load_imf_inflation_data(["US"], 2010, 2023, "US")
        datasets["US_Inflation"] = inflation_data
        print(
            f"   ✓ US Inflation: {len(inflation_data)} points, range: [{min(inflation_data):.2f}%, {max(inflation_data):.2f}%]"
        )
    except Exception as e:
        print(f"   ✗ US Inflation: {e}")

    # US GDP Growth data
    try:
        gdp_data = loader.load_imf_gdp_growth_data(["US"], 2010, 2023, "US")
        datasets["US_GDP_Growth"] = gdp_data
        print(
            f"   ✓ US GDP Growth: {len(gdp_data)} points, range: [{min(gdp_data):.2f}%, {max(gdp_data):.2f}%]"
        )
    except Exception as e:
        print(f"   ✗ US GDP Growth: {e}")

    # Exchange rate data
    try:
        fx_data = loader.load_imf_exchange_rate_data("USD", "EUR", 2010, 2023)
        datasets["USD_EUR"] = fx_data
        print(
            f"   ✓ USD/EUR: {len(fx_data)} points, range: [{min(fx_data):.4f}, {max(fx_data):.4f}]"
        )
    except Exception as e:
        print(f"   ✗ USD/EUR: {e}")

    # Multivariate data
    try:
        mv_data = loader.load_imf_multivariate_data(
            "US", ["gdp_growth", "inflation"], 2010, 2023
        )
        print(f"   ✓ Multivariate data: {list(mv_data.keys())}")
        datasets.update(mv_data)
    except Exception as e:
        print(f"   ✗ Multivariate data: {e}")

    return datasets


def run_advanced_preprocessing_demo(datasets: Dict[str, List[float]]):
    """
    Demonstrate advanced preprocessing capabilities
    """
    print("\\n" + "=" * 80)
    print("PHASE 2: ADVANCED PREPROCESSING DEMO")
    print("=" * 80)

    if not datasets:
        print("No datasets available for preprocessing demo")
        return None

    # Select a dataset for demonstration
    dataset_name = list(datasets.keys())[0]
    data = datasets[dataset_name]

    print(f"\\nUsing dataset: {dataset_name} ({len(data)} points)")

    # Test different tokenizer configurations
    configs = [
        {
            "name": "Basic",
            "config": {
                "window_size": 8,
                "scaling_method": "standard",
                "feature_selection": False,
                "economic_features": False,
            },
        },
        {
            "name": "Advanced",
            "config": {
                "window_size": 10,
                "scaling_method": "robust",
                "feature_selection": True,
                "max_features": 15,
                "economic_features": True,
            },
        },
        {
            "name": "Economic Focus",
            "config": {
                "window_size": 12,
                "scaling_method": "standard",
                "feature_selection": True,
                "max_features": 20,
                "economic_features": True,
            },
        },
    ]

    tokenizer_results = {}

    for config in configs:
        print(f"\\nTesting {config['name']} tokenizer...")
        try:
            tokenizer = AdvancedTokenizer(**config["config"])
            tokenized = tokenizer.fit_transform(data)

            feature_names = tokenizer.get_feature_names()
            importance = tokenizer.get_feature_importance()

            tokenizer_results[config["name"]] = {
                "tokenized_shape": tokenized.shape,
                "n_features": len(feature_names),
                "top_features": list(importance.keys())[:5] if importance else [],
            }

            print(f"   ✓ Output shape: {tokenized.shape}")
            print(f"   ✓ Features: {len(feature_names)}")
            if importance:
                print(f"   ✓ Top features: {list(importance.keys())[:3]}")

        except Exception as e:
            print(f"   ✗ Error: {e}")

    return tokenizer_results


def run_forecasting_comparison_demo(datasets: Dict[str, List[float]]):
    """
    Demonstrate forecasting with different models and configurations
    """
    print("\\n" + "=" * 80)
    print("PHASE 3: FORECASTING COMPARISON DEMO")
    print("=" * 80)

    if not datasets:
        print("No datasets available for forecasting demo")
        return None

    # Select dataset
    dataset_name = list(datasets.keys())[0]
    data = datasets[dataset_name]

    print(f"\\nForecasting on: {dataset_name} ({len(data)} points)")

    # Split data
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    test_data = data[train_size:]

    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    # Test different Chronos configurations
    chronos_configs = [
        {
            "name": "Chronos Basic",
            "use_advanced_tokenizer": False,
            "tokenizer_config": None,
        },
        {
            "name": "Chronos Advanced",
            "use_advanced_tokenizer": True,
            "tokenizer_config": {
                "window_size": 10,
                "scaling_method": "robust",
                "feature_selection": True,
                "max_features": 15,
                "economic_features": True,
            },
        },
    ]

    chronos_results = {}

    for config in chronos_configs:
        print(f"Testing {config['name']}...")
        try:
            forecaster = ComprehensiveForecaster(
                model_name="amazon/chronos-bolt-small",
                device="cpu",
                use_advanced_tokenizer=config["use_advanced_tokenizer"],
                tokenizer_config=config["tokenizer_config"],
            )

            # Prepare data and forecast
            tokenized_data, tokenizer = forecaster.prepare_data(train_data, 10)
            forecast_result = forecaster.forecast_zero_shot(
                tokenized_data, len(test_data)
            )

            predictions = forecast_result["mean"][0].numpy()

            # Calculate metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error

            mse = mean_squared_error(test_data, predictions)
            mae = mean_absolute_error(test_data, predictions)

            chronos_results[config["name"]] = {
                "mse": mse,
                "mae": mae,
                "predictions": predictions.tolist(),
            }

            print(f"   ✓ MSE: {mse:.6f}, MAE: {mae:.6f}")

        except Exception as e:
            print(f"   ✗ Error: {e}")

    # Compare with baseline models
    print("Testing baseline models...")
    baseline_comparison = BaselineComparison()
    baseline_comparison.add_standard_models()

    try:
        baseline_results = baseline_comparison.compare_models(
            train_data, test_data, show_progress=False
        )
        print("   ✓ Baseline models completed")
    except Exception as e:
        print(f"   ✗ Baseline error: {e}")
        baseline_results = {}

    return {
        "chronos_results": chronos_results,
        "baseline_results": baseline_results,
        "test_data": test_data,
        "train_data": train_data,
    }


def run_cross_validation_demo(datasets: Dict[str, List[float]]):
    """
    Demonstrate cross-validation framework
    """
    print("" + "=" * 80)
    print("PHASE 4: CROSS-VALIDATION DEMO")
    print("=" * 80)

    if not datasets:
        print("No datasets available for cross-validation demo")
        return None

    # Select dataset
    dataset_name = list(datasets.keys())[0]
    data = datasets[dataset_name]

    print(f"Cross-validating on: {dataset_name} ({len(data)} points)")

    # Initialize forecaster
    forecaster = ComprehensiveForecaster(
        use_advanced_tokenizer=True,
        tokenizer_config={
            "window_size": 8,
            "scaling_method": "robust",
            "feature_selection": True,
            "max_features": 12,
            "economic_features": True,
        },
    )

    # Initialize backtesting framework
    backtest = BacktestingFramework(
        forecaster=forecaster,
        cv_strategy="expanding",
        n_splits=5,
        prediction_length=3,
        window_size=8,
    )

    print("Running cross-validation...")
    try:
        cv_results = backtest.run_backtest(
            data, show_progress=True, return_predictions=True
        )
        backtest.print_results()

        # Compare CV strategies
        print("Comparing CV strategies...")
        strategy_comparison = backtest.compare_strategies(
            data, ["expanding", "sliding"]
        )

        return cv_results

    except Exception as e:
        print(f"Cross-validation error: {e}")
        return None


def run_visualization_demo(forecast_results: Dict[str, Any]):
    """
    Demonstrate visualization capabilities
    """
    print("" + "=" * 80)
    print("PHASE 5: VISUALIZATION DEMO")
    print("=" * 80)

    if not forecast_results:
        print("No forecast results available for visualization")
        return

    # Initialize visualizer
    visualizer = ForecastVisualizer()

    # Extract data
    chronos_results = forecast_results.get("chronos_results", {})
    baseline_results = forecast_results.get("baseline_results", {})
    test_data = forecast_results.get("test_data", [])
    train_data = forecast_results.get("train_data", [])

    if not test_data:
        print("No test data available for visualization")
        return

    print("1. Creating forecast visualization...")

    # Get best Chronos result
    if chronos_results:
        best_chronos = min(chronos_results.items(), key=lambda x: x[1]["mse"])
        best_chronos_name, best_chronos_data = best_chronos

        # Plot forecast results
        fig1 = visualizer.plot_forecast_results(
            test_data,
            best_chronos_data["predictions"],
            train_data,
            title=f"{best_chronos_name} Forecast Results",
        )
        plt.show()

        # Plot residual analysis
        print("2. Creating residual analysis...")
        fig2 = visualizer.plot_residual_analysis(
            test_data,
            best_chronos_data["predictions"],
            title=f"{best_chronos_name} Residual Analysis",
        )
        plt.show()

    # Plot model comparison
    if chronos_results or baseline_results:
        print("3. Creating model comparison...")

        # Combine results
        all_results = {}
        all_results.update(chronos_results)

        # Add baseline results (convert format)
        for model_name, metrics in baseline_results.items():
            if "mse" in metrics and "mae" in metrics:
                all_results[model_name] = {"mse": metrics["mse"], "mae": metrics["mae"]}

        if all_results:
            fig3 = visualizer.plot_model_comparison(
                all_results,
                metrics=["mse", "mae"],
                title="Model Performance Comparison",
            )
            plt.show()

    # Create comprehensive report
    if chronos_results:
        print("4. Creating comprehensive report...")
        best_chronos_name, best_chronos_data = min(
            chronos_results.items(), key=lambda x: x[1]["mse"]
        )

        report_path = create_forecast_report(
            test_data,
            best_chronos_data["predictions"],
            train_data,
            model_name=best_chronos_name,
            save_dir="comprehensive_demo_report",
        )

        print(f"   ✓ Report saved to: {report_path}")


def main():
    """
    Run comprehensive demo of all framework features
    """
    print("COMPREHENSIVE CHRONOS FORECASTING FRAMEWORK DEMO")
    print("=" * 80)
    print("This demo showcases all enhanced features:")
    print("• Phase 1: IMF Data Integration")
    print("• Phase 2: Advanced Preprocessing")
    print("• Phase 3: Forecasting Comparison")
    print("• Phase 4: Cross-Validation")
    print("• Phase 5: Visualization")
    print("=" * 80)

    # Phase 1: IMF Data Integration
    datasets = run_imf_data_demo()

    if not datasets:
        print("No datasets loaded. Generating synthetic data for demo...")
        # Generate synthetic economic data
        np.random.seed(42)
        n_samples = 30

        # Synthetic inflation data
        inflation = (
            2.5
            + 1.5 * np.sin(np.linspace(0, 4 * np.pi, n_samples))
            + 0.5 * np.random.randn(n_samples)
        )
        inflation = np.maximum(inflation, 0.5)  # Ensure positive

        # Synthetic GDP growth data
        gdp_growth = (
            2.0
            + 1.0 * np.sin(np.linspace(0, 2 * np.pi, n_samples))
            + 0.8 * np.random.randn(n_samples)
        )

        datasets = {
            "Synthetic_Inflation": inflation.tolist(),
            "Synthetic_GDP_Growth": gdp_growth.tolist(),
        }

        print(f"Generated {len(datasets)} synthetic datasets")

    # Phase 2: Advanced Preprocessing
    tokenizer_results = run_advanced_preprocessing_demo(datasets)

    # Phase 3: Forecasting Comparison
    forecast_results = run_forecasting_comparison_demo(datasets)

    # Phase 4: Cross-Validation
    cv_results = run_cross_validation_demo(datasets)

    # Phase 5: Visualization
    if forecast_results:
        run_visualization_demo(forecast_results)

    # Summary
    print("" + "=" * 80)
    print("DEMO SUMMARY")
    print("=" * 80)
    print(f"✓ Datasets processed: {len(datasets)}")
    print(
        f"✓ Tokenizer configurations tested: {len(tokenizer_results) if tokenizer_results else 0}"
    )
    print(
        f"✓ Forecasting models compared: {len(forecast_results['chronos_results']) + len(forecast_results['baseline_results']) if forecast_results else 0}"
    )
    print(f"✓ Cross-validation completed: {'Yes' if cv_results else 'No'}")
    print(f"✓ Visualizations created: {'Yes' if forecast_results else 'No'}")

    print("🎉 Comprehensive demo completed successfully!")
    print("Next steps:")
    print("• Explore the generated reports in 'comprehensive_demo_report/'")
    print("• Modify configurations in the demo functions")
    print("• Try with your own datasets")
    print(
        "• Experiment with different model sizes (chronos-bolt-base, chronos-bolt-large)"
    )

    return {
        "datasets": datasets,
        "tokenizer_results": tokenizer_results,
        "forecast_results": forecast_results,
        "cv_results": cv_results,
    }


if __name__ == "__main__":
    # Import matplotlib for visualization
    import matplotlib.pyplot as plt

    # Run comprehensive demo
    results = main()

