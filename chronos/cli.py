"""
Command Line Interface for Chronos Forecasting Framework
Provides easy access to all framework features through command line
"""

import traceback

import argparse
import json
import os
import sys
from typing import List
import warnings

import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Import framework modules
from enhanced_data_preparation import EnhancedBehavioralDataLoader
from cross_validation import BacktestingFramework
from baseline_models import BaselineComparison
from visualization import ForecastVisualizer, create_forecast_report
from comprehensive_demo import ComprehensiveForecaster
from comprehensive_demo import main as run_comprehensive_demo


warnings.filterwarnings("ignore")


class ChronosCLI:
    """
    Command Line Interface for Chronos Forecasting Framework
    """

    def __init__(self):
        self.data_loader = EnhancedBehavioralDataLoader()
        self.visualizer = ForecastVisualizer()

    def load_data(self, args) -> List[float]:
        """Load data based on CLI arguments"""
        if args.data_type == "imf_inflation":
            return self.data_loader.load_imf_inflation_data(
                countries=[args.country] if args.country else ["US"],
                start_year=args.start_year,
                end_year=args.end_year,
                country_to_use=args.country or "US",
            )

        elif args.data_type == "imf_gdp":
            return self.data_loader.load_imf_gdp_growth_data(
                countries=[args.country] if args.country else ["US"],
                start_year=args.start_year,
                end_year=args.end_year,
                country_to_use=args.country or "US",
            )

        elif args.data_type == "imf_exchange":
            return self.data_loader.load_imf_exchange_rate_data(
                base_currency=args.base_currency or "USD",
                target_currency=args.target_currency or "EUR",
                start_year=args.start_year,
                end_year=args.end_year,
            )

        elif args.data_type == "csv":
            if not args.file_path:
                raise ValueError("CSV file path required for CSV data type")

            df: pd.DataFrame = pd.read_csv(args.file_path)

            # Try to find a suitable column
            numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
            if args.column:
                if args.column in df.columns:
                    return df[args.column].tolist()
                else:
                    raise ValueError(f"Column '{args.column}' not found in CSV")
            elif len(numeric_cols) > 0:
                return df[numeric_cols[0]].tolist()
            else:
                raise ValueError("No numeric columns found in CSV")

        elif args.data_type == "synthetic":
            # Generate synthetic data

            np.random.seed(42)
            n_samples = args.n_samples or 50

            if args.synthetic_type == "trend":
                trend = np.linspace(0, 10, n_samples)
                noise = np.random.normal(0, 1, n_samples)
                return (trend + noise + 50).tolist()

            elif args.synthetic_type == "seasonal":
                seasonal = 5 * np.sin(2 * np.pi * np.arange(n_samples) / 12)
                noise = np.random.normal(0, 0.5, n_samples)
                return (seasonal + noise + 100).tolist()

            elif args.synthetic_type == "economic":
                # Economic-like data with cycles and shocks
                time_trend = np.linspace(0, 4 * np.pi, n_samples)
                cyclical = 2 * np.sin(time_trend) + np.sin(2 * time_trend)
                shocks = np.random.choice([0, 0, 0, 0, 5], n_samples) * np.random.randn(
                    n_samples
                )
                noise = 0.5 * np.random.randn(n_samples)
                return (cyclical + shocks + noise + 50).tolist()

            else:
                # Default random walk
                return np.cumsum(np.random.randn(n_samples)).tolist()

        else:
            raise ValueError(f"Unknown data type: {args.data_type}")

    def create_forecaster(self, args) -> ComprehensiveForecaster:
        """Create forecaster based on CLI arguments"""
        tokenizer_config = None

        if args.use_advanced_tokenizer:
            tokenizer_config = {
                "window_size": args.window_size,
                "quantization_levels": args.quantization_levels,
                "scaling_method": args.scaling_method,
                "feature_selection": args.feature_selection,
                "max_features": args.max_features,
                "economic_features": args.economic_features,
            }

        return ComprehensiveForecaster(
            model_name=args.model_name,
            device=args.device,
            use_advanced_tokenizer=args.use_advanced_tokenizer,
            tokenizer_config=tokenizer_config,
        )

    def run_forecast(self, args):
        """Run forecasting command"""
        print(f"Loading {args.data_type} data...")
        data = self.load_data(args)
        print(f"Loaded {len(data)} data points")

        # Split data
        train_size = int(len(data) * (1 - args.test_split))
        train_data = data[:train_size]
        test_data = data[train_size:]

        print(f"Train: {len(train_data)}, Test: {len(test_data)}")

        # Create forecaster
        print(f"Initializing {args.model_name} forecaster...")
        forecaster = self.create_forecaster(args)

        # Make forecast
        print("Making forecast...")
        tokenized_data, tokenizer = forecaster.prepare_data(
            train_data, args.window_size
        )
        forecast_result = forecaster.forecast_zero_shot(tokenized_data, len(test_data))

        predictions = forecast_result["mean"][0].numpy()

        # Calculate metrics

        mse = mean_squared_error(test_data, predictions)
        mae = mean_absolute_error(test_data, predictions)
        rmse = np.sqrt(mse)
        mape = (
            np.mean(
                np.abs(
                    (np.array(test_data) - np.array(predictions))
                    / (np.array(test_data) + 1e-8)
                )
            )
            * 100
        )

        # Print results
        print("\\n" + "=" * 50)
        print("FORECAST RESULTS")
        print("=" * 50)
        print(f"MSE:  {mse:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"MAE:  {mae:.6f}")
        print(f"MAPE: {mape:.2f}%")

        # Save results if requested
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)

            # Save predictions
            results = {
                "actual": test_data,
                "predicted": predictions.tolist(),
                "metrics": {"mse": mse, "rmse": rmse, "mae": mae, "mape": mape},
                "config": vars(args),
            }

            with open(os.path.join(args.output_dir, "forecast_results.json"), "w") as f:
                json.dump(results, f, indent=2)

            print(f"\\nResults saved to {args.output_dir}/forecast_results.json")

            # Create visualizations if requested
            if args.create_plots:
                print("Creating visualizations...")

                # Forecast plot
                fig1 = self.visualizer.plot_forecast_results(
                    test_data,
                    predictions.tolist(),
                    train_data,
                    title="Forecast Results",
                    save_path=os.path.join(args.output_dir, "forecast_plot.png"),
                )

                # Residual analysis
                fig2 = self.visualizer.plot_residual_analysis(
                    test_data,
                    predictions.tolist(),
                    title="Residual Analysis",
                    save_path=os.path.join(args.output_dir, "residual_analysis.png"),
                )

                # Create comprehensive report
                report_path = create_forecast_report(
                    test_data,
                    predictions.tolist(),
                    train_data,
                    model_name=args.model_name,
                    save_dir=args.output_dir,
                )

                print(f"Visualizations and report saved to {args.output_dir}")

    def run_comparison(self, args):
        """Run model comparison command"""
        print(f"Loading {args.data_type} data...")
        data = self.load_data(args)
        print(f"Loaded {len(data)} data points")

        # Split data
        train_size = int(len(data) * (1 - args.test_split))
        train_data = data[:train_size]
        test_data = data[train_size:]

        print(f"Train: {len(train_data)}, Test: {len(test_data)}")

        # Test Chronos models
        chronos_results = {}

        if args.include_chronos:
            print("\\nTesting Chronos models...")

            configs = [
                ("Chronos Basic", False, None),
                (
                    "Chronos Advanced",
                    True,
                    {
                        "window_size": args.window_size,
                        "scaling_method": "robust",
                        "feature_selection": True,
                        "max_features": 15,
                        "economic_features": True,
                    },
                ),
            ]

            for name, use_advanced, config in configs:
                try:
                    print(f"  Testing {name}...")
                    forecaster = ComprehensiveForecaster(
                        model_name=args.model_name,
                        device=args.device,
                        use_advanced_tokenizer=use_advanced,
                        tokenizer_config=config,
                    )

                    tokenized_data, tokenizer = forecaster.prepare_data(
                        train_data, args.window_size
                    )
                    forecast_result = forecaster.forecast_zero_shot(
                        tokenized_data, len(test_data)
                    )
                    predictions = forecast_result["mean"][0].numpy()

                    # Calculate metrics
                    mse = mean_squared_error(test_data, predictions)
                    mae = mean_absolute_error(test_data, predictions)

                    chronos_results[name] = {
                        "mse": mse,
                        "mae": mae,
                        "predictions": predictions.tolist(),
                    }

                    print(f"    MSE: {mse:.6f}, MAE: {mae:.6f}")

                except Exception as e:
                    print(f"    Error: {e}")

        # Test baseline models
        baseline_results = {}

        if args.include_baselines:
            print("Testing baseline models...")
            comparison = BaselineComparison()
            comparison.add_standard_models()

            try:
                baseline_results = comparison.compare_models(
                    train_data, test_data, show_progress=False
                )
                comparison.print_comparison()
            except Exception as e:
                print(f"Baseline error: {e}")

        # Print comparison
        print("" + "=" * 80)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 80)

        all_results = {}
        all_results.update(chronos_results)

        # Convert baseline results format
        for model_name, metrics in baseline_results.items():
            if "mse" in metrics and "mae" in metrics:
                all_results[model_name] = {"mse": metrics["mse"], "mae": metrics["mae"]}

        if all_results:
            # Sort by MSE
            sorted_results = sorted(all_results.items(), key=lambda x: x[1]["mse"])

            print(f"{'Model':<25} {'MSE':<12} {'MAE':<12}")
            print("-" * 50)

            for model_name, metrics in sorted_results:
                print(
                    f"{model_name:<25} {metrics['mse']:<12.6f} {metrics['mae']:<12.6f}"
                )

            best_model = sorted_results[0][0]
            print(f"Best Model: {best_model}")

        # Save results if requested
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)

            comparison_results = {
                "chronos_results": chronos_results,
                "baseline_results": baseline_results,
                "config": vars(args),
            }

            with open(
                os.path.join(args.output_dir, "comparison_results.json"), "w"
            ) as f:
                json.dump(comparison_results, f, indent=2)

            print(
                f"Comparison results saved to {args.output_dir}/comparison_results.json"
            )

            # Create comparison plot if requested
            if args.create_plots and all_results:
                print("Creating comparison visualization...")

                fig = self.visualizer.plot_model_comparison(
                    all_results,
                    metrics=["mse", "mae"],
                    title="Model Performance Comparison",
                    save_path=os.path.join(args.output_dir, "model_comparison.png"),
                )

                print(
                    f"Comparison plot saved to {args.output_dir}/model_comparison.png"
                )

    def run_cross_validation(self, args):
        """Run cross-validation command"""
        print(f"Loading {args.data_type} data...")
        data = self.load_data(args)
        print(f"Loaded {len(data)} data points")

        # Create forecaster
        forecaster = self.create_forecaster(args)

        # Initialize backtesting framework
        backtest = BacktestingFramework(
            forecaster=forecaster,
            cv_strategy=args.cv_strategy,
            n_splits=args.n_splits,
            prediction_length=args.prediction_length,
            window_size=args.window_size,
        )

        print(
            f"Running {args.cv_strategy} cross-validation with {args.n_splits} splits..."
        )

        try:
            cv_results = backtest.run_backtest(
                data, show_progress=True, return_predictions=True
            )
            backtest.print_results()

            # Save results if requested
            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)

                # Remove non-serializable objects
                save_results = cv_results.copy()
                if "all_predictions" in save_results:
                    del save_results["all_predictions"]
                if "all_actuals" in save_results:
                    del save_results["all_actuals"]

                with open(os.path.join(args.output_dir, "cv_results.json"), "w") as f:
                    json.dump(save_results, f, indent=2)

                print(f"CV results saved to {args.output_dir}/cv_results.json")

                # Create CV visualization if requested
                if args.create_plots:
                    print("Creating CV visualization...")

                    fig = self.visualizer.plot_cross_validation_results(
                        cv_results,
                        title="Cross-Validation Results",
                        save_path=os.path.join(args.output_dir, "cv_results.png"),
                    )

                    print(f"CV plot saved to {args.output_dir}/cv_results.png")

        except Exception as e:
            print(f"Cross-validation error: {e}")

    def run_demo(self, args):
        """Run demo command"""
        print("Running Chronos Forecasting Framework Demo...")

        # Run comprehensive demo

        try:
            results = run_comprehensive_demo()
            print("Demo completed successfully!")

            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)

                # Save demo results summary
                summary = {
                    "datasets_loaded": 0,
                    "tokenizer_configs_tested": 0,
                    "models_compared": 0,
                    "cv_completed": False,
                }
                if results:
                    summary["datasets_loaded"] = len(results.get("datasets", {}))
                    summary["tokenizer_configs_tested"] = len(
                        results.get("tokenizer_results", {})
                    )
                    forecast_results = results.get("forecast_results")
                    if forecast_results:
                        summary["models_compared"] = len(
                            forecast_results.get("chronos_results", {})
                        ) + len(forecast_results.get("baseline_results", {}))
                    summary["cv_completed"] = results.get("cv_results") is not None

                with open(os.path.join(args.output_dir, "demo_summary.json"), "w") as f:
                    json.dump(summary, f, indent=2)

                print(f"Demo summary saved to {args.output_dir}/demo_summary.json")

        except Exception as e:
            print(f"Demo error: {e}")


def create_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Chronos Forecasting Framework CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Forecast US inflation data
  python cli.py forecast --data-type imf_inflation --country US --start-year 2010 --end-year 2023

  # Compare models on GDP data
  python cli.py compare --data-type imf_gdp --country US --include-chronos --include-baselines

  # Run cross-validation on CSV data
  python cli.py cross-validate --data-type csv --file-path data.csv --column value

  # Run comprehensive demo
  python cli.py demo --output-dir demo_results
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Common arguments
    def add_common_args(parser):
        # Data arguments
        data_group = parser.add_argument_group("Data Options")
        data_group.add_argument(
            "--data-type",
            choices=["imf_inflation", "imf_gdp", "imf_exchange", "csv", "synthetic"],
            default="synthetic",
            help="Type of data to load",
        )
        data_group.add_argument(
            "--country", help="Country code for IMF data (e.g., US, GB, DE)"
        )
        data_group.add_argument(
            "--start-year", type=int, default=2010, help="Start year for IMF data"
        )
        data_group.add_argument(
            "--end-year", type=int, default=2023, help="End year for IMF data"
        )
        data_group.add_argument(
            "--base-currency", default="USD", help="Base currency for exchange rates"
        )
        data_group.add_argument(
            "--target-currency",
            default="EUR",
            help="Target currency for exchange rates",
        )
        data_group.add_argument("--file-path", help="Path to CSV file")
        data_group.add_argument("--column", help="Column name in CSV file")
        data_group.add_argument(
            "--synthetic-type",
            choices=["trend", "seasonal", "economic", "random"],
            default="economic",
            help="Type of synthetic data",
        )
        data_group.add_argument(
            "--n-samples", type=int, default=50, help="Number of synthetic samples"
        )

        # Model arguments
        model_group = parser.add_argument_group("Model Options")
        model_group.add_argument(
            "--model-name",
            default="amazon/chronos-bolt-small",
            help="Chronos model name",
        )
        model_group.add_argument(
            "--device", default="cpu", help="Device to run on (cpu/cuda)"
        )
        model_group.add_argument(
            "--window-size", type=int, default=10, help="Input window size"
        )

        # Tokenizer arguments
        tokenizer_group = parser.add_argument_group("Tokenizer Options")
        tokenizer_group.add_argument(
            "--use-advanced-tokenizer",
            action="store_true",
            help="Use advanced tokenizer",
        )
        tokenizer_group.add_argument(
            "--quantization-levels",
            type=int,
            default=1000,
            help="Number of quantization levels",
        )
        tokenizer_group.add_argument(
            "--scaling-method",
            choices=["standard", "robust", "minmax", "none"],
            default="standard",
            help="Scaling method",
        )
        tokenizer_group.add_argument(
            "--feature-selection", action="store_true", help="Enable feature selection"
        )
        tokenizer_group.add_argument(
            "--max-features", type=int, default=15, help="Maximum number of features"
        )
        tokenizer_group.add_argument(
            "--economic-features",
            action="store_true",
            help="Enable economic-specific features",
        )

        # Output arguments
        output_group = parser.add_argument_group("Output Options")
        output_group.add_argument("--output-dir", help="Directory to save results")
        output_group.add_argument(
            "--create-plots", action="store_true", help="Create visualization plots"
        )

    # Forecast command
    forecast_parser = subparsers.add_parser("forecast", help="Run forecasting")
    add_common_args(forecast_parser)
    forecast_parser.add_argument(
        "--test-split", type=float, default=0.2, help="Test set proportion"
    )

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare multiple models")
    add_common_args(compare_parser)
    compare_parser.add_argument(
        "--test-split", type=float, default=0.2, help="Test set proportion"
    )
    compare_parser.add_argument(
        "--include-chronos", action="store_true", help="Include Chronos models"
    )
    compare_parser.add_argument(
        "--include-baselines", action="store_true", help="Include baseline models"
    )

    # Cross-validation command
    cv_parser = subparsers.add_parser("cross-validate", help="Run cross-validation")
    add_common_args(cv_parser)
    cv_parser.add_argument(
        "--cv-strategy",
        choices=["expanding", "sliding", "blocked"],
        default="expanding",
        help="Cross-validation strategy",
    )
    cv_parser.add_argument(
        "--n-splits", type=int, default=5, help="Number of CV splits"
    )
    cv_parser.add_argument(
        "--prediction-length", type=int, default=5, help="Forecast horizon"
    )

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run comprehensive demo")
    demo_parser.add_argument("--output-dir", help="Directory to save demo results")

    return parser


def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Initialize CLI
    cli = ChronosCLI()

    try:
        if args.command == "forecast":
            cli.run_forecast(args)
        elif args.command == "compare":
            cli.run_comparison(args)
        elif args.command == "cross-validate":
            cli.run_cross_validation(args)
        elif args.command == "demo":
            cli.run_demo(args)
        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()

    except KeyboardInterrupt:
        print("Operation cancelled by user")
    except Exception as e:
        print(f"Error: {e}")
        if "--debug" in sys.argv:
            traceback.print_exc()


if __name__ == "__main__":
    main()
