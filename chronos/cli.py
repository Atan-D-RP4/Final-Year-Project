"""
Command Line Interface for Chronos Forecasting Framework
Provides easy access to all framework features through command line
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Any, Optional
import warnings

# Import framework modules
from enhanced_data_preparation import EnhancedBehavioralDataLoader
from advanced_tokenizer import AdvancedTokenizer
from cross_validation import BacktestingFramework
from baseline_models import BaselineComparison
from visualization import ForecastVisualizer, create_forecast_report
from chronos_behavioral_framework import ChronosBehavioralForecaster
from comprehensive_demo import ComprehensiveForecaster

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
        if args.data_type == 'imf_inflation':
            return self.data_loader.load_imf_inflation_data(
                countries=[args.country] if args.country else ['US'],
                start_year=args.start_year,
                end_year=args.end_year,
                country_to_use=args.country or 'US'
            )
        
        elif args.data_type == 'imf_gdp':
            return self.data_loader.load_imf_gdp_growth_data(
                countries=[args.country] if args.country else ['US'],
                start_year=args.start_year,
                end_year=args.end_year,
                country_to_use=args.country or 'US'
            )
        
        elif args.data_type == 'imf_exchange':
            return self.data_loader.load_imf_exchange_rate_data(
                base_currency=args.base_currency or 'USD',
                target_currency=args.target_currency or 'EUR',
                start_year=args.start_year,
                end_year=args.end_year
            )
        
        elif args.data_type == 'csv':
            if not args.file_path:
                raise ValueError("CSV file path required for CSV data type")
            
            import pandas as pd
            df = pd.read_csv(args.file_path)
            
            # Try to find a suitable column
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            if args.column:
                if args.column in df.columns:
                    return df[args.column].tolist()
                else:
                    raise ValueError(f"Column '{args.column}' not found in CSV")
            elif len(numeric_cols) > 0:
                return df[numeric_cols[0]].tolist()
            else:
                raise ValueError("No numeric columns found in CSV")
        
        elif args.data_type == 'synthetic':
            # Generate synthetic data
            import numpy as np
            np.random.seed(42)
            n_samples = args.n_samples or 50
            
            if args.synthetic_type == 'trend':
                trend = np.linspace(0, 10, n_samples)
                noise = np.random.normal(0, 1, n_samples)
                return (trend + noise + 50).tolist()
            
            elif args.synthetic_type == 'seasonal':
                seasonal = 5 * np.sin(2 * np.pi * np.arange(n_samples) / 12)
                noise = np.random.normal(0, 0.5, n_samples)
                return (seasonal + noise + 100).tolist()
            
            elif args.synthetic_type == 'economic':
                # Economic-like data with cycles and shocks
                time_trend = np.linspace(0, 4*np.pi, n_samples)
                cyclical = 2 * np.sin(time_trend) + np.sin(2*time_trend)
                shocks = np.random.choice([0, 0, 0, 0, 5], n_samples) * np.random.randn(n_samples)
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
                'window_size': args.window_size,
                'quantization_levels': args.quantization_levels,
                'scaling_method': args.scaling_method,
                'feature_selection': args.feature_selection,
                'max_features': args.max_features,
                'economic_features': args.economic_features
            }
        
        return ComprehensiveForecaster(
            model_name=args.model_name,
            device=args.device,
            use_advanced_tokenizer=args.use_advanced_tokenizer,
            tokenizer_config=tokenizer_config
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
        tokenized_data, tokenizer = forecaster.prepare_data(train_data, args.window_size)
        forecast_result = forecaster.forecast_zero_shot(tokenized_data, len(test_data))
        
        predictions = forecast_result['mean'][0].numpy()
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        import numpy as np
        
        mse = mean_squared_error(test_data, predictions)
        mae = mean_absolute_error(test_data, predictions)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((np.array(test_data) - np.array(predictions)) / (np.array(test_data) + 1e-8))) * 100
        
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
                'actual': test_data,
                'predicted': predictions.tolist(),
                'metrics': {
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'mape': mape
                },
                'config': vars(args)
            }
            
            with open(os.path.join(args.output_dir, 'forecast_results.json'), 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\\nResults saved to {args.output_dir}/forecast_results.json")
            
            # Create visualizations if requested
            if args.create_plots:
                print("Creating visualizations...")
                
                # Forecast plot
                fig1 = self.visualizer.plot_forecast_results(
                    test_data, predictions.tolist(), train_data,
                    title="Forecast Results",
                    save_path=os.path.join(args.output_dir, "forecast_plot.png")
                )
                
                # Residual analysis
                fig2 = self.visualizer.plot_residual_analysis(
                    test_data, predictions.tolist(),
                    title="Residual Analysis",
                    save_path=os.path.join(args.output_dir, "residual_analysis.png")
                )
                
                # Create comprehensive report
                report_path = create_forecast_report(
                    test_data, predictions.tolist(), train_data,
                    model_name=args.model_name,
                    save_dir=args.output_dir
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
                ('Chronos Basic', False, None),
                ('Chronos Advanced', True, {
                    'window_size': args.window_size,
                    'scaling_method': 'robust',
                    'feature_selection': True,
                    'max_features': 15,
                    'economic_features': True
                })
            ]
            
            for name, use_advanced, config in configs:
                try:\n                    print(f\"  Testing {name}...\")\n                    forecaster = ComprehensiveForecaster(\n                        model_name=args.model_name,\n                        device=args.device,\n                        use_advanced_tokenizer=use_advanced,\n                        tokenizer_config=config\n                    )\n                    \n                    tokenized_data, tokenizer = forecaster.prepare_data(train_data, args.window_size)\n                    forecast_result = forecaster.forecast_zero_shot(tokenized_data, len(test_data))\n                    predictions = forecast_result['mean'][0].numpy()\n                    \n                    # Calculate metrics\n                    from sklearn.metrics import mean_squared_error, mean_absolute_error\n                    import numpy as np\n                    \n                    mse = mean_squared_error(test_data, predictions)\n                    mae = mean_absolute_error(test_data, predictions)\n                    \n                    chronos_results[name] = {\n                        'mse': mse,\n                        'mae': mae,\n                        'predictions': predictions.tolist()\n                    }\n                    \n                    print(f\"    MSE: {mse:.6f}, MAE: {mae:.6f}\")\n                    \n                except Exception as e:\n                    print(f\"    Error: {e}\")\n        \n        # Test baseline models\n        baseline_results = {}\n        \n        if args.include_baselines:\n            print(\"\\nTesting baseline models...\")\n            comparison = BaselineComparison()\n            comparison.add_standard_models()\n            \n            try:\n                baseline_results = comparison.compare_models(train_data, test_data, show_progress=False)\n                comparison.print_comparison()\n            except Exception as e:\n                print(f\"Baseline error: {e}\")\n        \n        # Print comparison\n        print(\"\\n\" + \"=\" * 80)\n        print(\"MODEL COMPARISON SUMMARY\")\n        print(\"=\" * 80)\n        \n        all_results = {}\n        all_results.update(chronos_results)\n        \n        # Convert baseline results format\n        for model_name, metrics in baseline_results.items():\n            if 'mse' in metrics and 'mae' in metrics:\n                all_results[model_name] = {\n                    'mse': metrics['mse'],\n                    'mae': metrics['mae']\n                }\n        \n        if all_results:\n            # Sort by MSE\n            sorted_results = sorted(all_results.items(), key=lambda x: x[1]['mse'])\n            \n            print(f\"{'Model':<25} {'MSE':<12} {'MAE':<12}\")\n            print(\"-\" * 50)\n            \n            for model_name, metrics in sorted_results:\n                print(f\"{model_name:<25} {metrics['mse']:<12.6f} {metrics['mae']:<12.6f}\")\n            \n            best_model = sorted_results[0][0]\n            print(f\"\\nBest Model: {best_model}\")\n        \n        # Save results if requested\n        if args.output_dir:\n            os.makedirs(args.output_dir, exist_ok=True)\n            \n            comparison_results = {\n                'chronos_results': chronos_results,\n                'baseline_results': baseline_results,\n                'config': vars(args)\n            }\n            \n            with open(os.path.join(args.output_dir, 'comparison_results.json'), 'w') as f:\n                json.dump(comparison_results, f, indent=2)\n            \n            print(f\"\\nComparison results saved to {args.output_dir}/comparison_results.json\")\n            \n            # Create comparison plot if requested\n            if args.create_plots and all_results:\n                print(\"Creating comparison visualization...\")\n                \n                fig = self.visualizer.plot_model_comparison(\n                    all_results,\n                    metrics=['mse', 'mae'],\n                    title=\"Model Performance Comparison\",\n                    save_path=os.path.join(args.output_dir, \"model_comparison.png\")\n                )\n                \n                print(f\"Comparison plot saved to {args.output_dir}/model_comparison.png\")\n    \n    def run_cross_validation(self, args):\n        \"\"\"Run cross-validation command\"\"\"\n        print(f\"Loading {args.data_type} data...\")\n        data = self.load_data(args)\n        print(f\"Loaded {len(data)} data points\")\n        \n        # Create forecaster\n        forecaster = self.create_forecaster(args)\n        \n        # Initialize backtesting framework\n        backtest = BacktestingFramework(\n            forecaster=forecaster,\n            cv_strategy=args.cv_strategy,\n            n_splits=args.n_splits,\n            prediction_length=args.prediction_length,\n            window_size=args.window_size\n        )\n        \n        print(f\"\\nRunning {args.cv_strategy} cross-validation with {args.n_splits} splits...\")\n        \n        try:\n            cv_results = backtest.run_backtest(data, show_progress=True, return_predictions=True)\n            backtest.print_results()\n            \n            # Save results if requested\n            if args.output_dir:\n                os.makedirs(args.output_dir, exist_ok=True)\n                \n                # Remove non-serializable objects\n                save_results = cv_results.copy()\n                if 'all_predictions' in save_results:\n                    del save_results['all_predictions']\n                if 'all_actuals' in save_results:\n                    del save_results['all_actuals']\n                \n                with open(os.path.join(args.output_dir, 'cv_results.json'), 'w') as f:\n                    json.dump(save_results, f, indent=2)\n                \n                print(f\"\\nCV results saved to {args.output_dir}/cv_results.json\")\n                \n                # Create CV visualization if requested\n                if args.create_plots:\n                    print(\"Creating CV visualization...\")\n                    \n                    fig = self.visualizer.plot_cross_validation_results(\n                        cv_results,\n                        title=\"Cross-Validation Results\",\n                        save_path=os.path.join(args.output_dir, \"cv_results.png\")\n                    )\n                    \n                    print(f\"CV plot saved to {args.output_dir}/cv_results.png\")\n            \n        except Exception as e:\n            print(f\"Cross-validation error: {e}\")\n    \n    def run_demo(self, args):\n        \"\"\"Run demo command\"\"\"\n        print(\"Running Chronos Forecasting Framework Demo...\")\n        \n        # Import and run comprehensive demo\n        from comprehensive_demo import main as run_comprehensive_demo\n        \n        try:\n            results = run_comprehensive_demo()\n            print(\"\\nDemo completed successfully!\")\n            \n            if args.output_dir:\n                os.makedirs(args.output_dir, exist_ok=True)\n                \n                # Save demo results summary\n                summary = {\n                    'datasets_loaded': len(results.get('datasets', {})),\n                    'tokenizer_configs_tested': len(results.get('tokenizer_results', {})),\n                    'models_compared': len(results.get('forecast_results', {}).get('chronos_results', {})) + \n                                     len(results.get('forecast_results', {}).get('baseline_results', {})),\n                    'cv_completed': results.get('cv_results') is not None\n                }\n                \n                with open(os.path.join(args.output_dir, 'demo_summary.json'), 'w') as f:\n                    json.dump(summary, f, indent=2)\n                \n                print(f\"Demo summary saved to {args.output_dir}/demo_summary.json\")\n            \n        except Exception as e:\n            print(f\"Demo error: {e}\")\n\n\ndef create_parser():\n    \"\"\"Create command line argument parser\"\"\"\n    parser = argparse.ArgumentParser(\n        description=\"Chronos Forecasting Framework CLI\",\n        formatter_class=argparse.RawDescriptionHelpFormatter,\n        epilog=\"\"\"\nExamples:\n  # Forecast US inflation data\n  python cli.py forecast --data-type imf_inflation --country US --start-year 2010 --end-year 2023\n  \n  # Compare models on GDP data\n  python cli.py compare --data-type imf_gdp --country US --include-chronos --include-baselines\n  \n  # Run cross-validation on CSV data\n  python cli.py cross-validate --data-type csv --file-path data.csv --column value\n  \n  # Run comprehensive demo\n  python cli.py demo --output-dir demo_results\n        \"\"\"\n    )\n    \n    subparsers = parser.add_subparsers(dest='command', help='Available commands')\n    \n    # Common arguments\n    def add_common_args(parser):\n        # Data arguments\n        data_group = parser.add_argument_group('Data Options')\n        data_group.add_argument('--data-type', choices=['imf_inflation', 'imf_gdp', 'imf_exchange', 'csv', 'synthetic'],\n                               default='synthetic', help='Type of data to load')\n        data_group.add_argument('--country', help='Country code for IMF data (e.g., US, GB, DE)')\n        data_group.add_argument('--start-year', type=int, default=2010, help='Start year for IMF data')\n        data_group.add_argument('--end-year', type=int, default=2023, help='End year for IMF data')\n        data_group.add_argument('--base-currency', default='USD', help='Base currency for exchange rates')\n        data_group.add_argument('--target-currency', default='EUR', help='Target currency for exchange rates')\n        data_group.add_argument('--file-path', help='Path to CSV file')\n        data_group.add_argument('--column', help='Column name in CSV file')\n        data_group.add_argument('--synthetic-type', choices=['trend', 'seasonal', 'economic', 'random'],\n                               default='economic', help='Type of synthetic data')\n        data_group.add_argument('--n-samples', type=int, default=50, help='Number of synthetic samples')\n        \n        # Model arguments\n        model_group = parser.add_argument_group('Model Options')\n        model_group.add_argument('--model-name', default='amazon/chronos-bolt-small',\n                                help='Chronos model name')\n        model_group.add_argument('--device', default='cpu', help='Device to run on (cpu/cuda)')\n        model_group.add_argument('--window-size', type=int, default=10, help='Input window size')\n        \n        # Tokenizer arguments\n        tokenizer_group = parser.add_argument_group('Tokenizer Options')\n        tokenizer_group.add_argument('--use-advanced-tokenizer', action='store_true',\n                                    help='Use advanced tokenizer')\n        tokenizer_group.add_argument('--quantization-levels', type=int, default=1000,\n                                    help='Number of quantization levels')\n        tokenizer_group.add_argument('--scaling-method', choices=['standard', 'robust', 'minmax', 'none'],\n                                    default='standard', help='Scaling method')\n        tokenizer_group.add_argument('--feature-selection', action='store_true',\n                                    help='Enable feature selection')\n        tokenizer_group.add_argument('--max-features', type=int, default=15,\n                                    help='Maximum number of features')\n        tokenizer_group.add_argument('--economic-features', action='store_true',\n                                    help='Enable economic-specific features')\n        \n        # Output arguments\n        output_group = parser.add_argument_group('Output Options')\n        output_group.add_argument('--output-dir', help='Directory to save results')\n        output_group.add_argument('--create-plots', action='store_true', help='Create visualization plots')\n    \n    # Forecast command\n    forecast_parser = subparsers.add_parser('forecast', help='Run forecasting')\n    add_common_args(forecast_parser)\n    forecast_parser.add_argument('--test-split', type=float, default=0.2, help='Test set proportion')\n    \n    # Compare command\n    compare_parser = subparsers.add_parser('compare', help='Compare multiple models')\n    add_common_args(compare_parser)\n    compare_parser.add_argument('--test-split', type=float, default=0.2, help='Test set proportion')\n    compare_parser.add_argument('--include-chronos', action='store_true', help='Include Chronos models')\n    compare_parser.add_argument('--include-baselines', action='store_true', help='Include baseline models')\n    \n    # Cross-validation command\n    cv_parser = subparsers.add_parser('cross-validate', help='Run cross-validation')\n    add_common_args(cv_parser)\n    cv_parser.add_argument('--cv-strategy', choices=['expanding', 'sliding', 'blocked'],\n                          default='expanding', help='Cross-validation strategy')\n    cv_parser.add_argument('--n-splits', type=int, default=5, help='Number of CV splits')\n    cv_parser.add_argument('--prediction-length', type=int, default=5, help='Forecast horizon')\n    \n    # Demo command\n    demo_parser = subparsers.add_parser('demo', help='Run comprehensive demo')\n    demo_parser.add_argument('--output-dir', help='Directory to save demo results')\n    \n    return parser\n\n\ndef main():\n    \"\"\"Main CLI entry point\"\"\"\n    parser = create_parser()\n    args = parser.parse_args()\n    \n    if not args.command:\n        parser.print_help()\n        return\n    \n    # Initialize CLI\n    cli = ChronosCLI()\n    \n    try:\n        if args.command == 'forecast':\n            cli.run_forecast(args)\n        elif args.command == 'compare':\n            cli.run_comparison(args)\n        elif args.command == 'cross-validate':\n            cli.run_cross_validation(args)\n        elif args.command == 'demo':\n            cli.run_demo(args)\n        else:\n            print(f\"Unknown command: {args.command}\")\n            parser.print_help()\n    \n    except KeyboardInterrupt:\n        print(\"\\nOperation cancelled by user\")\n    except Exception as e:\n        print(f\"Error: {e}\")\n        if '--debug' in sys.argv:\n            import traceback\n            traceback.print_exc()\n\n\nif __name__ == \"__main__\":\n    main()