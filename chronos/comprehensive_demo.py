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
    
    def __init__(self, 
                 model_name: str = "amazon/chronos-bolt-small",
                 device: str = "cpu",
                 use_advanced_tokenizer: bool = True,
                 tokenizer_config: Dict = None):
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
                'window_size': 10,
                'quantization_levels': 1000,
                'scaling_method': 'standard',
                'feature_selection': True,
                'max_features': 15,
                'economic_features': True
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
        inflation_data = loader.load_imf_inflation_data(['US'], 2010, 2023, 'US')
        datasets['US_Inflation'] = inflation_data
        print(f"   ✓ US Inflation: {len(inflation_data)} points, range: [{min(inflation_data):.2f}%, {max(inflation_data):.2f}%]")
    except Exception as e:
        print(f"   ✗ US Inflation: {e}")
    
    # US GDP Growth data
    try:
        gdp_data = loader.load_imf_gdp_growth_data(['US'], 2010, 2023, 'US')
        datasets['US_GDP_Growth'] = gdp_data
        print(f"   ✓ US GDP Growth: {len(gdp_data)} points, range: [{min(gdp_data):.2f}%, {max(gdp_data):.2f}%]")
    except Exception as e:
        print(f"   ✗ US GDP Growth: {e}")
    
    # Exchange rate data
    try:
        fx_data = loader.load_imf_exchange_rate_data('USD', 'EUR', 2010, 2023)
        datasets['USD_EUR'] = fx_data
        print(f"   ✓ USD/EUR: {len(fx_data)} points, range: [{min(fx_data):.4f}, {max(fx_data):.4f}]")
    except Exception as e:
        print(f"   ✗ USD/EUR: {e}")
    
    # Multivariate data
    try:
        mv_data = loader.load_imf_multivariate_data('US', ['gdp_growth', 'inflation'], 2010, 2023)
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
            'name': 'Basic',
            'config': {
                'window_size': 8,
                'scaling_method': 'standard',
                'feature_selection': False,
                'economic_features': False
            }
        },
        {
            'name': 'Advanced',
            'config': {
                'window_size': 10,
                'scaling_method': 'robust',
                'feature_selection': True,
                'max_features': 15,
                'economic_features': True
            }
        },
        {
            'name': 'Economic Focus',
            'config': {
                'window_size': 12,
                'scaling_method': 'standard',
                'feature_selection': True,
                'max_features': 20,
                'economic_features': True
            }
        }
    ]
    
    tokenizer_results = {}
    
    for config in configs:
        print(f"\\nTesting {config['name']} tokenizer...")
        try:
            tokenizer = AdvancedTokenizer(**config['config'])
            tokenized = tokenizer.fit_transform(data)
            
            feature_names = tokenizer.get_feature_names()
            importance = tokenizer.get_feature_importance()
            
            tokenizer_results[config['name']] = {
                'tokenized_shape': tokenized.shape,
                'n_features': len(feature_names),
                'top_features': list(importance.keys())[:5] if importance else []
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
            'name': 'Chronos Basic',
            'use_advanced_tokenizer': False,
            'tokenizer_config': None
        },
        {
            'name': 'Chronos Advanced',
            'use_advanced_tokenizer': True,
            'tokenizer_config': {
                'window_size': 10,
                'scaling_method': 'robust',
                'feature_selection': True,
                'max_features': 15,
                'economic_features': True
            }
        }
    ]
    
    chronos_results = {}\n    \n    for config in chronos_configs:\n        print(f\"\\nTesting {config['name']}...\")\n        try:\n            forecaster = ComprehensiveForecaster(\n                model_name=\"amazon/chronos-bolt-small\",\n                device=\"cpu\",\n                use_advanced_tokenizer=config['use_advanced_tokenizer'],\n                tokenizer_config=config['tokenizer_config']\n            )\n            \n            # Prepare data and forecast\n            tokenized_data, tokenizer = forecaster.prepare_data(train_data, 10)\n            forecast_result = forecaster.forecast_zero_shot(tokenized_data, len(test_data))\n            \n            predictions = forecast_result['mean'][0].numpy()\n            \n            # Calculate metrics\n            from sklearn.metrics import mean_squared_error, mean_absolute_error\n            mse = mean_squared_error(test_data, predictions)\n            mae = mean_absolute_error(test_data, predictions)\n            \n            chronos_results[config['name']] = {\n                'mse': mse,\n                'mae': mae,\n                'predictions': predictions.tolist()\n            }\n            \n            print(f\"   ✓ MSE: {mse:.6f}, MAE: {mae:.6f}\")\n            \n        except Exception as e:\n            print(f\"   ✗ Error: {e}\")\n    \n    # Compare with baseline models\n    print(\"\\nTesting baseline models...\")\n    baseline_comparison = BaselineComparison()\n    baseline_comparison.add_standard_models()\n    \n    try:\n        baseline_results = baseline_comparison.compare_models(train_data, test_data, show_progress=False)\n        print(\"   ✓ Baseline models completed\")\n    except Exception as e:\n        print(f\"   ✗ Baseline error: {e}\")\n        baseline_results = {}\n    \n    return {\n        'chronos_results': chronos_results,\n        'baseline_results': baseline_results,\n        'test_data': test_data,\n        'train_data': train_data\n    }\n\n\ndef run_cross_validation_demo(datasets: Dict[str, List[float]]):\n    \"\"\"\n    Demonstrate cross-validation framework\n    \"\"\"\n    print(\"\\n\" + \"=\" * 80)\n    print(\"PHASE 4: CROSS-VALIDATION DEMO\")\n    print(\"=\" * 80)\n    \n    if not datasets:\n        print(\"No datasets available for cross-validation demo\")\n        return None\n    \n    # Select dataset\n    dataset_name = list(datasets.keys())[0]\n    data = datasets[dataset_name]\n    \n    print(f\"\\nCross-validating on: {dataset_name} ({len(data)} points)\")\n    \n    # Initialize forecaster\n    forecaster = ComprehensiveForecaster(\n        use_advanced_tokenizer=True,\n        tokenizer_config={\n            'window_size': 8,\n            'scaling_method': 'robust',\n            'feature_selection': True,\n            'max_features': 12,\n            'economic_features': True\n        }\n    )\n    \n    # Initialize backtesting framework\n    backtest = BacktestingFramework(\n        forecaster=forecaster,\n        cv_strategy='expanding',\n        n_splits=5,\n        prediction_length=3,\n        window_size=8\n    )\n    \n    print(\"\\nRunning cross-validation...\")\n    try:\n        cv_results = backtest.run_backtest(data, show_progress=True, return_predictions=True)\n        backtest.print_results()\n        \n        # Compare CV strategies\n        print(\"\\nComparing CV strategies...\")\n        strategy_comparison = backtest.compare_strategies(data, ['expanding', 'sliding'])\n        \n        return cv_results\n        \n    except Exception as e:\n        print(f\"Cross-validation error: {e}\")\n        return None\n\n\ndef run_visualization_demo(forecast_results: Dict[str, Any]):\n    \"\"\"\n    Demonstrate visualization capabilities\n    \"\"\"\n    print(\"\\n\" + \"=\" * 80)\n    print(\"PHASE 5: VISUALIZATION DEMO\")\n    print(\"=\" * 80)\n    \n    if not forecast_results:\n        print(\"No forecast results available for visualization\")\n        return\n    \n    # Initialize visualizer\n    visualizer = ForecastVisualizer()\n    \n    # Extract data\n    chronos_results = forecast_results.get('chronos_results', {})\n    baseline_results = forecast_results.get('baseline_results', {})\n    test_data = forecast_results.get('test_data', [])\n    train_data = forecast_results.get('train_data', [])\n    \n    if not test_data:\n        print(\"No test data available for visualization\")\n        return\n    \n    print(\"\\n1. Creating forecast visualization...\")\n    \n    # Get best Chronos result\n    if chronos_results:\n        best_chronos = min(chronos_results.items(), key=lambda x: x[1]['mse'])\n        best_chronos_name, best_chronos_data = best_chronos\n        \n        # Plot forecast results\n        fig1 = visualizer.plot_forecast_results(\n            test_data,\n            best_chronos_data['predictions'],\n            train_data,\n            title=f\"{best_chronos_name} Forecast Results\"\n        )\n        plt.show()\n        \n        # Plot residual analysis\n        print(\"\\n2. Creating residual analysis...\")\n        fig2 = visualizer.plot_residual_analysis(\n            test_data,\n            best_chronos_data['predictions'],\n            title=f\"{best_chronos_name} Residual Analysis\"\n        )\n        plt.show()\n    \n    # Plot model comparison\n    if chronos_results or baseline_results:\n        print(\"\\n3. Creating model comparison...\")\n        \n        # Combine results\n        all_results = {}\n        all_results.update(chronos_results)\n        \n        # Add baseline results (convert format)\n        for model_name, metrics in baseline_results.items():\n            if 'mse' in metrics and 'mae' in metrics:\n                all_results[model_name] = {\n                    'mse': metrics['mse'],\n                    'mae': metrics['mae']\n                }\n        \n        if all_results:\n            fig3 = visualizer.plot_model_comparison(\n                all_results,\n                metrics=['mse', 'mae'],\n                title=\"Model Performance Comparison\"\n            )\n            plt.show()\n    \n    # Create comprehensive report\n    if chronos_results:\n        print(\"\\n4. Creating comprehensive report...\")\n        best_chronos_name, best_chronos_data = min(chronos_results.items(), key=lambda x: x[1]['mse'])\n        \n        report_path = create_forecast_report(\n            test_data,\n            best_chronos_data['predictions'],\n            train_data,\n            model_name=best_chronos_name,\n            save_dir=\"comprehensive_demo_report\"\n        )\n        \n        print(f\"   ✓ Report saved to: {report_path}\")\n\n\ndef main():\n    \"\"\"\n    Run comprehensive demo of all framework features\n    \"\"\"\n    print(\"COMPREHENSIVE CHRONOS FORECASTING FRAMEWORK DEMO\")\n    print(\"=\" * 80)\n    print(\"This demo showcases all enhanced features:\")\n    print(\"• Phase 1: IMF Data Integration\")\n    print(\"• Phase 2: Advanced Preprocessing\")\n    print(\"• Phase 3: Forecasting Comparison\")\n    print(\"• Phase 4: Cross-Validation\")\n    print(\"• Phase 5: Visualization\")\n    print(\"=\" * 80)\n    \n    # Phase 1: IMF Data Integration\n    datasets = run_imf_data_demo()\n    \n    if not datasets:\n        print(\"\\nNo datasets loaded. Generating synthetic data for demo...\")\n        # Generate synthetic economic data\n        np.random.seed(42)\n        n_samples = 30\n        \n        # Synthetic inflation data\n        inflation = 2.5 + 1.5 * np.sin(np.linspace(0, 4*np.pi, n_samples)) + 0.5 * np.random.randn(n_samples)\n        inflation = np.maximum(inflation, 0.5)  # Ensure positive\n        \n        # Synthetic GDP growth data\n        gdp_growth = 2.0 + 1.0 * np.sin(np.linspace(0, 2*np.pi, n_samples)) + 0.8 * np.random.randn(n_samples)\n        \n        datasets = {\n            'Synthetic_Inflation': inflation.tolist(),\n            'Synthetic_GDP_Growth': gdp_growth.tolist()\n        }\n        \n        print(f\"Generated {len(datasets)} synthetic datasets\")\n    \n    # Phase 2: Advanced Preprocessing\n    tokenizer_results = run_advanced_preprocessing_demo(datasets)\n    \n    # Phase 3: Forecasting Comparison\n    forecast_results = run_forecasting_comparison_demo(datasets)\n    \n    # Phase 4: Cross-Validation\n    cv_results = run_cross_validation_demo(datasets)\n    \n    # Phase 5: Visualization\n    if forecast_results:\n        run_visualization_demo(forecast_results)\n    \n    # Summary\n    print(\"\\n\" + \"=\" * 80)\n    print(\"DEMO SUMMARY\")\n    print(\"=\" * 80)\n    print(f\"✓ Datasets processed: {len(datasets)}\")\n    print(f\"✓ Tokenizer configurations tested: {len(tokenizer_results) if tokenizer_results else 0}\")\n    print(f\"✓ Forecasting models compared: {len(forecast_results['chronos_results']) + len(forecast_results['baseline_results']) if forecast_results else 0}\")\n    print(f\"✓ Cross-validation completed: {'Yes' if cv_results else 'No'}\")\n    print(f\"✓ Visualizations created: {'Yes' if forecast_results else 'No'}\")\n    \n    print(\"\\n🎉 Comprehensive demo completed successfully!\")\n    print(\"\\nNext steps:\")\n    print(\"• Explore the generated reports in 'comprehensive_demo_report/'\")\n    print(\"• Modify configurations in the demo functions\")\n    print(\"• Try with your own datasets\")\n    print(\"• Experiment with different model sizes (chronos-bolt-base, chronos-bolt-large)\")\n    \n    return {\n        'datasets': datasets,\n        'tokenizer_results': tokenizer_results,\n        'forecast_results': forecast_results,\n        'cv_results': cv_results\n    }\n\n\nif __name__ == \"__main__\":\n    # Import matplotlib for visualization\n    import matplotlib.pyplot as plt\n    \n    # Run comprehensive demo\n    results = main()