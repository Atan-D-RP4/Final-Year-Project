"""
Working Demo of Enhanced Chronos Framework
Demonstrates the core functionality that is currently working
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

def demo_imf_data_integration():
    """Demonstrate IMF data integration (with fallback to synthetic data)"""
    print("=" * 60)
    print("DEMO 1: IMF DATA INTEGRATION")
    print("=" * 60)
    
    from enhanced_data_preparation import EnhancedBehavioralDataLoader
    
    loader = EnhancedBehavioralDataLoader()
    
    print("Loading economic data (will use synthetic data as fallback)...")
    
    # Load inflation data
    inflation_data = loader.load_imf_inflation_data(['US'], 2010, 2023, 'US')
    print(f"✓ US Inflation data: {len(inflation_data)} points")
    print(f"  Range: {min(inflation_data):.2f}% to {max(inflation_data):.2f}%")
    
    # Load GDP growth data
    gdp_data = loader.load_imf_gdp_growth_data(['US'], 2010, 2023, 'US')
    print(f"✓ US GDP Growth data: {len(gdp_data)} points")
    print(f"  Range: {min(gdp_data):.2f}% to {max(gdp_data):.2f}%")
    
    # Load exchange rate data
    fx_data = loader.load_imf_exchange_rate_data('USD', 'EUR', 2010, 2023)
    print(f"✓ USD/EUR Exchange Rate: {len(fx_data)} points")
    print(f"  Range: {min(fx_data):.4f} to {max(fx_data):.4f}")
    
    return {
        'inflation': inflation_data,
        'gdp_growth': gdp_data,
        'usd_eur': fx_data
    }

def demo_chronos_forecasting(data):
    """Demonstrate Chronos forecasting"""
    print("\\n" + "=" * 60)
    print("DEMO 2: CHRONOS FORECASTING")
    print("=" * 60)
    
    from chronos_behavioral_framework import ChronosBehavioralForecaster, BenchmarkRunner
    
    # Use inflation data for forecasting
    inflation_data = data['inflation']
    
    print(f"Forecasting on US inflation data ({len(inflation_data)} points)...")
    
    # Split data
    train_size = int(0.8 * len(inflation_data))
    train_data = inflation_data[:train_size]
    test_data = inflation_data[train_size:]
    
    print(f"Train: {len(train_data)} points, Test: {len(test_data)} points")
    
    # Initialize forecaster
    forecaster = ChronosBehavioralForecaster(
        model_name="amazon/chronos-bolt-tiny",  # Use smallest model for demo
        device="cpu"
    )
    
    # Run benchmark
    benchmark = BenchmarkRunner(forecaster)
    results = benchmark.run_benchmark(
        data=inflation_data,
        test_split=0.2,
        prediction_length=len(test_data),
        window_size=8
    )
    
    # Print results
    benchmark.print_results()
    
    return results

def demo_baseline_comparison(data):
    """Demonstrate baseline model comparison"""
    print("\\n" + "=" * 60)
    print("DEMO 3: BASELINE MODEL COMPARISON")
    print("=" * 60)
    
    from baseline_models import NaiveForecaster, DriftForecaster, MovingAverageForecaster
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    # Use GDP growth data
    gdp_data = data['gdp_growth']
    
    print(f"Comparing baseline models on GDP growth data ({len(gdp_data)} points)...")
    
    # Split data
    train_size = int(0.8 * len(gdp_data))
    train_data = gdp_data[:train_size]
    test_data = gdp_data[train_size:]
    
    print(f"Train: {len(train_data)} points, Test: {len(test_data)} points")
    
    # Test baseline models
    models = [
        NaiveForecaster(),
        DriftForecaster(),
        MovingAverageForecaster(window_size=3),
        MovingAverageForecaster(window_size=5)
    ]
    
    results = {}
    
    print("\\nModel Performance:")
    print("-" * 40)
    
    for model in models:
        try:
            predictions = model.fit_predict(train_data, len(test_data))
            
            mse = mean_squared_error(test_data, predictions)
            mae = mean_absolute_error(test_data, predictions)
            
            results[model.name] = {'mse': mse, 'mae': mae}
            
            print(f"{model.name:<20} MSE: {mse:.4f}, MAE: {mae:.4f}")
            
        except Exception as e:
            print(f"{model.name:<20} Error: {e}")
    
    # Find best model
    if results:
        best_model = min(results.items(), key=lambda x: x[1]['mse'])
        print(f"\\nBest Model: {best_model[0]} (MSE: {best_model[1]['mse']:.4f})")
    
    return results

def demo_data_validation():
    """Demonstrate data validation capabilities"""
    print("\\n" + "=" * 60)
    print("DEMO 4: DATA VALIDATION")
    print("=" * 60)
    
    from imf_data_loader import IMFDataLoader
    
    loader = IMFDataLoader()
    
    # Generate different types of synthetic data for validation
    print("Validating different data types...")
    
    # Good quality data
    good_data = loader._generate_synthetic_inflation(24, 'US')
    validation = loader.validate_data(good_data, "inflation")
    print(f"\\nGood Data (US Inflation):")
    print(f"  Quality Score: {validation['quality_score']:.2f}")
    print(f"  Outliers: {len(validation['outliers'])}")
    print(f"  Warnings: {len(validation['warnings'])}")
    
    # Data with outliers
    outlier_data = good_data.copy()
    outlier_data[10] = 50.0  # Add extreme outlier
    validation = loader.validate_data(outlier_data, "inflation")
    print(f"\\nData with Outliers:")
    print(f"  Quality Score: {validation['quality_score']:.2f}")
    print(f"  Outliers: {len(validation['outliers'])}")
    print(f"  Warnings: {validation['warnings']}")
    
    # GDP data validation
    gdp_data = loader._generate_synthetic_gdp_growth(24, 'US')
    validation = loader.validate_data(gdp_data, "gdp_growth")
    print(f"\\nGDP Growth Data:")
    print(f"  Quality Score: {validation['quality_score']:.2f}")
    print(f"  Statistics: Mean={validation['statistics']['mean']:.2f}, Std={validation['statistics']['std']:.2f}")

def demo_feature_engineering():
    """Demonstrate basic feature engineering"""
    print("\\n" + "=" * 60)
    print("DEMO 5: FEATURE ENGINEERING")
    print("=" * 60)
    
    from chronos_behavioral_framework import BehavioralDataTokenizer
    
    # Generate economic-like time series
    np.random.seed(42)
    n_samples = 50
    
    # Create data with trend and seasonality
    trend = np.linspace(0, 5, n_samples)
    seasonal = 2 * np.sin(2 * np.pi * np.arange(n_samples) / 12)
    noise = np.random.normal(0, 0.5, n_samples)
    data = (trend + seasonal + noise + 100).tolist()
    
    print(f"Generated economic time series with {len(data)} points")
    print(f"Data range: [{min(data):.2f}, {max(data):.2f}]")
    
    # Test different tokenizer configurations
    configs = [
        {'window_size': 5, 'quantization_levels': 500},
        {'window_size': 10, 'quantization_levels': 1000},
        {'window_size': 15, 'quantization_levels': 1500}
    ]
    
    print("\\nTesting different tokenizer configurations:")
    print("-" * 50)
    
    for i, config in enumerate(configs):
        tokenizer = BehavioralDataTokenizer(**config)
        tokenized = tokenizer.fit_transform(data)
        
        print(f"Config {i+1}: window={config['window_size']}, levels={config['quantization_levels']}")
        print(f"  Output shape: {tokenized.shape}")
        print(f"  Value range: [{tokenized.min():.2f}, {tokenized.max():.2f}]")

def main():
    """Run the complete working demo"""
    print("ENHANCED CHRONOS FORECASTING FRAMEWORK")
    print("Working Demo - Core Functionality")
    print("=" * 60)
    
    try:
        # Demo 1: Data Integration
        data = demo_imf_data_integration()
        
        # Demo 2: Chronos Forecasting
        forecast_results = demo_chronos_forecasting(data)
        
        # Demo 3: Baseline Comparison
        baseline_results = demo_baseline_comparison(data)
        
        # Demo 4: Data Validation
        demo_data_validation()
        
        # Demo 5: Feature Engineering
        demo_feature_engineering()
        
        # Summary
        print("\\n" + "=" * 60)
        print("DEMO SUMMARY")
        print("=" * 60)
        print("✓ IMF Data Integration (with synthetic fallback)")
        print("✓ Chronos-Bolt Forecasting")
        print("✓ Baseline Model Comparison")
        print("✓ Data Validation and Quality Assessment")
        print("✓ Feature Engineering and Tokenization")
        
        print("\\n🎉 Demo completed successfully!")
        print("\\nThe enhanced Chronos framework provides:")
        print("• Real macroeconomic data integration")
        print("• Advanced preprocessing capabilities")
        print("• Comprehensive model comparison")
        print("• Robust data validation")
        print("• Flexible tokenization strategies")
        
        print("\\nNext steps:")
        print("• Try with your own CSV data")
        print("• Experiment with different model sizes")
        print("• Explore advanced tokenization features")
        print("• Use the CLI interface for batch processing")
        
        return True
        
    except Exception as e:
        print(f"\\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)