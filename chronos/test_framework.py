"""
Simple test script to verify the enhanced framework functionality
"""

import numpy as np
import warnings

warnings.filterwarnings("ignore")


def test_imf_loader():
    """Test IMF data loader"""
    print("Testing IMF Data Loader...")
    try:
        from imf_data_loader import IMFDataLoader

        loader = IMFDataLoader()

        # Test synthetic data generation (fallback when API fails)
        inflation_data = loader._generate_synthetic_inflation(24)
        gdp_data = loader._generate_synthetic_gdp_growth(24)

        print(f"✓ Generated {len(inflation_data)} inflation data points")
        print(f"✓ Generated {len(gdp_data)} GDP growth data points")

        # Test data validation
        validation = loader.validate_data(inflation_data, "inflation")
        print(f"✓ Data validation: quality score = {validation['quality_score']:.2f}")

        return True
    except Exception as e:
        print(f"✗ IMF Loader error: {e}")
        return False


def test_enhanced_data_loader():
    """Test enhanced data preparation"""
    print("\nTesting Enhanced Data Loader...")
    try:
        from enhanced_data_preparation import EnhancedBehavioralDataLoader

        loader = EnhancedBehavioralDataLoader()

        # Test IMF data loading (will use synthetic data)
        inflation_data = loader.load_imf_inflation_data(["US"], 2010, 2020, "US")
        gdp_data = loader.load_imf_gdp_growth_data(["US"], 2010, 2020, "US")

        print(f"✓ Loaded {len(inflation_data)} inflation data points")
        print(f"✓ Loaded {len(gdp_data)} GDP growth data points")

        # Test multivariate data
        mv_data = loader.load_imf_multivariate_data(
            "US", ["gdp_growth", "inflation"], 2010, 2020
        )
        print(f"✓ Loaded multivariate data: {list(mv_data.keys())}")

        return True
    except Exception as e:
        print(f"✗ Enhanced Data Loader error: {e}")
        return False


def test_basic_tokenizer():
    """Test basic tokenizer functionality"""
    print("\nTesting Basic Tokenizer...")
    try:
        from chronos_behavioral_framework import BehavioralDataTokenizer

        # Generate test data
        np.random.seed(42)
        data = np.cumsum(np.random.randn(50)).tolist()

        tokenizer = BehavioralDataTokenizer(window_size=10)
        tokenized = tokenizer.fit_transform(data)

        print(f"✓ Tokenized data shape: {tokenized.shape}")
        print(f"✓ Input length: {len(data)}, Output length: {len(tokenized)}")

        return True
    except Exception as e:
        print(f"✗ Basic Tokenizer error: {e}")
        return False


def test_chronos_forecaster():
    """Test Chronos forecaster"""
    print("\nTesting Chronos Forecaster...")
    try:
        from chronos_behavioral_framework import ChronosBehavioralForecaster

        # Generate test data
        np.random.seed(42)
        data = (np.cumsum(np.random.randn(50)) + 100).tolist()

        forecaster = ChronosBehavioralForecaster(
            model_name="amazon/chronos-bolt-tiny",  # Use smallest model for testing
            device="cpu",
        )

        # Prepare data
        tokenized_data, tokenizer = forecaster.prepare_data(data, window_size=10)
        print(f"✓ Prepared data shape: {tokenized_data.shape}")

        # Make forecast
        forecast_result = forecaster.forecast_zero_shot(tokenized_data, 5)
        predictions = forecast_result["mean"][0].numpy()

        print(f"✓ Generated {len(predictions)} predictions")
        print(f"✓ Prediction range: [{predictions.min():.2f}, {predictions.max():.2f}]")

        return True
    except Exception as e:
        print(f"✗ Chronos Forecaster error: {e}")
        return False


def test_baseline_models():
    """Test baseline models"""
    print("\nTesting Baseline Models...")
    try:
        from baseline_models import (
            NaiveForecaster,
            DriftForecaster,
            MovingAverageForecaster,
        )

        # Generate test data
        np.random.seed(42)
        data = (np.cumsum(np.random.randn(30)) + 100).tolist()
        train_data = data[:20]
        test_length = 5

        # Test different baseline models
        models = [
            NaiveForecaster(),
            DriftForecaster(),
            MovingAverageForecaster(window_size=5),
        ]

        for model in models:
            predictions = model.fit_predict(train_data, test_length)
            print(f"✓ {model.name}: {len(predictions)} predictions")

        return True
    except Exception as e:
        print(f"✗ Baseline Models error: {e}")
        return False


def test_visualization():
    """Test visualization capabilities"""
    print("\nTesting Visualization...")
    try:
        from visualization import ForecastVisualizer

        # Generate test data
        np.random.seed(42)
        actual = np.random.randn(10) + 100
        predicted = actual + np.random.normal(0, 0.5, 10)

        visualizer = ForecastVisualizer()

        # Test time series plot (don't show, just create)
        import matplotlib

        matplotlib.use("Agg")  # Use non-interactive backend

        fig = visualizer.plot_time_series(actual.tolist(), title="Test Time Series")
        print("✓ Created time series plot")

        # Test forecast plot
        fig = visualizer.plot_forecast_results(
            actual.tolist(), predicted.tolist(), title="Test Forecast"
        )
        print("✓ Created forecast plot")

        return True
    except Exception as e:
        print(f"✗ Visualization error: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("ENHANCED CHRONOS FRAMEWORK - BASIC FUNCTIONALITY TEST")
    print("=" * 60)

    tests = [
        test_imf_loader,
        test_enhanced_data_loader,
        test_basic_tokenizer,
        test_chronos_forecaster,
        test_baseline_models,
        test_visualization,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append(False)

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed / total * 100:.1f}%")

    if passed == total:
        print("\n🎉 All tests passed! The framework is working correctly.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the error messages above.")

    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
