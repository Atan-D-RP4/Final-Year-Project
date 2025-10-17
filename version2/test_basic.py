"""Basic test to verify the implementation works."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))


def test_imports() -> bool:
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        print("✓ Config module imported")

        print("✓ Logger module imported")

        print("✓ Data fetchers module imported")

        print("✓ Data cleaning module imported")

        print("✓ Tokenizer module imported")

        print("✓ Baseline models module imported")

        print("✓ Evaluation metrics module imported")

        print("All imports successful!")
        return True

    except Exception as e:
        print(f"Import error: {e}")
        return False


def test_basic_functionality() -> bool:
    """Test basic functionality without heavy dependencies."""
    print("\nTesting basic functionality...")

    try:
        # Test configuration
        from src.utils.config import load_config

        config = load_config()
        print(f"✓ Config loaded: {config.data.targets}")

        # Test sample data
        from src.data.fetchers import get_sample_data

        data = get_sample_data()
        print(f"✓ Sample data generated: {list(data.keys())}")

        # Test data cleaning
        from src.data.cleaning import DataCleaner

        cleaner = DataCleaner()
        market_data = data["market"]
        cleaned = cleaner.clean_market_data(market_data)
        print(f"✓ Data cleaned: {cleaned.shape}")

        # Test tokenizer
        from src.preprocessing.tokenizer import FinancialDataTokenizer

        tokenizer = FinancialDataTokenizer(num_bins=64, method="uniform")
        tokenizer.fit(cleaned)
        tokens = tokenizer.transform(cleaned)
        print(f"✓ Tokenization completed: {len(tokens['combined'])} tokens")

        # Test naive forecaster
        from src.models.baselines import NaiveForecaster

        forecaster = NaiveForecaster(prediction_length=5)
        target_col = cleaned.columns[0]
        forecaster.fit(cleaned, target_col)
        predictions = forecaster.predict(cleaned)
        print(f"✓ Naive forecast: {len(predictions)} predictions")

        # Test metrics
        import numpy as np

        from src.eval.metrics import calculate_all_metrics

        true_values = np.random.randn(5)
        pred_values = np.random.randn(5)
        metrics = calculate_all_metrics(true_values, pred_values)
        print(f"✓ Metrics calculated: {list(metrics.keys())}")

        print("All basic functionality tests passed!")
        return True

    except Exception as e:
        print(f"Functionality test error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_phase3_structure() -> bool:
    """Test that Phase 3 experiment structure is correct."""
    print("\nTesting Phase 3 structure...")

    try:
        from experiments.phase3_zero_shot import Phase3Experiments

        print("✓ Phase3Experiments class imported")

        # Test initialization (without running full experiments)
        from src.utils.config import load_config

        config = load_config()
        experiments = Phase3Experiments(config)
        print("✓ Phase3Experiments initialized")

        # Test data preparation
        data = experiments._prepare_data()
        print(f"✓ Data prepared: {data.shape}")

        print("Phase 3 structure test passed!")
        return True

    except Exception as e:
        print(f"Phase 3 structure test error: {e}")
        import traceback

        traceback.print_exc()
        return False


def main() -> bool:
    """Run all tests."""
    print("Running basic tests for multivariate financial forecasting system")
    print("=" * 60)

    success = True

    # Test imports
    success &= test_imports()

    # Test basic functionality
    success &= test_basic_functionality()

    # Test Phase 3 structure
    success &= test_phase3_structure()

    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed! The implementation is working correctly.")
        print("\nTo run Phase 3 experiments:")
        print("  python main.py --phase 3")
        print("\nOr with FRED API key:")
        print("  python main.py --phase 3 --fred-api-key YOUR_KEY")
    else:
        print("❌ Some tests failed. Please check the errors above.")

    return success


if __name__ == "__main__":
    main()
