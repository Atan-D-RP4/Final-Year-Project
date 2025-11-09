#!/usr/bin/env python3
"""Simple test of Phase 3 experiment without heavy dependencies."""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_experiment_structure():
    """Test the experiment structure without running full computation."""
    
    print("🧪 Testing Phase 3 Experiment Structure")
    print("=" * 50)
    
    try:
        from experiments.phase3.zero_shot import ZeroShotExperiment
        from src.utils.config import ExperimentConfig
        
        # Test config creation
        config = ExperimentConfig()
        print("✅ ExperimentConfig created")
        
        # Test experiment creation
        experiment = ZeroShotExperiment()
        print("✅ ZeroShotExperiment created")
        
        # Test that the run method exists
        if hasattr(experiment, 'run'):
            print("✅ ZeroShotExperiment has run method")
        else:
            print("❌ ZeroShotExperiment missing run method")
            return False
            
        print("✅ Experiment structure test passed")
        return True
        
    except Exception as e:
        print(f"❌ Experiment structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_creation():
    """Test model creation without training."""
    
    print("\n🤖 Testing Model Creation")
    print("=" * 30)
    
    try:
        from src.models.baselines import (
            NaiveForecaster, 
            SeasonalNaiveForecaster,
            MeanForecaster,
            ExponentialSmoothingForecaster,
            ARIMAForecaster,
            LinearRegressionForecaster,
            VARForecaster,
            LSTMForecaster,
            EnsembleForecaster
        )
        from src.models.chronos_wrapper import ChronosFinancialForecaster
        
        # Test baseline model creation
        models = {
            "Naive": NaiveForecaster(),
            "Seasonal Naive": SeasonalNaiveForecaster(),
            "Mean": MeanForecaster(),
            "Exponential Smoothing": ExponentialSmoothingForecaster(),
            "ARIMA": ARIMAForecaster(),
            "Linear Regression": LinearRegressionForecaster(lags=20, horizon=1),
            "VAR": VARForecaster(),
            "LSTM": LSTMForecaster(sequence_length=20, device="cpu"),
            "Ensemble": EnsembleForecaster(),
        }
        
        print(f"✅ Created {len(models)} baseline models")
        
        # Test Chronos model creation
        chronos = ChronosFinancialForecaster(
            prediction_length=24,
            device="cpu",
        )
        print("✅ Created ChronosFinancialForecaster")
        
        print("✅ Model creation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Model creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_evaluation_setup():
    """Test evaluation setup."""
    
    print("\n📊 Testing Evaluation Setup")
    print("=" * 35)
    
    try:
        from src.eval.metrics import ForecastEvaluator
        
        evaluator = ForecastEvaluator()
        print("✅ Created ForecastEvaluator")
        
        print("✅ Evaluation setup test passed")
        return True
        
    except Exception as e:
        print(f"❌ Evaluation setup test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    
    tests = [
        test_experiment_structure,
        test_model_creation,
        test_evaluation_setup,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📈 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Project structure is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())