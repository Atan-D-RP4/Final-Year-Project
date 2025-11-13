#!/usr/bin/env python3
"""Minimal Phase 3 experiment test with synthetic data."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent))


def create_synthetic_data() -> pd.DataFrame:
    """Create synthetic financial data for testing."""

    np.random.seed(42)
    dates = pd.date_range("2020-01-01", "2023-12-31", freq="D")

    # Create synthetic price series with trend and noise
    trend = np.linspace(100, 150, len(dates))
    noise = np.random.normal(0, 2, len(dates))
    prices = trend + noise

    # Create synthetic features
    data = pd.DataFrame(
        {
            "date": dates,
            "price": prices,
            "volume": np.random.randint(1000, 10000, len(dates)),
            "returns": np.concatenate([[0], np.diff(prices) / prices[:-1]]),
        }
    )

    data.set_index("date", inplace=True)
    return data


def test_minimal_experiment() -> bool:
    """Test minimal experiment with synthetic data."""

    print("🧪 Testing Minimal Phase 3 Experiment")
    print("=" * 45)

    try:
        from experiments.phase3.zero_shot import ZeroShotExperiment
        from src.eval.metrics import ForecastEvaluator
        from src.models.baselines import MeanForecaster, NaiveForecaster

        # Create synthetic data
        data = create_synthetic_data()
        print(f"✅ Created synthetic data: {data.shape}")

        # Create experiment
        ZeroShotExperiment()
        print("✅ Created experiment")

        # Simple test with just 2 models
        models = {
            "Naive": NaiveForecaster(),
            "Mean": MeanForecaster(),
        }

        # Split data
        split_idx = int(len(data) * 0.8)
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]

        target_col = "price"
        horizon = 10

        print(f"✅ Split data: train={train_data.shape}, test={test_data.shape}")

        # Fit models
        for model_name, model in models.items():
            try:
                model.fit(train_data, target_col)
                print(f"✅ Fitted {model_name}")
            except Exception as e:
                print(f"❌ Failed to fit {model_name}: {e}")
                return False

        # Generate forecasts
        forecasts = {}
        for model_name, model in models.items():
            try:
                pred = model.forecast(test_data, target_col, horizon)
                forecasts[model_name] = pred
                print(f"✅ Forecasted with {model_name}: {len(pred)} values")
            except Exception as e:
                print(f"❌ Failed to forecast with {model_name}: {e}")
                return False

        # Evaluate
        evaluator = ForecastEvaluator()
        actual = test_data[target_col].values[:horizon]

        for model_name, pred in forecasts.items():
            try:
                metrics = evaluator.evaluate(actual, pred)
                print(f"✅ Evaluated {model_name}: MAE={metrics.get('mae', 'N/A'):.3f}")
            except Exception as e:
                print(f"❌ Failed to evaluate {model_name}: {e}")
                return False

        print("✅ Minimal experiment test passed")
        return True

    except Exception as e:
        print(f"❌ Minimal experiment test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main() -> int:
    """Run minimal experiment test."""

    success = test_minimal_experiment()

    if success:
        print("\n🎉 Minimal experiment test successful!")
        print("The project structure and basic functionality are working.")
        return 0
    else:
        print("\n⚠️  Minimal experiment test failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
