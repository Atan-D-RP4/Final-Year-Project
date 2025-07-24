#!/usr/bin/env python3
"""
Quick Setup and Demo Script for Chronos-Bolt Behavioral Forecasting
Run this script to test the framework with sample data
"""

import sys
import subprocess
import os


def install_dependencies():
    """Install required packages"""

    print("Installing required packages...")

    try:
        subprocess.check_call(["uv", "sync"])
        print("✓ dependencies synced installed successfully")
    except subprocess.CalledProcessError:
        print("✗ Failed to sync dependencies. Please check your environment.")


def create_sample_data():
    """Create sample datasets for testing"""
    import pandas as pd
    import numpy as np

    # Create sample sentiment data
    np.random.seed(42)
    n_samples = 300

    # Sentiment scores (0-1 range)
    sentiment_data = {
        "timestamp": pd.date_range("2024-01-01", periods=n_samples, freq="H"),
        "sentiment_score": np.random.beta(
            2, 2, n_samples
        ),  # Beta distribution for realistic sentiment
        "user_id": np.random.randint(1, 50, n_samples),
    }

    # User engagement data
    engagement_data = {
        "timestamp": pd.date_range("2024-01-01", periods=n_samples, freq="H"),
        "likes": np.random.poisson(10, n_samples),
        "shares": np.random.poisson(3, n_samples),
        "comments": np.random.poisson(5, n_samples),
        "views": np.random.poisson(100, n_samples),
    }

    # Clickstream data
    events = ["click", "scroll", "page_view", "hover", "exit", "purchase"]
    clickstream_data = {
        "timestamp": pd.date_range("2024-01-01", periods=n_samples, freq="min"),
        "event_type": np.random.choice(
            events, n_samples, p=[0.3, 0.2, 0.2, 0.15, 0.1, 0.05]
        ),
        "user_session": np.random.randint(1, 30, n_samples),
    }

    # Save sample data
    os.makedirs("sample_data", exist_ok=True)

    pd.DataFrame(sentiment_data).to_csv("sample_data/sentiment_scores.csv", index=False)
    pd.DataFrame(engagement_data).to_csv("sample_data/engagement_data.csv", index=False)
    pd.DataFrame(clickstream_data).to_csv(
        "sample_data/clickstream_data.csv", index=False
    )

    print("✓ Sample data created in 'sample_data/' directory")


def run_basic_demo():
    """Run a basic demo of the framework"""
    try:
        from chronos_behavioral_framework import (
            ChronosBehavioralForecaster,
            BenchmarkRunner,
        )
        from data_preparation_script import BehavioralDataLoader

        print("\n" + "=" * 50)
        print("RUNNING CHRONOS-BOLT BEHAVIORAL FORECASTING DEMO")
        print("=" * 50)

        # Load sample data
        loader = BehavioralDataLoader()

        # Try to load from file, fallback to synthetic
        try:
            sentiment_data = pd.read_csv("sample_data/sentiment_scores.csv")[
                "sentiment_score"
            ].tolist()
            print("✓ Loaded sentiment data from file")
        except:
            sentiment_data = loader.generate_synthetic_sentiment(200)
            print("✓ Generated synthetic sentiment data")

        print(f"Dataset size: {len(sentiment_data)} samples")
        print(f"Value range: [{min(sentiment_data):.3f}, {max(sentiment_data):.3f}]")

        # Initialize forecaster
        print("\nInitializing Chronos-Bolt model...")
        forecaster = ChronosBehavioralForecaster(
            model_name="amazon/chronos-bolt-small", device="cpu"
        )

        # Run benchmark
        print("\nRunning benchmark evaluation...")
        benchmark = BenchmarkRunner(forecaster)
        results = benchmark.run_benchmark(
            data=sentiment_data, test_split=0.2, prediction_length=5, window_size=10
        )

        # Display results
        benchmark.print_results()

        # Create simple visualization
        try:
            benchmark.plot_results()
        except Exception as e:
            print(f"Note: Could not create plots - {e}")

        print("\n✓ Demo completed successfully!")
        return results

    except ImportError as e:
        print(f"✗ Import error: {e}")
        print(
            "Make sure all required packages are installed and the framework files are in the same directory"
        )
        return None
    except Exception as e:
        print(f"✗ Error running demo: {e}")
        return None


def print_usage_instructions():
    """Print usage instructions"""
    print("\n" + "=" * 60)
    print("USAGE INSTRUCTIONS")
    print("=" * 60)
    print("""
1. BASIC USAGE:
   python quick_setup_script.py

2. WITH YOUR OWN DATA:
   - Replace sample data in 'sample_data/' directory
   - Ensure CSV files have appropriate columns:
     * sentiment_scores.csv: 'sentiment_score' column
     * engagement_data.csv: 'likes', 'shares', 'comments', 'views'
     * clickstream_data.csv: 'event_type' column

3. CUSTOMIZE PARAMETERS:
   - Edit the framework files to adjust:
     * Window size (default: 10)
     * Prediction length (default: 5)
     * Model size (chronos-bolt-small/base/large)
     * Quantization levels (default: 1000)

4. EVALUATION METRICS:
   - Classification: F1-score, Accuracy, AUC
   - Regression: MSE, MAE, MAPE, MASE

5. FOR RESEARCH:
   - Use the comprehensive benchmark in data_preparation_script.py
   - Experiment with different tokenization strategies
   - Compare zero-shot vs fine-tuned performance
""")


def main():
    """Main execution function"""
    print("Chronos-Bolt Behavioral Forecasting Setup")
    print("=" * 45)

    # Check if dependencies should be installed
    install_deps = input("Install dependencies? (y/n): ").lower().strip() == "y"
    if install_deps:
        install_dependencies()

    # Create sample data
    create_sample = input("Create sample data? (y/n): ").lower().strip() == "y"
    if create_sample:
        create_sample_data()

    # Run demo
    run_demo = input("Run demo? (y/n): ").lower().strip() == "y"
    if run_demo:
        results = run_basic_demo()
        if results:
            print(f"\nDemo Results Summary:")
            print(f"F1 Score: {results['classification_metrics']['f1_score']:.4f}")
            print(f"MASE: {results['regression_metrics']['mase']:.4f}")

    # Print usage instructions
    print_usage_instructions()


if __name__ == "__main__":
    main()
