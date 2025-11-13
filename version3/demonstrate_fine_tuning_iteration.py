"""Demonstration of fine-tuning iteration with detailed observability.

This script shows how to use the FineTuningIterator module to systematically
iterate on model fine-tuning with comprehensive tracking and analysis.
"""

import pandas as pd
import numpy as np

from src.models.fine_tuning_iterator import (
    FineTuningIterator,
    HyperparameterConfig,
    ExperimentAnalyzer,
)


def create_sample_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create sample financial data for demonstration.

    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    # Create synthetic time series data
    dates = pd.date_range("2020-01-01", "2024-12-31", freq="D")
    n = len(dates)

    # Generate price series with trend and seasonality
    np.random.seed(42)
    trend = np.linspace(100, 150, n)
    seasonal = 10 * np.sin(np.linspace(0, 4 * np.pi, n))
    noise = np.random.normal(0, 2, n)
    close_price = trend + seasonal + noise

    # Generate features
    data = pd.DataFrame(
        {
            "Close": close_price,
            "SMA_20": pd.Series(close_price).rolling(20).mean().values,
            "SMA_50": pd.Series(close_price).rolling(50).mean().values,
            "Volume": np.random.uniform(1000000, 5000000, n),
            "Volatility": pd.Series(close_price).pct_change().rolling(20).std().values * 100,
        },
        index=dates,
    )

    # Forward fill NaNs from rolling calculations
    data = data.bfill().ffill()

    # Split data: 70% train, 15% val, 15% test
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    train_data = data.iloc[:n_train].copy()
    val_data = data.iloc[n_train : n_train + n_val].copy()
    test_data = data.iloc[n_train + n_val :].copy()

    return train_data, val_data, test_data


def example_1_single_experiment():
    """Example 1: Run a single fine-tuning experiment."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Single Fine-Tuning Experiment")
    print("=" * 80)

    # Setup data
    train_data, val_data, test_data = create_sample_data()

    # Initialize iterator
    iterator = FineTuningIterator(base_dir="results/fine_tuning_demo")

    # Create hyperparameter configuration
    config = HyperparameterConfig(learning_rate=1e-4, epochs=3, batch_size=32)

    # Run experiment
    result = iterator.run_experiment(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        target_col="Close",
        hyperparams=config,
        experiment_name="single_exp",
        notes="Baseline configuration with default settings",
    )

    print(f"\nExperiment ID: {result.experiment_id}")
    print(f"Timestamp: {result.timestamp}")
    print(f"\nHyperparameters:")
    for key, value in result.hyperparams.items():
        print(f"  {key}: {value}")

    print(f"\nTest Metrics:")
    for key, value in result.test_metrics.items():
        print(f"  {key}: {value:.6f}")

    print(f"\nData Insights:")
    print(f"  Train quality: {result.data_insights.get('train_quality', {})}")
    print(f"  Target distribution: {result.data_insights.get('target_distribution', {})}")

    if result.error:
        print(f"\nError occurred: {result.error}")

    return result


def example_2_grid_search():
    """Example 2: Run hyperparameter grid search."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Hyperparameter Grid Search")
    print("=" * 80)

    # Setup data
    train_data, val_data, test_data = create_sample_data()

    # Initialize iterator
    iterator = FineTuningIterator(base_dir="results/fine_tuning_demo")

    # Define parameter grid
    param_grid = {
        "learning_rate": [5e-5, 1e-4, 5e-4],
        "epochs": [3, 5],
        "weight_decay": [0.0, 0.01],
    }

    # Run grid search
    results = iterator.grid_search(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        target_col="Close",
        param_grid=param_grid,
        experiment_name="grid_search",
    )

    print(f"\nCompleted {len(results)} experiments")

    # Compare results
    comparison_df = iterator.compare_experiments()
    print("\nComparison of experiments:")
    print(
        comparison_df[["experiment_id", "hp_learning_rate", "hp_epochs", "test_mae", "test_rmse"]]
    )

    # Find best experiment
    best = iterator.identify_best_experiment(metric="mae")
    if best:
        print(f"\nBest experiment: {best.experiment_id}")
        print(f"Best MAE: {best.test_metrics.get('mae', 'N/A'):.6f}")

    return results


def example_3_experiment_analysis():
    """Example 3: Analyze experiments to identify what helps/hurts."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Experiment Analysis & Insights")
    print("=" * 80)

    # Setup data and run grid search
    train_data, val_data, test_data = create_sample_data()

    iterator = FineTuningIterator(base_dir="results/fine_tuning_demo")

    # Run smaller grid search
    param_grid = {
        "learning_rate": [1e-4, 5e-4],
        "batch_size": [16, 32, 64],
    }

    results = iterator.grid_search(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        target_col="Close",
        param_grid=param_grid,
        experiment_name="analysis_demo",
    )

    # Analyze results
    analyzer = ExperimentAnalyzer(iterator.tracker.experiments)

    print("\nSummary Statistics:")
    summary = analyzer.get_summary_statistics()
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print("\nMost Important Hyperparameters (by correlation with MAE):")
    important_hps = analyzer.identify_important_hyperparameters(metric="mae", top_n=3)
    if not important_hps.empty:
        print(important_hps.to_string(index=False))
    else:
        print("  No significant correlations found")

    print("\nOverfitting Analysis:")
    overfitting_df = analyzer.detect_overfitting()
    if not overfitting_df.empty:
        print(
            overfitting_df[["experiment_id", "train_mae", "val_mae", "overfitting_flag"]].to_string(
                index=False
            )
        )
    else:
        print("  No experiments to analyze")

    print("\nConvergence Analysis:")
    convergence_df = analyzer.identify_convergence_issues()
    if not convergence_df.empty:
        print(
            convergence_df[["experiment_id", "final_loss", "convergence_issue"]].to_string(
                index=False
            )
        )
    else:
        print("  No training history to analyze")

    return analyzer


def example_4_learning_curves():
    """Example 4: Analyze learning curves and training dynamics."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Learning Curves & Training Dynamics")
    print("=" * 80)

    train_data, val_data, test_data = create_sample_data()

    iterator = FineTuningIterator(base_dir="results/fine_tuning_demo")

    # Run experiment with different learning rates
    configs = [
        HyperparameterConfig(learning_rate=5e-5, epochs=5),
        HyperparameterConfig(learning_rate=1e-4, epochs=5),
        HyperparameterConfig(learning_rate=5e-4, epochs=5),
    ]

    results = []
    for i, config in enumerate(configs):
        result = iterator.run_experiment(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            target_col="Close",
            hyperparams=config,
            experiment_name=f"learning_curve_{i + 1}",
            notes=f"Learning rate: {config.learning_rate}",
        )
        results.append(result)

    print("\nLearning Rate Impact Analysis:")
    for result in results:
        lr = result.hyperparams.get("learning_rate")
        mae = result.test_metrics.get("mae", "N/A")
        print(f"  LR={lr:.0e}: MAE={mae:.6f if isinstance(mae, float) else mae}")

    print("\nTraining History Summary:")
    for i, result in enumerate(results):
        if result.training_history:
            history_df = pd.DataFrame(result.training_history)
            if "loss" in history_df.columns:
                final_loss = history_df["loss"].iloc[-1]
                initial_loss = history_df["loss"].iloc[0]
                print(
                    f"  Exp {i + 1}: Initial loss={initial_loss:.6f}, Final loss={final_loss:.6f}"
                )


def example_5_practical_workflow():
    """Example 5: Practical workflow for iterative refinement."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Practical Iteration Workflow")
    print("=" * 80)

    train_data, val_data, test_data = create_sample_data()
    iterator = FineTuningIterator(base_dir="results/fine_tuning_demo")

    print("\nPhase 1: Baseline Establishment")
    print("-" * 40)

    baseline = iterator.run_experiment(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        target_col="Close",
        hyperparams=HyperparameterConfig(),
        experiment_name="phase1_baseline",
        notes="Establish baseline with default config",
    )

    baseline_mae = baseline.test_metrics.get("mae", float("inf"))
    print(f"Baseline MAE: {baseline_mae:.6f}")

    print("\nPhase 2: Learning Rate Optimization")
    print("-" * 40)

    lr_results = iterator.grid_search(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        target_col="Close",
        param_grid={"learning_rate": [5e-5, 1e-4, 5e-4, 1e-3]},
        experiment_name="phase2_lr_search",
    )

    best_lr_exp = iterator.identify_best_experiment(metric="mae")
    if best_lr_exp:
        best_mae = best_lr_exp.test_metrics.get("mae", float("inf"))
        improvement = (baseline_mae - best_mae) / baseline_mae * 100
        print(f"Best MAE after LR search: {best_mae:.6f} ({improvement:.1f}% improvement)")

    print("\nPhase 3: Regularization Tuning")
    print("-" * 40)

    best_lr = best_lr_exp.hyperparams.get("learning_rate") if best_lr_exp else 1e-4

    reg_results = iterator.grid_search(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        target_col="Close",
        param_grid={
            "learning_rate": [best_lr],
            "weight_decay": [0.0, 0.001, 0.01, 0.05],
        },
        experiment_name="phase3_regularization",
    )

    best_final = iterator.identify_best_experiment(metric="mae")
    if best_final:
        final_mae = best_final.test_metrics.get("mae", float("inf"))
        total_improvement = (baseline_mae - final_mae) / baseline_mae * 100
        print(f"Final MAE: {final_mae:.6f} ({total_improvement:.1f}% total improvement)")

    print(f"\nBest configuration found:")
    for key, value in best_final.hyperparams.items():
        print(f"  {key}: {value}")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("FINE-TUNING ITERATION EXAMPLES")
    print("With Comprehensive Observability & Analysis")
    print("=" * 80)

    # Note: These examples use mock/simplified implementations since full
    # model loading may require additional dependencies.

    print("\nNote: These examples demonstrate the API and structure.")
    print("Full functionality requires model loading and infrastructure.")

    try:
        example_1_single_experiment()
    except Exception as e:
        print(f"Example 1 note: {e}")

    try:
        example_2_grid_search()
    except Exception as e:
        print(f"Example 2 note: {e}")

    try:
        example_3_experiment_analysis()
    except Exception as e:
        print(f"Example 3 note: {e}")

    try:
        example_4_learning_curves()
    except Exception as e:
        print(f"Example 4 note: {e}")

    try:
        example_5_practical_workflow()
    except Exception as e:
        print(f"Example 5 note: {e}")

    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
