"""Demonstration of Phase 4: Chronos Fine-tuning and Phase 3 Integration."""

from typing import Any

from experiments.phase3.zero_shot import ZeroShotExperiment
from experiments.phase4.fine_tune import Phase4Experiment


def demonstrate_phase4() -> str | None:
    """Demonstrate Phase 4 fine-tuning capabilities."""
    print("🚀 Phase 4: Chronos Fine-tuning Demonstration")
    print("=" * 60)

    # Initialize Phase 4 experiment
    phase4_exp = Phase4Experiment(results_dir="results/phase4")

    print("\n📋 Phase 4 Configuration:")
    print("  - Target: S&P 500 (^GSPC)")
    print("  - Base Model: amazon/chronos-t5-small")
    print("  - Fine-tuning: Mock implementation (for demo)")
    print("  - Results saved to: results/phase4/")

    # Run fine-tuning experiment
    print("\n🔄 Running Phase 4 Fine-tuning Experiment...")
    try:
        results = phase4_exp.run_fine_tuning_experiment(
            target_symbol="^GSPC",
            experiment_name="demo_fine_tune",
            do_hyperparam_search=False,
        )

        print("✅ Phase 4 experiment completed!")
        print(f"   Model saved to: {results.get('fine_tuning', {}).get('model_path', 'N/A')}")

        # Get the model path for Phase 3 integration
        model_path = results.get("fine_tuning", {}).get("model_path")
        return model_path

    except Exception as e:
        print(f"❌ Phase 4 experiment failed: {e}")
        return None


def demonstrate_phase3_with_fine_tuned(model_path: str | None = None) -> None:
    """Demonstrate Phase 3 with fine-tuned Chronos model."""
    print("\n🚀 Phase 3: Zero-shot + Fine-tuned Comparison")
    print("=" * 60)

    # Initialize Phase 3 experiment
    phase3_exp = ZeroShotExperiment(results_dir="results/phase3", log_dir="results/phase3/logs")

    print("\n📋 Phase 3 Configuration:")
    print("  - Target: S&P 500 (^GSPC)")
    print("  - Models: All baselines + Chronos (zero-shot)")
    if model_path:
        print("  - Fine-tuned Model: ✓ Included")
    else:
        print("  - Fine-tuned Model: ✗ Not available")
    print("  - Prediction horizon: 20 steps")
    print("  - Results saved to: results/phase3/")

    # Run comparison experiment
    print("\n🔄 Running Phase 3 Comparison Experiment...")
    try:
        results = phase3_exp.run(
            target_col="Close",  # Use 'Close' after data cleaning
            prediction_length=20,
            experiment_name="comparison_with_fine_tuned",
            fine_tuned_model_path=model_path,
        )

        print("✅ Phase 3 experiment completed!")
        print("   Results saved to: results/phase3/comparison_with_fine_tuned_*")

        # Show model performance summary
        model_results = results.get("model_results", {})
        if model_results:
            print("\n📊 Model Performance Summary:")
            for _model_name, metrics in model_results.items():
                mae = metrics.get("mae", "N/A")
                directional_acc = metrics.get("directional_accuracy", "N/A")
                print(f"  MAE: {mae}, Directional Accuracy: {directional_acc}")

    except Exception as e:
        print(f"❌ Phase 3 experiment failed: {e}")


def show_comparison_insights() -> None:
    """Show insights from comparing zero-shot vs fine-tuned models."""
    print("\n🎯 Key Insights: Zero-shot vs Fine-tuned Chronos")
    print("=" * 60)

    print("\n📈 Expected Performance Improvements:")
    print("  • Fine-tuned Chronos should outperform zero-shot on in-domain financial data")
    print("  • Better calibration and reduced forecast variance")
    print("  • Improved directional accuracy for short-term predictions")
    print("  • More stable performance across different market regimes")

    print("\n🔧 Technical Advantages:")
    print("  • Model adapts to specific data patterns and market dynamics")
    print("  • Reduced reliance on general time-series knowledge")
    print("  • Better handling of financial-specific features and indicators")
    print("  • Potential for domain-specific optimizations")

    print("\n⚠️ Important Considerations:")
    print("  • Fine-tuning requires sufficient training data")
    print("  • Risk of overfitting to historical patterns")
    print("  • Need for careful validation on out-of-sample data")
    print("  • Computational cost vs performance benefit trade-off")

    print("\n📁 Generated Files:")
    print("  • results/phase4/demo_fine_tune_results.json - Fine-tuning results")
    print("  • results/phase4/demo_fine_tune/final_model/ - Saved model")
    print("  • results/phase3/comparison_with_fine_tuned_* - Comparison results")
    print("  • results/phase3/comparison_with_fine_tuned_metrics.png - Performance plots")


def main() -> None:
    """Run complete Phase 4 + Phase 3 integration demonstration."""
    print("🎯 Phase 4 + Phase 3 Integration Demonstration")
    print("=" * 80)
    print("This demo shows:")
    print("1. Phase 4: Fine-tuning Chronos on financial data")
    print("2. Phase 3: Comparing zero-shot vs fine-tuned performance")
    print("3. Integration: Using fine-tuned models in inference")
    print("=" * 80)

    # Phase 4: Fine-tuning
    model_path = demonstrate_phase4()

    # Phase 3: Comparison with fine-tuned model
    demonstrate_phase3_with_fine_tuned(model_path)

    # Show insights
    show_comparison_insights()

    print("\n🎉 Demonstration completed!")
    print("Check the results directories for detailed outputs and plots.")


if __name__ == "__main__":
    main()
