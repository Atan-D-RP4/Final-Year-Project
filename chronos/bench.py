from chronos_behavioral_framework import ChronosBehavioralForecaster, BenchmarkRunner
from data_preparation_script import BehavioralDataLoader

# Load your data
loader = BehavioralDataLoader()
your_data = loader.load_sentiment_data("your_file.csv")  # or other load methods

# Initialize model
forecaster = ChronosBehavioralForecaster(
    model_name="amazon/chronos-bolt-small",  # or "chronos-bolt-base" for better performance
    device="cpu",  # or "cuda" if you have GPU
)

# Run benchmark
benchmark = BenchmarkRunner(forecaster)
results = benchmark.run_benchmark(
    data=your_data,
    test_split=0.2,  # 20% for testing
    prediction_length=5,  # Predict 5 steps ahead
    window_size=15,  # Use 15 previous points as context
)

# Display results
benchmark.print_results()
benchmark.plot_results()
