"""
Chronos-Bolt for Sequential Behavioral Data Forecasting
Framework for applying Chronos-Bolt to non-traditional time series data
"""

import numpy as np
import torch
from chronos import BaseChronosPipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from scipy import stats
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
import warnings

warnings.filterwarnings("ignore")


class BehavioralDataTokenizer:
    """
    Tokenizes behavioral data (sentiment scores, user interactions, etc.)
    into pseudo time-series format suitable for Chronos-Bolt
    """

    def __init__(self, window_size: int = 10, quantization_levels: int = 1000):
        self.window_size = window_size
        self.quantization_levels = quantization_levels
        self.scaler = MinMaxScaler(feature_range=(0, quantization_levels - 1))
        self.is_fitted = False

    def sliding_window_features(self, data: np.ndarray) -> np.ndarray:
        """
        Create sliding window features from sequential data
        """
        features = []
        for i in range(len(data) - self.window_size + 1):
            window = data[i : i + self.window_size]

            # Extract statistical features from window
            window_features = [
                np.mean(window),
                np.std(window),
                np.min(window),
                np.max(window),
                np.median(window),
                stats.skew(window),
                stats.kurtosis(window),
                np.sum(np.diff(window) > 0),  # trend up count
                np.sum(np.diff(window) < 0),  # trend down count
                window[-1] - window[0],  # total change
            ]
            features.append(window_features)

        return np.array(features)

    def fit_transform(self, behavioral_data: List[float]) -> torch.Tensor:
        """
        Transform behavioral data into tokenized pseudo time-series
        """
        data = np.array(behavioral_data)

        # Create sliding window features
        features = self.sliding_window_features(data)

        # Normalize and quantize features
        self.scaler.fit(features)
        normalized_features = self.scaler.transform(features)

        # Convert to integer tokens (quantization)
        tokenized_data = np.round(normalized_features).astype(int)

        # Take mean across features to create single time series
        pseudo_timeseries = np.mean(tokenized_data, axis=1)

        self.is_fitted = True
        return torch.tensor(pseudo_timeseries, dtype=torch.float32)

    def transform(self, behavioral_data: List[float]) -> torch.Tensor:
        """
        Transform new behavioral data using fitted scaler
        """
        if not self.is_fitted:
            raise ValueError("Tokenizer must be fitted first")

        data = np.array(behavioral_data)
        features = self.sliding_window_features(data)
        normalized_features = self.scaler.transform(features)
        tokenized_data = np.round(normalized_features).astype(int)
        pseudo_timeseries = np.mean(tokenized_data, axis=1)

        return torch.tensor(pseudo_timeseries, dtype=torch.float32)


class ChronosBehavioralForecaster:
    """
    Main class for applying Chronos-Bolt to behavioral data forecasting
    """

    def __init__(
        self, model_name: str = "amazon/chronos-bolt-small", device: str = "cpu"
    ):
        self.model_name = model_name
        self.device = device
        self.pipeline = None
        self.tokenizer = None
        self.load_model()

    def load_model(self):
        """Load Chronos-Bolt model"""
        print(f"Loading {self.model_name}...")
        self.pipeline = BaseChronosPipeline.from_pretrained(
            self.model_name,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
        )
        print("Model loaded successfully!")

    def prepare_data(
        self, data: List[float], window_size: int = 10
    ) -> Tuple[torch.Tensor, BehavioralDataTokenizer]:
        """
        Prepare behavioral data for forecasting
        """
        self.tokenizer = BehavioralDataTokenizer(window_size=window_size)
        tokenized_data = self.tokenizer.fit_transform(data)
        return tokenized_data, self.tokenizer

    def forecast_zero_shot(
        self, context_data: torch.Tensor, prediction_length: int = 5
    ) -> Dict[str, Any]:
        """
        Perform zero-shot forecasting on behavioral data
        """
        print(f"Performing zero-shot forecast for {prediction_length} steps...")

        # Get quantile predictions
        quantiles, mean = self.pipeline.predict_quantiles(
            context=context_data,
            prediction_length=prediction_length,
            quantile_levels=[0.1, 0.25, 0.5, 0.75, 0.9],
        )

        return {
            "quantiles": quantiles,
            "mean": mean,
            "prediction_length": prediction_length,
        }

    def evaluate_classification(
        self, true_values: np.ndarray, predictions: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate forecasting performance as classification task
        """
        # Convert continuous predictions to binary classifications (above/below median)
        true_median = np.median(true_values)
        pred_median = np.median(predictions)

        true_binary = (true_values > true_median).astype(int)
        pred_binary = (predictions > pred_median).astype(int)

        metrics = {
            "accuracy": accuracy_score(true_binary, pred_binary),
            "f1_score": f1_score(true_binary, pred_binary, average="weighted"),
        }

        # ROC-AUC for probabilistic predictions
        try:
            metrics["auc"] = roc_auc_score(true_binary, predictions)
        except:
            metrics["auc"] = 0.5  # Random baseline

        return metrics

    def evaluate_regression(
        self, true_values: np.ndarray, predictions: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate forecasting performance as regression task
        """
        mse = np.mean((true_values - predictions) ** 2)
        mae = np.mean(np.abs(true_values - predictions))
        mape = np.mean(np.abs((true_values - predictions) / (true_values + 1e-8))) * 100

        # MASE (Mean Absolute Scaled Error)
        naive_forecast = np.roll(true_values, 1)[
            1:
        ]  # Use previous value as naive forecast
        naive_mae = np.mean(np.abs(true_values[1:] - naive_forecast))
        mase = mae / (naive_mae + 1e-8)

        return {
            "mse": float(mse),
            "mae": float(mae),
            "mape": float(mape),
            "mase": float(mase),
        }


class BenchmarkRunner:
    """
    Run comprehensive benchmarks on behavioral forecasting
    """

    def __init__(self, forecaster: ChronosBehavioralForecaster):
        self.forecaster = forecaster
        self.results = {}

    def run_benchmark(
        self,
        data: List[float] | None,
        test_split: float = 0.2,
        prediction_length: int = 5,
        window_size: int = 10,
    ) -> Dict[str, Any]:
        """
        Run complete benchmark pipeline
        """
        print("Starting benchmark evaluation...")

        # Split data
        if data is None or len(data) < (window_size + prediction_length + 10):
            raise ValueError("Insufficient data for benchmarking")
        split_point = int(len(data) * (1 - test_split))
        train_data = data[:split_point]
        test_data = data[split_point:]

        # Prepare training data
        tokenized_train, tokenizer = self.forecaster.prepare_data(
            train_data, window_size
        )

        # Forecast on test data
        test_contexts = []
        test_targets = []

        for i in range(len(test_data) - window_size - prediction_length + 1):
            context = test_data[i : i + window_size]
            target = test_data[i + window_size : i + window_size + prediction_length]

            test_contexts.append(context)
            test_targets.append(target)

        all_predictions = []
        all_targets = []

        for context, target in zip(
            test_contexts[: min(10, len(test_contexts))],
            test_targets[: min(10, len(test_targets))],
        ):
            # Tokenize context
            context_tokenized = tokenizer.transform(
                context + target[:0]
            )  # Only context

            # Forecast
            forecast_result = self.forecaster.forecast_zero_shot(
                context_tokenized, prediction_length
            )

            predictions = forecast_result["mean"][0].numpy()
            all_predictions.extend(predictions)
            all_targets.extend(target)

        # Evaluate
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        classification_metrics = self.forecaster.evaluate_classification(
            all_targets, all_predictions
        )
        regression_metrics = self.forecaster.evaluate_regression(
            all_targets, all_predictions
        )

        self.results = {
            "classification_metrics": classification_metrics,
            "regression_metrics": regression_metrics,
            "predictions": all_predictions,
            "targets": all_targets,
            "data_info": {
                "total_samples": len(data),
                "train_samples": len(train_data),
                "test_samples": len(test_data),
                "prediction_length": prediction_length,
                "window_size": window_size,
            },
        }

        return self.results

    def print_results(self):
        """Print benchmark results"""
        if not self.results:
            print("No results to display. Run benchmark first.")
            return

        print("\n" + "=" * 50)
        print("CHRONOS-BOLT BEHAVIORAL FORECASTING BENCHMARK")
        print("=" * 50)

        print("\nClassification Metrics:")
        for metric, value in self.results["classification_metrics"].items():
            print(f"  {metric.upper()}: {value:.4f}")

        print("\nRegression Metrics:")
        for metric, value in self.results["regression_metrics"].items():
            print(f"  {metric.upper()}: {value:.4f}")

        print("\nData Information:")
        for key, value in self.results["data_info"].items():
            print(f"  {key}: {value}")

    def plot_results(self):
        """Plot forecast results"""
        if not self.results:
            print("No results to plot. Run benchmark first.")
            return

        predictions = self.results["predictions"]
        targets = self.results["targets"]

        plt.figure(figsize=(12, 8))

        # Subplot 1: Predictions vs Targets
        plt.subplot(2, 2, 1)
        plt.scatter(targets, predictions, alpha=0.6)
        plt.plot(
            [targets.min(), targets.max()], [targets.min(), targets.max()], "r--", lw=2
        )
        plt.xlabel("True Values")
        plt.ylabel("Predictions")
        plt.title("Predictions vs True Values")

        # Subplot 2: Time series plot
        plt.subplot(2, 2, 2)
        plt.plot(targets, label="True", alpha=0.7)
        plt.plot(predictions, label="Predicted", alpha=0.7)
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.title("Time Series Comparison")
        plt.legend()

        # Subplot 3: Residuals
        plt.subplot(2, 2, 3)
        residuals = targets - predictions
        plt.hist(residuals, bins=20, alpha=0.7)
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        plt.title("Residual Distribution")

        # Subplot 4: Error over time
        plt.subplot(2, 2, 4)
        plt.plot(np.abs(residuals))
        plt.xlabel("Time Step")
        plt.ylabel("Absolute Error")
        plt.title("Absolute Error Over Time")

        plt.tight_layout()
        plt.show()


# Example usage function
def run_sentiment_analysis_example():
    """
    Example: Apply Chronos-Bolt to sentiment score data
    """
    # Generate synthetic sentiment data (replace with your actual data)
    np.random.seed(42)
    n_samples = 200
    time_steps = np.linspace(0, 4 * np.pi, n_samples)

    # Create synthetic sentiment scores with trend and noise
    sentiment_scores = (
        0.5
        + 0.3 * np.sin(time_steps)
        + 0.2 * np.sin(3 * time_steps)
        + 0.1 * np.random.randn(n_samples)
        + 0.001 * time_steps  # Slight upward trend
    )

    # Normalize to [0, 1] range (typical for sentiment scores)
    sentiment_scores = (sentiment_scores - sentiment_scores.min()) / (
        sentiment_scores.max() - sentiment_scores.min()
    )

    print("Generated synthetic sentiment data with", len(sentiment_scores), "samples")

    # Initialize forecaster
    forecaster = ChronosBehavioralForecaster(
        model_name="amazon/chronos-bolt-small",
        device="cpu",  # Use "cuda" if you have GPU
    )

    # Run benchmark
    benchmark = BenchmarkRunner(forecaster)
    results = benchmark.run_benchmark(
        data=sentiment_scores.tolist(),
        test_split=0.2,
        prediction_length=5,
        window_size=15,
    )

    # Display results
    benchmark.print_results()
    benchmark.plot_results()

    return results


if __name__ == "__main__":
    # Run the example
    results = run_sentiment_analysis_example()
