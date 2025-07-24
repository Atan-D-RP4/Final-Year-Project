"""
Data Preparation Scripts for Different Types of Behavioral Data
Supports: Sentiment scores, User clickstreams, Engagement metrics, etc.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import json
from datetime import datetime, timedelta
import re

class BehavioralDataLoader:
    """
    Load and preprocess various types of behavioral data
    """

    @staticmethod
    def load_sentiment_data(file_path: str = None, format: str = 'csv') -> List[float]:
        """
        Load sentiment score data from various formats
        """
        if file_path is None:
            # Generate synthetic sentiment data for testing
            return BehavioralDataLoader.generate_synthetic_sentiment(500)

        if format.lower() == 'csv':
            df = pd.read_csv(file_path)
            # Assume sentiment column exists (adjust column name as needed)
            sentiment_cols = ['sentiment', 'sentiment_score', 'score', 'rating']
            for col in sentiment_cols:
                if col in df.columns:
                    return df[col].tolist()
            raise ValueError(f"No sentiment column found. Available columns: {df.columns.tolist()}")

        elif format.lower() == 'json':
            with open(file_path, 'r') as f:
                data = json.load(f)
            # Assume data is list of dicts with sentiment scores
            if isinstance(data, list) and len(data) > 0:
                if isinstance(data[0], dict):
                    sentiment_keys = ['sentiment', 'sentiment_score', 'score']
                    for key in sentiment_keys:
                        if key in data[0]:
                            return [item[key] for item in data]
                elif isinstance(data[0], (int, float)):
                    return data
            raise ValueError("Could not extract sentiment scores from JSON data")

    @staticmethod
    def load_clickstream_data(file_path: str = None) -> List[float]:
        """
        Convert clickstream data to numeric sequence
        """
        if file_path is None:
            return BehavioralDataLoader.generate_synthetic_clickstream(300)

        df = pd.read_csv(file_path)

        # Convert clickstream events to numeric values
        if 'event_type' in df.columns:
            event_mapping = {
                'click': 1.0,
                'scroll': 0.5,
                'hover': 0.3,
                'page_view': 0.8,
                'purchase': 2.0,
                'add_to_cart': 1.5,
                'search': 0.7,
                'exit': 0.0
            }

            # Map events to numeric values
            numeric_sequence = []
            for event in df['event_type']:
                numeric_sequence.append(event_mapping.get(event.lower(), 0.5))

            return numeric_sequence

        raise ValueError("No 'event_type' column found in clickstream data")

    @staticmethod
    def load_engagement_data(file_path: str = None) -> List[float]:
        """
        Load user engagement metrics (likes, shares, comments, etc.)
        """
        if file_path is None:
            return BehavioralDataLoader.generate_synthetic_engagement(400)

        df = pd.read_csv(file_path)

        # Combine multiple engagement metrics
        engagement_cols = ['likes', 'shares', 'comments', 'views', 'engagement_rate']
        available_cols = [col for col in engagement_cols if col in df.columns]

        if not available_cols:
            raise ValueError(f"No engagement columns found. Available: {df.columns.tolist()}")

        # Normalize and combine metrics
        engagement_scores = []
        for _, row in df.iterrows():
            score = sum(row[col] for col in available_cols if pd.notna(row[col]))
            engagement_scores.append(score)

        # Normalize to [0, 1] range
        min_score, max_score = min(engagement_scores), max(engagement_scores)
        if max_score > min_score:
            engagement_scores = [(score - min_score) / (max_score - min_score)
                               for score in engagement_scores]

        return engagement_scores

    @staticmethod
    def load_dialogue_sentiment(file_path: str = None) -> List[float]:
        """
        Load dialogue sentiment progression data
        """
        if file_path is None:
            return BehavioralDataLoader.generate_synthetic_dialogue_sentiment(250)

        df = pd.read_csv(file_path)

        # Assume dialogue data has 'message' and 'sentiment' columns
        if 'sentiment' in df.columns:
            return df['sentiment'].tolist()
        elif 'message' in df.columns:
            # Simple sentiment analysis (replace with actual sentiment analyzer)
            sentiments = []
            positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'like', 'happy']
            negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'sad', 'angry']

            for message in df['message']:
                if pd.isna(message):
                    sentiments.append(0.5)
                    continue

                message_lower = str(message).lower()
                pos_count = sum(1 for word in positive_words if word in message_lower)
                neg_count = sum(1 for word in negative_words if word in message_lower)

                # Simple sentiment score
                if pos_count > neg_count:
                    sentiments.append(0.7 + 0.3 * min(pos_count / 3, 1))
                elif neg_count > pos_count:
                    sentiments.append(0.3 - 0.3 * min(neg_count / 3, 1))
                else:
                    sentiments.append(0.5)

            return sentiments

        raise ValueError("No sentiment or message column found")

    # Synthetic data generators for testing
    @staticmethod
    def generate_synthetic_sentiment(n_samples: int = 500) -> List[float]:
        """Generate synthetic sentiment score data"""
        np.random.seed(42)
        t = np.linspace(0, 4*np.pi, n_samples)

        # Create sentiment pattern with trends and noise
        sentiment = (
            0.5 +  # Baseline neutral sentiment
            0.2 * np.sin(t) +  # Daily pattern
            0.1 * np.sin(5*t) +  # Shorter cycles
            0.05 * np.cumsum(np.random.randn(n_samples)) / n_samples +  # Random walk
            0.1 * np.random.randn(n_samples)  # Noise
        )

        # Clip to [0, 1] range
        sentiment = np.clip(sentiment, 0, 1)
        return sentiment.tolist()

    @staticmethod
    def generate_synthetic_clickstream(n_samples: int = 300) -> List[float]:
        """Generate synthetic clickstream data"""
        np.random.seed(123)

        # Simulate user session patterns
        events = []
        session_length = np.random.poisson(20, n_samples // 20)

        for length in session_length:
            # Start with page view
            session = [0.8]

            for _ in range(length - 1):
                # Probability of different events based on current state
                last_event = session[-1]

                if last_event == 0.8:  # After page view
                    next_event = np.random.choice([1.0, 0.5, 0.7], p=[0.4, 0.4, 0.2])
                elif last_event == 1.0:  # After click
                    next_event = np.random.choice([0.8, 1.5, 0.0], p=[0.5, 0.3, 0.2])
                else:  # Other events
                    next_event = np.random.choice([1.0, 0.5, 0.0], p=[0.5, 0.3, 0.2])

                session.append(next_event)

            events.extend(session)

            if len(events) >= n_samples:
                break

        return events[:n_samples]

    @staticmethod
    def generate_synthetic_engagement(n_samples: int = 400) -> List[float]:
        """Generate synthetic engagement data"""
        np.random.seed(456)

        # Simulate engagement patterns with viral effects
        engagement = []
        current_level = 0.3

        for i in range(n_samples):
            # Random walk with occasional viral spikes
            change = np.random.normal(0, 0.02)

            # Occasional viral spike
            if np.random.random() < 0.05:
                change += np.random.exponential(0.3)

            current_level += change
            current_level = max(0, min(1, current_level))  # Clip to [0, 1]

            engagement.append(current_level)

        return engagement

    @staticmethod
    def generate_synthetic_dialogue_sentiment(n_samples: int = 250) -> List[float]:
        """Generate synthetic dialogue sentiment progression"""
        np.random.seed(789)

        # Simulate conversation sentiment evolution
        sentiments = []
        current_sentiment = 0.5  # Start neutral

        for i in range(n_samples):
            # Sentiment tends to persist but can change
            momentum = 0.8
            current_sentiment = (momentum * current_sentiment +
                               (1 - momentum) * (0.5 + np.random.normal(0, 0.2)))

            # Add conversational dynamics
            if i > 0:
                # Response to previous sentiment
                if sentiments[-1] > 0.7:  # Positive response to positive
                    current_sentiment += np.random.normal(0.1, 0.05)
                elif sentiments[-1] < 0.3:  # Negative response to negative
                    current_sentiment += np.random.normal(-0.1, 0.05)

            current_sentiment = np.clip(current_sentiment, 0, 1)
            sentiments.append(current_sentiment)

        return sentiments

class DatasetBenchmarker:
    """
    Run benchmarks on multiple datasets
    """

    def __init__(self, forecaster):
        self.forecaster = forecaster
        self.results = {}

    def run_multiple_datasets(self, datasets: Dict[str, List[float]],
                            test_split: float = 0.2,
                            prediction_length: int = 5,
                            window_size: int = 10) -> Dict[str, Any]:
        """
        Run benchmarks on multiple datasets
        """
        from chronos_behavioral_framework import BenchmarkRunner

        all_results = {}

        for dataset_name, data in datasets.items():
            print(f"\n{'='*20} {dataset_name.upper()} {'='*20}")

            benchmark = BenchmarkRunner(self.forecaster)
            results = benchmark.run_benchmark(
                data=data,
                test_split=test_split,
                prediction_length=prediction_length,
                window_size=window_size
            )

            all_results[dataset_name] = results
            benchmark.print_results()

        return all_results

    def compare_datasets(self, results: Dict[str, Any]):
        """
        Compare performance across datasets
        """
        import matplotlib.pyplot as plt

        # Extract metrics for comparison
        dataset_names = list(results.keys())
        f1_scores = [results[name]['classification_metrics']['f1_score'] for name in dataset_names]
        mase_scores = [results[name]['regression_metrics']['mase'] for name in dataset_names]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # F1 scores
        ax1.bar(dataset_names, f1_scores)
        ax1.set_title('F1 Scores Across Datasets')
        ax1.set_ylabel('F1 Score')
        ax1.tick_params(axis='x', rotation=45)

        # MASE scores
        ax2.bar(dataset_names, mase_scores)
        ax2.set_title('MASE Scores Across Datasets')
        ax2.set_ylabel('MASE')
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

        # Print summary table
        print("\n" + "="*60)
        print("PERFORMANCE COMPARISON ACROSS DATASETS")
        print("="*60)
        print(f"{'Dataset':<20} {'F1 Score':<12} {'MASE':<12} {'Accuracy':<12}")
        print("-"*60)

        for name in dataset_names:
            f1 = results[name]['classification_metrics']['f1_score']
            mase = results[name]['regression_metrics']['mase']
            acc = results[name]['classification_metrics']['accuracy']
            print(f"{name:<20} {f1:<12.4f} {mase:<12.4f} {acc:<12.4f}")

# Example usage function
def run_comprehensive_benchmark():
    """
    Run comprehensive benchmark on multiple behavioral datasets
    """
    from chronos_behavioral_framework import ChronosBehavioralForecaster

    # Load different types of behavioral data
    loader = BehavioralDataLoader()

    datasets = {
        'sentiment_scores': loader.load_sentiment_data(),
        'clickstream': loader.load_clickstream_data(),
        'engagement': loader.load_engagement_data(),
        'dialogue_sentiment': loader.load_dialogue_sentiment()
    }

    print("Loaded datasets:")
    for name, data in datasets.items():
        print(f"  {name}: {len(data)} samples, range: [{min(data):.3f}, {max(data):.3f}]")

    # Initialize forecaster
    forecaster = ChronosBehavioralForecaster(
        model_name="amazon/chronos-bolt-small",
        device="cpu"
    )

    # Run benchmarks
    benchmarker = DatasetBenchmarker(forecaster)
    results = benchmarker.run_multiple_datasets(
        datasets=datasets,
        test_split=0.2,
        prediction_length=5,
        window_size=15
    )

    # Compare results
    benchmarker.compare_datasets(results)

    return results

if __name__ == "__main__":
    results = run_comprehensive_benchmark()
