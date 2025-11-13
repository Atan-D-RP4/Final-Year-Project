"""Smoke tests for end-to-end pipeline validation."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis import AblationImportance, AttributionAnalyzer
from src.data.cleaning import DataCleaner, create_features
from src.eval.metrics import (
    ForecastEvaluator,
    calculate_all_metrics,
    directional_accuracy,
    mae,
    mase,
    rmse,
)
from src.models.baselines import (
    ARIMAForecaster,
    EnsembleForecaster,
    ExponentialSmoothingForecaster,
    LinearRegressionForecaster,
    LSTMForecaster,
    MeanForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
)
from src.models.chronos_wrapper import ChronosFinancialForecaster
from src.preprocessing.tokenizer import AdvancedTokenizer, FinancialDataTokenizer
from src.utils.config import DataConfig, EvalConfig, PreprocessingConfig
from src.utils.logger import setup_logger


class TestDataPipeline:
    """Test data fetching and cleaning pipeline."""

    @pytest.fixture
    def sample_data(self):
        """Create sample financial data."""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        data = pd.DataFrame(
            {
                "Open": np.random.uniform(100, 110, 100),
                "High": np.random.uniform(110, 120, 100),
                "Low": np.random.uniform(90, 100, 100),
                "Close": np.random.uniform(100, 110, 100),
                "Volume": np.random.uniform(1e6, 1e7, 100),
            },
            index=dates,
        )
        return data

    def test_data_cleaner(self, sample_data):
        """Test data cleaning."""
        cleaner = DataCleaner()
        cleaned = cleaner.clean_market_data(sample_data.copy())
        assert cleaned.shape[0] > 0
        assert not cleaned.isna().sum().sum() > 0

    def test_feature_creation(self, sample_data):
        """Test feature creation."""
        features = create_features(sample_data.copy())
        assert features.shape[1] > sample_data.shape[1]
        assert "Returns" in features.columns or "Log_Returns" in features.columns

    def test_config_creation(self):
        """Test configuration creation."""
        config = DataConfig()
        assert config.market_symbols is not None
        assert config.start_date is not None
        assert config.end_date is not None


class TestTokenization:
    """Test tokenization strategies."""

    @pytest.fixture
    def sample_series(self):
        """Create sample time series."""
        return pd.DataFrame(
            {"Close": np.sin(np.linspace(0, 4 * np.pi, 200)) + np.random.normal(0, 0.1, 200)}
        )

    def test_financial_tokenizer_uniform(self, sample_series):
        """Test uniform tokenizer."""
        tokenizer = FinancialDataTokenizer(method="uniform", num_bins=10)
        tokenizer.fit(sample_series)
        tokens = tokenizer.transform(sample_series)
        assert "Close" in tokens
        assert len(tokens["Close"]) == len(sample_series)

    def test_financial_tokenizer_quantile(self, sample_series):
        """Test quantile tokenizer."""
        tokenizer = FinancialDataTokenizer(method="quantile", num_bins=10)
        tokenizer.fit(sample_series)
        tokens = tokenizer.transform(sample_series)
        assert "Close" in tokens
        assert len(tokens["Close"]) == len(sample_series)

    def test_financial_tokenizer_kmeans(self, sample_series):
        """Test k-means tokenizer."""
        tokenizer = FinancialDataTokenizer(method="kmeans", num_bins=10)
        tokenizer.fit(sample_series)
        tokens = tokenizer.transform(sample_series)
        assert "Close" in tokens
        assert len(tokens["Close"]) == len(sample_series)

    def test_advanced_tokenizer(self, sample_series):
        """Test advanced tokenizer with technical indicators."""
        tokenizer = AdvancedTokenizer(
            include_technical_indicators=True,
            include_time_features=True,
        )
        tokenizer.fit(sample_series)
        tokens = tokenizer.transform(sample_series)
        assert "Close" in tokens


class TestBaselineModels:
    """Test baseline forecasting models."""

    @pytest.fixture
    def train_test_data(self):
        """Create train/test split."""
        dates = pd.date_range(start="2023-01-01", periods=200, freq="D")
        data = pd.DataFrame(
            {
                "Close": np.sin(np.linspace(0, 4 * np.pi, 200)) + np.random.normal(0, 0.1, 200),
            },
            index=dates,
        )
        split_idx = 150
        return data.iloc[:split_idx], data.iloc[split_idx:]

    def test_naive_forecaster(self, train_test_data):
        """Test naive forecaster."""
        train, test = train_test_data
        model = NaiveForecaster()
        model.fit(train, "Close")
        pred = model.forecast(test, "Close", 20)
        assert len(pred) == 20
        assert not np.isnan(pred).any()

    def test_seasonal_naive_forecaster(self, train_test_data):
        """Test seasonal naive forecaster."""
        train, test = train_test_data
        model = SeasonalNaiveForecaster(seasonal_period=7)
        model.fit(train, "Close")
        pred = model.forecast(test, "Close", 20)
        assert len(pred) == 20
        assert not np.isnan(pred).any()

    def test_mean_forecaster(self, train_test_data):
        """Test mean forecaster."""
        train, test = train_test_data
        model = MeanForecaster()
        model.fit(train, "Close")
        pred = model.forecast(test, "Close", 20)
        assert len(pred) == 20
        assert not np.isnan(pred).any()

    def test_exponential_smoothing_forecaster(self, train_test_data):
        """Test exponential smoothing forecaster."""
        train, test = train_test_data
        model = ExponentialSmoothingForecaster()
        model.fit(train, "Close")
        pred = model.forecast(test, "Close", 20)
        assert len(pred) == 20
        assert not np.isnan(pred).any()

    def test_arima_forecaster(self, train_test_data):
        """Test ARIMA forecaster."""
        train, test = train_test_data
        model = ARIMAForecaster(order=(1, 1, 1))
        model.fit(train, "Close")
        pred = model.forecast(test, "Close", 20)
        assert len(pred) == 20

    def test_linear_regression_forecaster(self, train_test_data):
        """Test linear regression forecaster."""
        train, test = train_test_data
        model = LinearRegressionForecaster(lags=10)
        model.fit(train, "Close")
        pred = model.forecast(test, "Close", 20)
        assert len(pred) == 20
        assert not np.isnan(pred).any()

    def test_ensemble_forecaster(self, train_test_data):
        """Test ensemble forecaster."""
        train, test = train_test_data
        model = EnsembleForecaster()
        model.fit(train, "Close")
        pred = model.forecast(test, "Close", 20)
        assert len(pred) == 20
        assert not np.isnan(pred).any()

    @pytest.mark.slow
    def test_lstm_forecaster(self, train_test_data):
        """Test LSTM forecaster."""
        train, test = train_test_data
        model = LSTMForecaster(sequence_length=10, device="cpu")
        model.fit(train, "Close", epochs=5, batch_size=16)
        pred = model.forecast(test, "Close", 20)
        assert len(pred) == 20
        assert not np.isnan(pred).any()


class TestChronosModel:
    """Test Chronos forecasting model."""

    @pytest.fixture
    def sample_data(self):
        """Create sample financial data."""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        data = pd.DataFrame(
            {
                "Close": np.sin(np.linspace(0, 4 * np.pi, 100)) + np.random.normal(0, 0.1, 100),
            },
            index=dates,
        )
        return data

    def test_chronos_initialization(self):
        """Test Chronos model initialization."""
        model = ChronosFinancialForecaster(prediction_length=20, device="cpu")
        assert model.prediction_length == 20
        assert model.device == "cpu"

    def test_chronos_fit(self, sample_data):
        """Test Chronos model fitting."""
        model = ChronosFinancialForecaster(device="cpu")
        model.fit(sample_data, "Close")
        assert model.tokenizer_data is not None

    def test_chronos_forecast(self, sample_data):
        """Test Chronos zero-shot forecast."""
        model = ChronosFinancialForecaster(prediction_length=20, device="cpu")
        model.fit(sample_data, "Close")
        forecasts = model.forecast_zero_shot(sample_data, "Close", num_samples=10)
        assert "median" in forecasts
        assert "mean" in forecasts
        assert "quantiles" in forecasts
        assert len(forecasts["median"]) == 20

    def test_chronos_save_load(self, sample_data):
        """Test Chronos model save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = ChronosFinancialForecaster(device="cpu")
            model.fit(sample_data, "Close")
            model.save_model(tmpdir)

            # Load model
            model2 = ChronosFinancialForecaster(device="cpu")
            model2.load_saved_model(tmpdir)
            assert model2.tokenizer_data is not None


class TestEvaluationMetrics:
    """Test evaluation metrics."""

    @pytest.fixture
    def predictions(self):
        """Create sample predictions."""
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predicted = np.array([1.1, 2.1, 2.9, 4.1, 4.9])
        return actual, predicted

    def test_mae(self, predictions):
        """Test MAE calculation."""
        actual, predicted = predictions
        error = mae(actual, predicted)
        assert error > 0
        assert error < 1

    def test_rmse(self, predictions):
        """Test RMSE calculation."""
        actual, predicted = predictions
        error = rmse(actual, predicted)
        assert error > 0
        assert error < 1

    def test_mase(self, predictions):
        """Test MASE calculation."""
        actual, predicted = predictions
        error = mase(actual, predicted)
        assert error >= 0

    def test_directional_accuracy(self, predictions):
        """Test directional accuracy."""
        actual, predicted = predictions
        acc = directional_accuracy(actual, predicted)
        assert 0 <= acc <= 1

    def test_forecast_evaluator(self, predictions):
        """Test ForecastEvaluator."""
        actual, predicted = predictions
        evaluator = ForecastEvaluator()
        results = evaluator.evaluate(
            actual, predicted, metrics=["mae", "rmse", "directional_accuracy"]
        )
        assert "mae" in results
        assert "rmse" in results
        assert "directional_accuracy" in results

    def test_calculate_all_metrics(self, predictions):
        """Test calculate_all_metrics function."""
        actual, predicted = predictions
        results = calculate_all_metrics(actual, predicted)
        assert len(results) > 0


class TestAttributionAnalysis:
    """Test attribution analysis methods."""

    @pytest.fixture
    def setup_attribution(self):
        """Setup for attribution tests."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)

        def predict_fn(x):
            return np.dot(x, np.array([1, 2, 3, 0.5, 0.1]))

        return X, y, predict_fn

    def test_ablation_importance(self, setup_attribution):
        """Test ablation importance."""
        X, y, predict_fn = setup_attribution
        ablation = AblationImportance()
        importance = ablation.compute_importance(X, y, predict_fn)
        assert len(importance) == 5
        assert np.sum(np.abs(importance)) > 0

    def test_attribution_analyzer(self, setup_attribution):
        """Test attribution analyzer."""
        X, y, predict_fn = setup_attribution
        X_df = pd.DataFrame(X)
        X_df.columns = [f"Feature_{i}" for i in range(5)]

        analyzer = AttributionAnalyzer()
        results = analyzer.analyze(X_df, y, predict_fn, methods=["ablation"])
        assert "ablation" in results


class TestEndToEndPipeline:
    """End-to-end pipeline tests."""

    @pytest.fixture
    def full_pipeline_data(self):
        """Create full pipeline data."""
        dates = pd.date_range(start="2023-01-01", periods=150, freq="D")
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "Open": np.random.uniform(100, 110, 150),
                "High": np.random.uniform(110, 120, 150),
                "Low": np.random.uniform(90, 100, 150),
                "Close": np.random.uniform(100, 110, 150),
                "Volume": np.random.uniform(1e6, 1e7, 150),
            },
            index=dates,
        )
        return data

    def test_full_pipeline_naive(self, full_pipeline_data):
        """Test full pipeline with naive forecaster."""
        # Clean data
        cleaner = DataCleaner()
        data = cleaner.clean_market_data(full_pipeline_data.copy())

        # Create features
        data = create_features(data)

        # Split
        split_idx = int(len(data) * 0.8)
        train = data.iloc[:split_idx]
        test = data.iloc[split_idx:]

        # Train and forecast
        model = NaiveForecaster()
        model.fit(train, "Close")
        pred = model.forecast(test, "Close", 20)

        # Evaluate
        actual = test["Close"].values[:20]
        evaluator = ForecastEvaluator()
        metrics = evaluator.evaluate(actual, pred)

        assert "mae" in metrics
        assert "rmse" in metrics

    def test_full_pipeline_chronos(self, full_pipeline_data):
        """Test full pipeline with Chronos."""
        # Clean data
        cleaner = DataCleaner()
        data = cleaner.clean_market_data(full_pipeline_data.copy())

        # Split
        split_idx = int(len(data) * 0.8)
        train = data.iloc[:split_idx]
        test = data.iloc[split_idx:]

        # Train and forecast
        model = ChronosFinancialForecaster(prediction_length=20, device="cpu")
        model.fit(train, "Close")
        forecasts = model.forecast_zero_shot(test, "Close", num_samples=10)

        # Verify output format
        assert "median" in forecasts
        assert "mean" in forecasts
        assert len(forecasts["median"]) == 20

    def test_full_pipeline_with_tokenization(self, full_pipeline_data):
        """Test pipeline with tokenization."""
        # Split
        split_idx = int(len(full_pipeline_data) * 0.8)
        train = full_pipeline_data.iloc[:split_idx]
        test = full_pipeline_data.iloc[split_idx:]

        # Tokenize
        tokenizer = FinancialDataTokenizer(method="quantile", num_bins=100)
        tokenizer.fit(train[["Close"]])
        tokens = tokenizer.transform(test[["Close"]])

        assert "Close" in tokens
        assert len(tokens["Close"]) > 0


class TestConfigurationManagement:
    """Test configuration management."""

    def test_data_config(self):
        """Test DataConfig."""
        config = DataConfig()
        assert hasattr(config, "market_symbols")
        assert hasattr(config, "start_date")
        assert hasattr(config, "end_date")

    def test_preprocessing_config(self):
        """Test PreprocessingConfig."""
        config = PreprocessingConfig()
        assert hasattr(config, "tokenization_method")
        assert hasattr(config, "num_bins")

    def test_eval_config(self):
        """Test EvalConfig."""
        config = EvalConfig()
        assert hasattr(config, "test_ratio")
        assert hasattr(config, "metrics")


class TestLogger:
    """Test logging setup."""

    def test_logger_creation(self):
        """Test logger creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            logger = setup_logger("test", str(log_file))
            logger.info("Test message")
            assert log_file.exists()


def test_smoke_imports():
    """Test that all modules can be imported."""
    from src.analysis import AttributionAnalyzer
    from src.data.cleaning import DataCleaner
    from src.data.fetchers import DataFetcher
    from src.eval.metrics import ForecastEvaluator
    from src.models.baselines import NaiveForecaster
    from src.models.chronos_wrapper import ChronosFinancialForecaster
    from src.preprocessing.tokenizer import FinancialDataTokenizer
    from src.utils.config import DataConfig
    from src.utils.logger import setup_logger

    assert AttributionAnalyzer is not None
    assert DataCleaner is not None
    assert DataFetcher is not None
    assert ForecastEvaluator is not None
    assert NaiveForecaster is not None
    assert ChronosFinancialForecaster is not None
    assert FinancialDataTokenizer is not None
    assert DataConfig is not None
    assert setup_logger is not None
