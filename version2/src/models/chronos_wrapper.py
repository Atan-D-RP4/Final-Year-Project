"""Chronos model wrapper for financial forecasting."""


import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from src.preprocessing.tokenizer import FinancialDataTokenizer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ChronosFinancialForecaster:
    """Wrapper for Chronos model with financial data tokenization."""

    def __init__(
        self,
        model_name: str = "amazon/chronos-t5-small",
        prediction_length: int = 24,
        context_length: int = 512,
        device: str | None = None,
    ) -> None:
        """
        Initialize Chronos forecaster.

        Args:
            model_name: Chronos model name/path
            prediction_length: Number of steps to forecast
            context_length: Context length for the model
            device: Device to run the model on
        """
        self.model_name = model_name
        self.prediction_length = prediction_length
        self.context_length = context_length

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Model components
        self.model: PreTrainedModel | None = None
        self.tokenizer: AutoTokenizer | None = None
        self.data_tokenizer: FinancialDataTokenizer | None = None

        # State
        self.is_loaded = False
        self.is_fitted = False

        # Try to load the model
        self._load_model()

    def _load_model(self) -> None:
        """Load the Chronos model and tokenizer."""
        try:
            logger.info(f"Loading Chronos model: {self.model_name}")

            # Try to load from Hugging Face
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=torch.float32, device_map=self.device
            )
            if self.model is None or self.tokenizer is None:
                raise ValueError("Failed to load Chronos model or tokenizer")

            if hasattr(self.model, "eval"):
                self.model.eval()
            self.is_loaded = True
            logger.info("Chronos model loaded successfully")

        except Exception as e:
            logger.warning(f"Could not load Chronos model: {e}")
            logger.info("Will use mock Chronos implementation for demonstration")
            self.is_loaded = False

    def fit(
        self,
        data: pd.DataFrame,
        target_col: str,
        tokenizer_config: dict | None = None,
    ) -> "ChronosFinancialForecaster":
        """
        Fit the forecaster (mainly the data tokenizer).

        Args:
            data: Training data
            target_col: Target column name
            tokenizer_config: Configuration for data tokenizer

        Returns:
            Self for method chaining
        """
        logger.info("Fitting Chronos financial forecaster")

        if tokenizer_config is None:
            tokenizer_config = {
                "num_bins": 1024,
                "method": "quantile",
                "context_length": self.context_length,
            }

        # Initialize and fit data tokenizer
        self.data_tokenizer = FinancialDataTokenizer(**tokenizer_config)
        self.data_tokenizer.fit(data)

        self.target_col = target_col
        self.is_fitted = True

        logger.info("Chronos financial forecaster fitted")
        return self

    def forecast_zero_shot(self, data: pd.DataFrame) -> np.ndarray:
        """
        Generate zero-shot forecasts using pretrained Chronos.

        Args:
            data: Input data for forecasting

        Returns:
            Forecast array
        """
        if not self.is_fitted:
            raise ValueError("Forecaster must be fitted before prediction")

        logger.info("Generating zero-shot forecast")

        if self.is_loaded and self.model is not None:
            return self._chronos_forecast(data)
        # Use mock implementation
        return self._mock_chronos_forecast(data)

    def _chronos_forecast(self, data: pd.DataFrame) -> np.ndarray:
        """Generate forecast using actual Chronos model."""
        try:
            if (
                self.data_tokenizer is None
                or self.model is None
                or self.tokenizer is None
            ):
                raise ValueError("Model or tokenizer not properly initialized")
            # Tokenize the data
            tokenized = self.data_tokenizer.transform(data)
            combined_tokens = tokenized["combined"]

            # Prepare input for Chronos
            # Note: This is a simplified implementation
            # Real Chronos integration would require proper token handling

            if isinstance(combined_tokens, np.ndarray):
                input_ids = torch.tensor(combined_tokens[-self.context_length:]).unsqueeze(0)
                input_ids = input_ids.to(self.device)
            else:
                raise ValueError("Combined tokens should be numpy array")

            # Generate predictions
            with torch.no_grad():
                if hasattr(self.model, "generate"):
                    eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
                    outputs = self.model.generate(
                        input_ids,
                        max_new_tokens=self.prediction_length,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=eos_token_id,
                    )
                else:
                    raise AttributeError("Model does not have generate method")

            # Extract generated tokens
            generated_tokens = outputs[0, len(input_ids[0]):].cpu().numpy()

            # Convert tokens back to values
            # This is simplified - real implementation would need proper decoding
            predictions = self.data_tokenizer.inverse_transform(
                generated_tokens[: self.prediction_length], self.target_col
            )

            return predictions

        except Exception as e:
            logger.error(f"Error in Chronos forecast: {e}")
            return self._mock_chronos_forecast(data)

    def _mock_chronos_forecast(self, data: pd.DataFrame) -> np.ndarray:
        """
        Mock Chronos forecast for demonstration purposes.

        This implements a simple pattern-based forecast that mimics
        what a foundation model might produce.
        """
        logger.info("Using mock Chronos implementation")

        if self.target_col not in data.columns:
            raise ValueError(f"Target column {self.target_col} not found")

        # Get recent values
        target_series = data[self.target_col].dropna()

        if len(target_series) == 0:
            return np.zeros(self.prediction_length)

        # Simple pattern-based forecast
        recent_values = target_series.iloc[-min(50, len(target_series)) :].values

        # Calculate trend and seasonality components
        if len(recent_values) > 1:
            # Linear trend
            x = np.arange(len(recent_values))
            trend_coef = np.polyfit(x, recent_values, 1)[0]
        else:
            trend_coef = 0

        # Seasonal pattern (weekly for daily data)
        seasonal_period = min(7, len(recent_values))
        if seasonal_period > 1:
            seasonal_pattern = recent_values[-seasonal_period:]
        else:
            seasonal_pattern = recent_values

        # Generate forecast
        forecast = []
        last_value = recent_values[-1]

        for i in range(self.prediction_length):
            # Trend component
            trend_component = trend_coef * (i + 1) * 0.5  # Damped trend

            # Seasonal component
            seasonal_idx = i % len(seasonal_pattern)
            seasonal_component = (
                seasonal_pattern[seasonal_idx] - np.mean(seasonal_pattern)
            ) * 0.3

            # Random component (small)
            np.random.seed(42 + i)  # For reproducibility
            random_component = np.random.normal(0, np.std(recent_values) * 0.1)

            # Combine components
            pred_value = (
                last_value + trend_component + seasonal_component + random_component
            )
            forecast.append(pred_value)

        return np.array(forecast)

    def fine_tune(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        adapter_config: dict | None = None,
    ) -> "ChronosFinancialForecaster":
        """
        Fine-tune Chronos on financial data.

        Args:
            train_data: Training data
            val_data: Validation data
            adapter_config: Configuration for fine-tuning

        Returns:
            Self for method chaining
        """
        logger.info("Fine-tuning Chronos model")

        if not self.is_loaded:
            logger.warning("Cannot fine-tune: Chronos model not loaded")
            return self

        # This would implement actual fine-tuning logic
        # For now, we'll just log that fine-tuning would happen here
        logger.info("Fine-tuning not implemented in this demo version")

        return self

    def evaluate(
        self, test_data: pd.DataFrame, metrics: list[str] | None = None
    ) -> dict[str, float]:
        """
        Evaluate the forecaster on test data.

        Args:
            test_data: Test data
            metrics: List of metrics to compute

        Returns:
            Dictionary of metric values
        """
        if not self.is_fitted:
            raise ValueError("Forecaster must be fitted before evaluation")

        # Generate predictions
        predictions = self.forecast_zero_shot(test_data)

        # Get true values (last prediction_length values)
        true_values = (
            test_data[self.target_col].dropna().iloc[-self.prediction_length :].values
        )

        # Calculate metrics
        from ..eval.metrics import calculate_all_metrics

        if len(true_values) != len(predictions):
            min_len = min(len(true_values), len(predictions))
            true_values = true_values[:min_len]
            predictions = predictions[:min_len]

        results = calculate_all_metrics(true_values, predictions)

        return results


def check_chronos_availability() -> bool:
    """Check if Chronos models are available."""
    try:
        from transformers import AutoTokenizer

        # Try to load a small Chronos model
        AutoTokenizer.from_pretrained("amazon/chronos-t5-small")
        return True
    except Exception:
        return False


def get_available_chronos_models() -> list[str]:
    """Get list of available Chronos models."""
    models = [
        "amazon/chronos-t5-small",
        "amazon/chronos-t5-base",
        "amazon/chronos-t5-large",
    ]

    available = []
    for model in models:
        try:
            from transformers import AutoTokenizer

            AutoTokenizer.from_pretrained(model)
            available.append(model)
        except Exception:
            continue

    return available
