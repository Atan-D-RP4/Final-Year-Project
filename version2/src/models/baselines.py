"""Baseline models for time series forecasting."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseForecaster(ABC):
    """Abstract base class for forecasting models."""

    def __init__(self, prediction_length: int = 24) -> None:
        """
        Initialize the forecaster.

        Args:
            prediction_length: Number of steps to forecast
        """
        self.prediction_length = prediction_length
        self.is_fitted = False

    @abstractmethod
    def fit(self, data: pd.DataFrame, target_col: str) -> "BaseForecaster":
        """Fit the model to training data."""

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""

    def forecast(self, data: pd.DataFrame, target_col: str) -> np.ndarray:
        """Fit and predict in one step."""
        return self.fit(data, target_col).predict(data)


class NaiveForecaster(BaseForecaster):
    """Naive forecaster that repeats the last value."""

    def __init__(self, prediction_length: int = 24, method: str = "last") -> None:
        """
        Initialize naive forecaster.

        Args:
            prediction_length: Number of steps to forecast
            method: Naive method ('last', 'mean', 'seasonal')
        """
        super().__init__(prediction_length)
        self.method = method
        self.last_value: float | None = None
        self.mean_value: float | None = None
        self.seasonal_values: np.ndarray | None = None
        self.seasonal_period = 252  # Daily data, yearly seasonality

    def fit(self, data: pd.DataFrame, target_col: str) -> "NaiveForecaster":
        """Fit the naive model."""
        if target_col not in data.columns:
            raise ValueError(f"Target column {target_col} not found in data")

        series = data[target_col].dropna()

        if len(series) == 0:
            raise ValueError("No valid data to fit")

        self.last_value = float(series.iloc[-1])
        self.mean_value = float(series.mean())

        # For seasonal naive, store last seasonal_period values
        if len(series) >= self.seasonal_period:
            self.seasonal_values = series.iloc[-self.seasonal_period :].values
        else:
            self.seasonal_values = series.values

        self.is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Generate naive predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if self.method == "last":
            return np.full(self.prediction_length, self.last_value)
        if self.method == "mean":
            return np.full(self.prediction_length, self.mean_value)
        if self.method == "seasonal":
            # Repeat seasonal pattern
            if self.seasonal_values is not None:
                n_repeats = (self.prediction_length // len(self.seasonal_values)) + 1
                seasonal_forecast = np.tile(self.seasonal_values, n_repeats)
                return seasonal_forecast[: self.prediction_length]
            return np.full(self.prediction_length, self.last_value or 0.0)
        raise ValueError(f"Unknown naive method: {self.method}")


class ARIMAForecaster(BaseForecaster):
    """ARIMA forecaster for univariate time series."""

    def __init__(
        self,
        prediction_length: int = 24,
        order: tuple[int, int, int] = (1, 1, 1),
        auto_order: bool = True,
    ) -> None:
        """
        Initialize ARIMA forecaster.

        Args:
            prediction_length: Number of steps to forecast
            order: ARIMA order (p, d, q)
            auto_order: Whether to automatically select order
        """
        super().__init__(prediction_length)
        self.order = order
        self.auto_order = auto_order
        self.model: ARIMA | None = None
        self.fitted_model = None

    def fit(self, data: pd.DataFrame, target_col: str) -> "ARIMAForecaster":
        """Fit ARIMA model."""
        if target_col not in data.columns:
            raise ValueError(f"Target column {target_col} not found in data")

        series = data[target_col].dropna()

        if len(series) < 10:
            raise ValueError("Need at least 10 observations for ARIMA")

        try:
            if self.auto_order:
                # Simple auto-selection (could be improved with AIC/BIC search)
                self.model = ARIMA(series, order=(2, 1, 2))
            else:
                self.model = ARIMA(series, order=self.order)

            self.fitted_model = self.model.fit()
            self.is_fitted = True

        except Exception as e:
            logger.error(f"Error fitting ARIMA model: {e}")
            # Fallback to simpler model
            try:
                self.model = ARIMA(series, order=(1, 1, 1))
                self.fitted_model = self.model.fit()
                self.is_fitted = True
            except Exception as e2:
                logger.error(f"Error fitting fallback ARIMA model: {e2}")
                raise e2

        return self

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Generate ARIMA predictions."""
        if not self.is_fitted or self.fitted_model is None:
            raise ValueError("Model must be fitted before prediction")

        try:
            forecast = self.fitted_model.forecast(steps=self.prediction_length)
            return forecast.values if hasattr(forecast, "values") else forecast
        except Exception as e:
            logger.error(f"Error generating ARIMA predictions: {e}")
            # Return naive forecast as fallback
            return np.full(self.prediction_length, 0.0)


class VARForecaster(BaseForecaster):
    """Vector Autoregression forecaster for multivariate time series."""

    def __init__(
        self, prediction_length: int = 24, maxlags: int = 5, ic: str = "aic"
    ) -> None:
        """
        Initialize VAR forecaster.

        Args:
            prediction_length: Number of steps to forecast
            maxlags: Maximum number of lags to consider
            ic: Information criterion for lag selection
        """
        super().__init__(prediction_length)
        self.maxlags = maxlags
        self.ic = ic
        self.model: VAR | None = None
        self.fitted_model = None
        self.feature_cols: list[str] = []
        self.target_col: str = ""

    def fit(self, data: pd.DataFrame, target_col: str) -> "VARForecaster":
        """Fit VAR model."""
        if target_col not in data.columns:
            raise ValueError(f"Target column {target_col} not found in data")

        # Select numeric columns for VAR
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            raise ValueError("VAR requires at least 2 variables")

        # Ensure target is first column
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        self.feature_cols = [target_col] + numeric_cols[
            : min(5, len(numeric_cols))
        ]  # Limit to 6 variables
        self.target_col = target_col

        var_data = data[self.feature_cols].dropna()

        if len(var_data) < 20:
            raise ValueError("Need at least 20 observations for VAR")

        try:
            self.model = VAR(var_data)
            self.fitted_model = self.model.fit(maxlags=self.maxlags, ic=self.ic)
            self.is_fitted = True

        except Exception as e:
            logger.error(f"Error fitting VAR model: {e}")
            # Fallback to simple VAR with fewer lags
            try:
                if self.model is not None:
                    self.fitted_model = self.model.fit(maxlags=2)
                    self.is_fitted = True
                else:
                    raise ValueError("VAR model is None")
            except Exception as e2:
                logger.error(f"Error fitting fallback VAR model: {e2}")
                raise e2

        return self

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Generate VAR predictions."""
        if not self.is_fitted or self.fitted_model is None:
            raise ValueError("Model must be fitted before prediction")

        try:
            # Get last observations for forecasting
            var_data = data[self.feature_cols].dropna()

            if len(var_data) == 0:
                raise ValueError("No valid data for prediction")

            # Forecast
            forecast = self.fitted_model.forecast(
                var_data.values[-self.fitted_model.k_ar :], steps=self.prediction_length
            )

            # Return predictions for target variable (first column)
            return forecast[:, 0]

        except Exception as e:
            logger.error(f"Error generating VAR predictions: {e}")
            # Return naive forecast as fallback
            last_value = data[self.target_col].dropna().iloc[-1]
            return np.full(self.prediction_length, last_value)


class LSTMForecaster(BaseForecaster):
    """LSTM forecaster for time series."""

    def __init__(
        self,
        prediction_length: int = 24,
        sequence_length: int = 60,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
    ) -> None:
        """
        Initialize LSTM forecaster.

        Args:
            prediction_length: Number of steps to forecast
            sequence_length: Length of input sequences
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            learning_rate: Learning rate
            epochs: Training epochs
            batch_size: Batch size
        """
        super().__init__(prediction_length)
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        self.model: nn.Module | None = None
        self.scaler: StandardScaler | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _create_model(self, input_size: int) -> nn.Module:
        """Create LSTM model."""

        class LSTMModel(nn.Module):  # type: ignore[misc]
            def __init__(
                self,
                input_size: int,
                hidden_size: int,
                num_layers: int,
                output_size: int,
                dropout: float,
            ):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size,
                    hidden_size,
                    num_layers,
                    batch_first=True,
                    dropout=dropout,
                )
                self.fc = nn.Linear(hidden_size, output_size)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                lstm_out, _ = self.lstm(x)
                output = self.fc(lstm_out[:, -1, :])  # Use last timestep
                return output

        return LSTMModel(
            input_size,
            self.hidden_size,
            self.num_layers,
            self.prediction_length,
            self.dropout,
        )

    def _prepare_data(
        self, data: pd.DataFrame, target_col: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM training."""
        # Select numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        # Ensure target is included
        if target_col not in numeric_cols:
            raise ValueError(f"Target column {target_col} is not numeric")

        # Use target + up to 9 other features (limit for computational efficiency)
        feature_cols = [target_col]
        other_cols = [col for col in numeric_cols if col != target_col]
        feature_cols.extend(other_cols[:9])

        data_array = data[feature_cols].dropna().to_numpy()

        if len(data_array) < self.sequence_length + self.prediction_length:
            raise ValueError("Not enough data for LSTM training")

        # Normalize data

        self.scaler = StandardScaler()
        data_scaled = self.scaler.fit_transform(data_array)

        # Create sequences
        X, y = [], []
        target_idx = feature_cols.index(target_col)

        for i in range(
            len(data_scaled) - self.sequence_length - self.prediction_length + 1
        ):
            X.append(data_scaled[i : i + self.sequence_length])
            y.append(
                data_scaled[
                    i + self.sequence_length : i
                    + self.sequence_length
                    + self.prediction_length,
                    target_idx,
                ]
            )

        return np.array(X), np.array(y)

    def fit(self, data: pd.DataFrame, target_col: str) -> "LSTMForecaster":
        """Fit LSTM model."""
        logger.info("Fitting LSTM model")

        try:
            # Prepare data
            X, y = self._prepare_data(data, target_col)

            # Create model
            input_size = X.shape[2]
            self.model = self._create_model(input_size).to(self.device)

            # Create data loader
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.FloatTensor(y).to(self.device)
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            # Training
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            criterion = nn.MSELoss()

            self.model.train()
            for epoch in range(self.epochs):
                total_loss = 0
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                if (epoch + 1) % 10 == 0:
                    avg_loss = total_loss / len(dataloader)
                    logger.debug(
                        f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}"
                    )

            self.is_fitted = True
            logger.info("LSTM model fitted successfully")

        except Exception as e:
            logger.error(f"Error fitting LSTM model: {e}")
            raise

        return self

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Generate LSTM predictions."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")

        try:
            # Get the same features used in training
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            data_array = data[numeric_cols].dropna().values

            if len(data_array) < self.sequence_length:
                raise ValueError("Not enough data for prediction")

            # Use last sequence_length observations
            last_sequence = data_array[-self.sequence_length :]

            # Scale using fitted scaler
            if self.scaler is not None:
                last_sequence_scaled = self.scaler.transform(last_sequence)
            else:
                raise ValueError("Scaler not fitted")

            # Predict
            self.model.eval()
            with torch.no_grad():
                X_pred = (
                    torch.FloatTensor(last_sequence_scaled).unsqueeze(0).to(self.device)
                )
                prediction_scaled = self.model(X_pred).cpu().numpy().flatten()

            # Inverse transform (approximate)
            # Create dummy array for inverse transform
            if self.scaler is not None:
                dummy_array = np.zeros(
                    (self.prediction_length, self.scaler.n_features_in_)
                )
                dummy_array[:, 0] = prediction_scaled  # Target is first column
                prediction_original = self.scaler.inverse_transform(dummy_array)[:, 0]
            else:
                prediction_original = prediction_scaled

            return prediction_original

        except Exception as e:
            logger.error(f"Error generating LSTM predictions: {e}")
            # Return naive forecast as fallback
            last_value = float(data.iloc[-1, 0]) if len(data) > 0 else 0.0
            return np.full(self.prediction_length, last_value)


class LinearForecaster(BaseForecaster):
    """Simple linear regression forecaster."""

    def __init__(
        self,
        prediction_length: int = 24,
        lookback_window: int = 60,
        include_trend: bool = True,
    ) -> None:
        """
        Initialize linear forecaster.

        Args:
            prediction_length: Number of steps to forecast
            lookback_window: Number of past observations to use as features
            include_trend: Whether to include time trend
        """
        super().__init__(prediction_length)
        self.lookback_window = lookback_window
        self.include_trend = include_trend
        self.model = LinearRegression()
        self.feature_cols: list[str] = []
        self.target_col: str = ""

    def _create_features(
        self, data: pd.DataFrame, target_col: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create features for linear regression."""
        # Select numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        if target_col not in numeric_cols:
            raise ValueError(f"Target column {target_col} is not numeric")

        # Use target + other features
        self.feature_cols = numeric_cols
        self.target_col = target_col

        data_clean = data[self.feature_cols].dropna()

        if len(data_clean) < self.lookback_window + self.prediction_length:
            raise ValueError("Not enough data for linear forecaster")

        X, y = [], []

        for i in range(
            len(data_clean) - self.lookback_window - self.prediction_length + 1
        ):
            # Features: lagged values of all variables
            features = []

            # Add lagged values
            for col in self.feature_cols:
                features.extend(
                    data_clean[col].iloc[i : i + self.lookback_window].values
                )

            # Add time trend if requested
            if self.include_trend:
                features.append(i)

            X.append(features)

            # Target: next prediction_length values of target
            target_values = (
                data_clean[target_col]
                .iloc[
                    i + self.lookback_window : i
                    + self.lookback_window
                    + self.prediction_length
                ]
                .values
            )
            y.append(target_values)

        return np.array(X), np.array(y)

    def fit(self, data: pd.DataFrame, target_col: str) -> "LinearForecaster":
        """Fit linear model."""
        logger.info("Fitting linear forecaster")

        try:
            X, y = self._create_features(data, target_col)

            # For multi-output regression, we'll predict each step separately
            # and store multiple models
            self.models = []

            for step in range(self.prediction_length):
                model = LinearRegression()
                model.fit(X, y[:, step])
                self.models.append(model)

            self.is_fitted = True
            logger.info("Linear forecaster fitted successfully")

        except Exception as e:
            logger.error(f"Error fitting linear forecaster: {e}")
            raise

        return self

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Generate linear predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        try:
            data_clean = data[self.feature_cols].dropna()

            if len(data_clean) < self.lookback_window:
                raise ValueError("Not enough data for prediction")

            # Create features from last lookback_window observations
            features = []

            # Add lagged values
            for col in self.feature_cols:
                features.extend(data_clean[col].iloc[-self.lookback_window :].values)

            # Add time trend if requested
            if self.include_trend:
                features.append(len(data_clean))

            X_pred = np.array(features).reshape(1, -1)

            # Predict each step
            predictions = []
            for model in self.models:
                pred = model.predict(X_pred)[0]
                predictions.append(pred)

            return np.array(predictions)

        except Exception as e:
            logger.error(f"Error generating linear predictions: {e}")
            # Return naive forecast as fallback
            last_value = (
                data[self.target_col].dropna().iloc[-1] if len(data) > 0 else 0.0
            )
            return np.full(self.prediction_length, last_value)
