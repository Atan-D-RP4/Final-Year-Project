"""Baseline forecasting models for financial time series."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.seasonal import seasonal_decompose

import torch
import torch.nn as nn


def _extract_target_column(data: pd.DataFrame, target_col: str) -> np.ndarray:
    """Extract target column properly handling MultiIndex columns."""
    if isinstance(data.columns, pd.MultiIndex):
        # Try (Ticker, Price) format first
        if (target_col, 'Close') in data.columns:
            return data[(target_col, 'Close')].values
        elif (target_col, 'Adj Close') in data.columns:
            return data[(target_col, 'Adj Close')].values
        
        # Try (Price, Ticker) format (yfinance default)
        elif ('Close', target_col) in data.columns:
            return data[('Close', target_col)].values
        elif ('Adj Close', target_col) in data.columns:
            return data[('Adj Close', target_col)].values
        
        else:
            # Find any column with the target symbol
            for col in data.columns:
                if target_col in str(col) and 'Close' in str(col):
                    return data[col].values
            raise ValueError(f"Could not find Close price column for {target_col}")
    else:
        return data[target_col].values


class BaselineForecaster(ABC):
    """Abstract base class for baseline forecasters."""

    @abstractmethod
    def fit(
        self,
        train_data: pd.DataFrame,
        target_col: str,
    ) -> None:
        """Fit model on training data.

        Args:
            train_data: Training DataFrame
            target_col: Target column name
        """

    @abstractmethod
    def forecast(
        self,
        test_data: pd.DataFrame,
        target_col: str,
        horizon: int,
    ) -> np.ndarray:
        """Generate forecasts.

        Args:
            test_data: Test DataFrame
            target_col: Target column
            horizon: Forecast horizon

        Returns:
            Forecast array of shape (horizon,)
        """


class NaiveForecaster(BaselineForecaster):
    """Naive forecast: repeat last value."""

    def fit(
        self,
        train_data: pd.DataFrame,
        target_col: str,
    ) -> None:
        """Fit is a no-op for naive forecaster.

        Args:
            train_data: Training DataFrame
            target_col: Target column name
        """

    def forecast(
        self,
        test_data: pd.DataFrame,
        target_col: str,
        horizon: int,
    ) -> np.ndarray:
        """Repeat last value.

        Args:
            test_data: Test DataFrame
            target_col: Target column
            horizon: Forecast horizon

        Returns:
            Repeated last value
        """
        target_values = _extract_target_column(test_data, target_col)
        last_value = target_values[-1]
        return np.full(horizon, last_value)


class SeasonalNaiveForecaster(BaselineForecaster):
    """Seasonal naive forecast: repeat value from same season."""

    def __init__(self, seasonal_period: int = 252):
        """Initialize with seasonal period.

        Args:
            seasonal_period: Number of periods in a season (default: 252 for
            daily trading days)
        """
        self.seasonal_period = seasonal_period
        self.seasonal_values = None

    def fit(
        self,
        train_data: pd.DataFrame,
        target_col: str,
    ) -> None:
        """Store seasonal values.

        Args:
            train_data: Training DataFrame
            target_col: Target column name
        """
        target_values = _extract_target_column(train_data, target_col)
        self.seasonal_values = target_values[
            -self.seasonal_period :
        ]

    def forecast(
        self,
        test_data: pd.DataFrame,
        target_col: str,
        horizon: int,
    ) -> np.ndarray:
        """Repeat seasonal pattern.

        Args:
            test_data: Test DataFrame
            target_col: Target column
            horizon: Forecast horizon

        Returns:
            Forecast following seasonal pattern
        """
        if self.seasonal_values is None:
            return np.zeros(horizon)

        forecasts = []
        for i in range(horizon):
            idx = i % len(self.seasonal_values)
            forecasts.append(self.seasonal_values[idx])

        return np.array(forecasts)


class MeanForecaster(BaselineForecaster):
    """Constant mean forecast."""

    def __init__(self):
        """Initialize mean forecaster."""
        self.mean = None

    def fit(
        self,
        train_data: pd.DataFrame,
        target_col: str,
    ) -> None:
        """Compute mean.

        Args:
            train_data: Training DataFrame
            target_col: Target column name
        """
        self.mean = train_data[target_col].mean()

    def forecast(
        self,
        test_data: pd.DataFrame,
        target_col: str,
        horizon: int,
    ) -> np.ndarray:
        """Return constant mean.

        Args:
            test_data: Test DataFrame
            target_col: Target column
            horizon: Forecast horizon

        Returns:
            Constant mean values
        """
        if self.mean is None:
            return np.zeros(horizon)
        return np.full(horizon, self.mean)


class ARIMAForecaster(BaselineForecaster):
    """ARIMA model for forecasting."""

    def __init__(self, order: tuple = (1, 1, 1)):
        """Initialize ARIMA.

        Args:
            order: ARIMA order (p, d, q)
        """
        self.order = order
        self.model = None

    def fit(
        self,
        train_data: pd.DataFrame,
        target_col: str,
    ) -> None:
        """Fit ARIMA model.

        Args:
            train_data: Training DataFrame
            target_col: Target column name
        """
        try:
            target_values = _extract_target_column(train_data, target_col)
            self.model = ARIMA(
                target_values, order=self.order
            ).fit()
        except Exception as e:
            print(f"Warning: ARIMA fit failed: {e}")
            self.model = None

    def forecast(
        self,
        test_data: pd.DataFrame,
        target_col: str,
        horizon: int,
    ) -> np.ndarray:
        """Forecast with ARIMA.

        Args:
            test_data: Test DataFrame
            target_col: Target column
            horizon: Forecast horizon

        Returns:
            ARIMA forecast
        """
        if self.model is None:
            return np.zeros(horizon)

        try:
            result = self.model.get_forecast(steps=horizon)
            return result.predicted_mean.values
        except Exception as e:
            print(f"Warning: ARIMA forecast failed: {e}")
            return np.zeros(horizon)


class VARForecaster(BaselineForecaster):
    """Vector Autoregression model."""

    def __init__(self, lags: int = 2):
        """Initialize VAR.

        Args:
            lags: Number of lags to use
        """
        self.lags = lags
        self.model = None

    def fit(
        self,
        train_data: pd.DataFrame,
        target_col: str,
    ) -> None:
        """Fit VAR model.

        Args:
            train_data: Training DataFrame
            target_col: Target column name
        """
        try:
            # Use all numeric columns for multivariate forecasting
            numeric_cols = train_data.select_dtypes(
                include=[np.number]
            ).columns.tolist()
            if not numeric_cols:
                self.model = None
                return

            self.model = VARMAX(
                train_data[numeric_cols], order=(self.lags, 0)
            ).fit(disp=False)
        except Exception as e:
            print(f"Warning: VAR fit failed: {e}")
            self.model = None

    def forecast(
        self,
        test_data: pd.DataFrame,
        target_col: str,
        horizon: int,
    ) -> np.ndarray:
        """Forecast with VAR.

        Args:
            test_data: Test DataFrame
            target_col: Target column
            horizon: Forecast horizon

        Returns:
            VAR forecast
        """
        if self.model is None:
            return np.zeros(horizon)

        try:
            result = self.model.get_forecast(steps=horizon)
            forecast_data = result.predicted_mean

            # Extract target column
            if target_col in forecast_data.columns:
                return forecast_data[target_col].values
            else:
                return forecast_data.iloc[:, 0].values
        except Exception as e:
            print(f"Warning: VAR forecast failed: {e}")
            return np.zeros(horizon)


class LinearRegressionForecaster(BaselineForecaster):
    """Linear regression with lagged features."""

    def __init__(self, lags: int = 20, horizon: int = 1):
        """Initialize linear regression forecaster.

        Args:
            lags: Number of lags to use as features
            horizon: Forecast horizon
        """
        self.lags = lags
        self.horizon = horizon
        self.model = None
        self.scaler = StandardScaler()

    def fit(
        self,
        train_data: pd.DataFrame,
        target_col: str,
    ) -> None:
        """Fit linear regression model.

        Args:
            train_data: Training DataFrame
            target_col: Target column name
        """
        target = _extract_target_column(train_data, target_col)

        # Create lagged features
        X = []
        y = []

        for i in range(self.lags, len(target) - self.horizon):
            X.append(target[i - self.lags : i])
            y.append(target[i + self.horizon])

        if len(X) == 0:
            self.model = None
            return

        X = np.array(X)
        y = np.array(y)

        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        # Fit model
        self.model = LinearRegression()
        self.model.fit(X_scaled, y)

    def forecast(
        self,
        test_data: pd.DataFrame,
        target_col: str,
        horizon: int,
    ) -> np.ndarray:
        """Forecast with linear regression.

        Args:
            test_data: Test DataFrame
            target_col: Target column
            horizon: Forecast horizon

        Returns:
            Linear regression forecast
        """
        if self.model is None:
            return np.zeros(horizon)

        target = _extract_target_column(test_data, target_col)
        forecasts = []

        for h in range(horizon):
            if len(target) < self.lags:
                forecasts.append(target[-1])
                continue

            # Use last lags values
            X_test = target[-self.lags :].reshape(1, -1)
            X_test_scaled = self.scaler.transform(X_test)

            # Predict
            pred = self.model.predict(X_test_scaled)[0]
            forecasts.append(pred)

            # Update target for next step
            target = np.append(target, pred)

        return np.array(forecasts)


class LSTMForecaster(BaselineForecaster):
    """LSTM neural network forecaster."""

    def __init__(
        self,
        sequence_length: int = 20,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        device: str = "cpu",
    ):
        """Initialize LSTM forecaster.

        Args:
            sequence_length: Length of input sequences
            hidden_size: Size of hidden layer
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            device: Device to use (cpu, cuda, mps)
        """
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device
        self.model = None
        self.scaler = StandardScaler()

    def _build_model(self, input_size: int = 1) -> nn.Module:
        """Build LSTM model.

        Args:
            input_size: Size of input features

        Returns:
            LSTM model
        """
        class LSTMModel(nn.Module):
            """LSTM model for time series forecasting."""

            def __init__(
                self,
                input_size: int,
                hidden_size: int,
                num_layers: int,
                dropout: float,
            ):
                """Initialize LSTM model."""
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size,
                    hidden_size,
                    num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0,
                )
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x):
                """Forward pass."""
                lstm_out, _ = self.lstm(x)
                last_out = lstm_out[:, -1, :]
                pred = self.fc(last_out)
                return pred

        return LSTMModel(
            input_size, self.hidden_size, self.num_layers, self.dropout
        )

    def fit(
        self,
        train_data: pd.DataFrame,
        target_col: str,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
    ) -> None:
        """Fit LSTM model.

        Args:
            train_data: Training DataFrame
            target_col: Target column name
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        target = _extract_target_column(train_data, target_col).reshape(-1, 1)

        # Standardize
        target_scaled = self.scaler.fit_transform(target)

        # Create sequences
        sequences = []
        targets = []

        for i in range(len(target_scaled) - self.sequence_length):
            sequences.append(target_scaled[i : i + self.sequence_length])
            targets.append(target_scaled[i + self.sequence_length, 0])

        if len(sequences) == 0:
            print("Warning: Not enough data for LSTM training")
            return

        X_train = torch.FloatTensor(np.array(sequences))
        y_train = torch.FloatTensor(np.array(targets)).reshape(-1, 1)

        # Build model
        self.model = self._build_model(input_size=1)
        self.model.to(self.device)

        # Training
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate
        )
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            epoch_loss = 0.0
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i : i + batch_size].to(self.device)
                batch_y = y_train[i : i + batch_size].to(self.device)

                # Forward
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs}, "
                    f"Loss: {epoch_loss / len(X_train):.4f}"
                )

    def forecast(
        self,
        test_data: pd.DataFrame,
        target_col: str,
        horizon: int,
    ) -> np.ndarray:
        """Forecast with LSTM.

        Args:
            test_data: Test DataFrame
            target_col: Target column
            horizon: Forecast horizon

        Returns:
            LSTM forecast
        """
        if self.model is None:
            return np.zeros(horizon)

        target = _extract_target_column(test_data, target_col).reshape(-1, 1)
        target_scaled = self.scaler.transform(target)

        self.model.eval()
        forecasts = []

        with torch.no_grad():
            sequence = target_scaled[-self.sequence_length :]

            for _ in range(horizon):
                X = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
                pred = self.model(X).item()
                forecasts.append(pred)

                # Update sequence
                sequence = np.vstack([sequence[1:], [[pred]]])

        # Inverse scale
        forecasts_array = np.array(forecasts).reshape(-1, 1)
        forecasts_original = self.scaler.inverse_transform(
            forecasts_array
        ).flatten()

        return forecasts_original


class ExponentialSmoothingForecaster(BaselineForecaster):
    """Exponential smoothing forecaster."""

    def __init__(self, smoothing_level: float = 0.3):
        """Initialize exponential smoothing.

        Args:
            smoothing_level: Smoothing parameter (0 to 1)
        """
        self.smoothing_level = smoothing_level
        self.last_level = None

    def fit(
        self,
        train_data: pd.DataFrame,
        target_col: str,
    ) -> None:
        """Fit exponential smoothing.

        Args:
            train_data: Training DataFrame
            target_col: Target column name
        """
        target = _extract_target_column(train_data, target_col)
        level = target[0]

        for val in target[1:]:
            level = (
                self.smoothing_level * val
                + (1 - self.smoothing_level) * level
            )

        self.last_level = level

    def forecast(
        self,
        test_data: pd.DataFrame,
        target_col: str,
        horizon: int,
    ) -> np.ndarray:
        """Forecast with exponential smoothing.

        Args:
            test_data: Test DataFrame
            target_col: Target column
            horizon: Forecast horizon

        Returns:
            Exponential smoothing forecast
        """
        if self.last_level is None:
            return np.zeros(horizon)

        return np.full(horizon, self.last_level)


class EnsembleForecaster(BaselineForecaster):
    """Ensemble of baseline forecasters."""

    def __init__(self, forecasters: Optional[list] = None):
        """Initialize ensemble.

        Args:
            forecasters: List of forecaster instances
        """
        if forecasters is None:
            forecasters = [
                NaiveForecaster(),
                MeanForecaster(),
                ExponentialSmoothingForecaster(),
            ]
        self.forecasters = forecasters

    def fit(
        self,
        train_data: pd.DataFrame,
        target_col: str,
    ) -> None:
        """Fit all forecasters.

        Args:
            train_data: Training DataFrame
            target_col: Target column name
        """
        for forecaster in self.forecasters:
            try:
                forecaster.fit(train_data, target_col)
            except Exception as e:
                print(f"Warning: Forecaster fit failed: {e}")

    def forecast(
        self,
        test_data: pd.DataFrame,
        target_col: str,
        horizon: int,
    ) -> np.ndarray:
        """Generate ensemble forecast (average).

        Args:
            test_data: Test DataFrame
            target_col: Target column
            horizon: Forecast horizon

        Returns:
            Average forecast from all models
        """
        forecasts = []

        for forecaster in self.forecasters:
            try:
                pred = forecaster.forecast(test_data, target_col, horizon)
                if pred is not None and len(pred) == horizon:
                    forecasts.append(pred)
            except Exception as e:
                print(f"Warning: Forecaster forecast failed: {e}")

        if not forecasts:
            return np.zeros(horizon)

        return np.mean(np.array(forecasts), axis=0)
