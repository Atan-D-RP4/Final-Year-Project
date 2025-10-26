"""TCN Forecaster for time series forecasting."""

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.baselines import BaseForecaster
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network (TCN)."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_channels: list[int],
        kernel_size: int,
        dropout: float,
    ):
        """
        Initialize TCN.

        Args:
            input_size: Number of input features
            output_size: Number of output features
            num_channels: List of channel sizes for each layer
            kernel_size: Kernel size for convolution
            dropout: Dropout rate
        """
        super().__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    padding=(kernel_size - 1),
                    dilation=2**i,
                ),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.network(x.transpose(1, 2))  # Transpose for Conv1d
        return torch.Tensor(self.fc(x[:, :, -1]))  # Use last timestep


class TCNForecaster(BaseForecaster):
    """TCN forecaster for time series forecasting."""

    def __init__(
        self,
        prediction_length: int = 24,
        sequence_length: int = 60,
        num_channels: list[int] = [32, 32, 32],
        kernel_size: int = 3,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
    ) -> None:
        """
        Initialize TCN forecaster.

        Args:
            prediction_length: Number of steps to forecast
            sequence_length: Length of input sequences
            num_channels: List of channel sizes for the TCN
            kernel_size: Kernel size for TCN layers
            dropout: Dropout rate
            learning_rate: Learning rate
            epochs: Number of training epochs
            batch_size: Batch size
        """
        super().__init__(prediction_length)
        self.sequence_length = sequence_length
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        self.model: nn.Module | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _create_model(self, input_size: int) -> nn.Module:
        """Create TCN model."""
        return TemporalConvNet(
            input_size,
            self.prediction_length,
            self.num_channels,
            self.kernel_size,
            self.dropout,
        ).to(self.device)

    def _prepare_data(
        self, data: np.ndarray, target_idx: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prepare data for TCN training."""
        x, y = [], []

        for i in range(len(data) - self.sequence_length - self.prediction_length + 1):
            x.append(data[i : i + self.sequence_length])
            y.append(
                data[
                    i + self.sequence_length : i
                    + self.sequence_length
                    + self.prediction_length,
                    target_idx,
                ]
            )

        return np.array(x), np.array(y)

    def fit(self, data: pd.DataFrame, target_col: str) -> "TCNForecaster":
        """Fit TCN model."""
        logger.info("Fitting TCN model")

        try:
            # Select numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

            if target_col not in numeric_cols:
                raise ValueError(f"Target column {target_col} is not numeric")

            # Use target + other features
            self.feature_cols = numeric_cols
            target_idx = self.feature_cols.index(target_col)

            data_clean = data[self.feature_cols].dropna().values

            if len(data_clean) < self.sequence_length + self.prediction_length:
                raise ValueError("Not enough data for TCN training")

            # Prepare data
            x, y = self._prepare_data(data_clean, target_idx)

            # Create model
            input_size = x.shape[2]
            self.model = self._create_model(input_size)

            # Create data loader
            x_tensor = torch.FloatTensor(x).to(self.device)
            y_tensor = torch.FloatTensor(y).to(self.device)
            dataset = TensorDataset(x_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            # Training
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            criterion = nn.MSELoss()

            self.model.train()
            for epoch in range(self.epochs):
                total_loss = 0
                for batch_x, batch_y in dataloader:
                    optimizer.zero_grad()
                    outputs = self.model(batch_x)
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
            logger.info("TCN model fitted successfully")

        except Exception as e:
            logger.error(f"Error fitting TCN model: {e}")
            raise

        return self

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Generate TCN predictions."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")

        try:
            data_clean = data[self.feature_cols].dropna().values

            if len(data_clean) < self.sequence_length:
                raise ValueError("Not enough data for prediction")

            # Use last sequence_length observations
            last_sequence = data_clean[-self.sequence_length :]

            # Predict
            self.model.eval()
            with torch.no_grad():
                x_pred = torch.FloatTensor(last_sequence).unsqueeze(0).to(self.device)
                prediction = self.model(x_pred).cpu().numpy().flatten()

            return np.array(prediction)

        except Exception as e:
            logger.error(f"Error generating TCN predictions: {e}")
            # Return naive forecast as fallback
            last_value = (
                data[self.feature_cols[0]].dropna().iloc[-1] if len(data) > 0 else 0.0
            )
            return np.full(self.prediction_length, last_value)
