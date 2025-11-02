"""Tokenization utilities for financial time series data."""


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FinancialDataTokenizer:
    """Base tokenizer for converting financial time series to tokens."""

    def __init__(
        self, num_bins: int = 1024, method: str = "uniform", context_length: int = 512
    ) -> None:
        """
        Initialize the tokenizer.

        Args:
            num_bins: Number of bins for quantization
            method: Tokenization method ('uniform', 'quantile', 'kmeans')
            context_length: Maximum context length for sequences
        """
        self.num_bins = num_bins
        self.method = method
        self.context_length = context_length

        # Fitted parameters
        self.is_fitted = False
        self.feature_names: list[str] = []
        self.scalers: dict[str, StandardScaler] = {}
        self.bin_edges: dict[str, np.ndarray] = {}
        self.kmeans_models: dict[str, KMeans] = {}

        # Special tokens
        self.pad_token = 0
        self.unk_token = 1
        self.start_token = 2
        self.end_token = 3
        self.special_tokens = 4

    def fit(self, data: pd.DataFrame) -> "FinancialDataTokenizer":
        """
        Fit the tokenizer on training data.

        Args:
            data: Training data DataFrame

        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting tokenizer with method '{self.method}'")

        if data.empty:
            raise ValueError("Cannot fit tokenizer on empty data")

        self.feature_names = list(data.columns)

        for col in self.feature_names:
            series = data[col].dropna()

            if len(series) == 0:
                logger.warning(f"No valid data for feature {col}")
                continue

            # Fit scaler
            scaler = StandardScaler()
            scaled_values = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
            self.scalers[col] = scaler

            # Fit binning strategy
            if self.method == "uniform":
                self._fit_uniform_bins(col, scaled_values)
            elif self.method == "quantile":
                self._fit_quantile_bins(col, scaled_values)
            elif self.method == "kmeans":
                self._fit_kmeans_bins(col, scaled_values)
            else:
                raise ValueError(f"Unknown tokenization method: {self.method}")

        self.is_fitted = True
        logger.info("Tokenizer fitting completed")
        return self

    def transform(self, data: pd.DataFrame) -> dict[str, dict[str, np.ndarray] | np.ndarray]:
        """
        Transform data to tokens.

        Args:
            data: Data to transform

        Returns:
            Dictionary with tokenized sequences
        """
        if not self.is_fitted:
            raise ValueError("Tokenizer must be fitted before transform")

        logger.info("Transforming data to tokens")

        tokens_dict = {}

        for col in self.feature_names:
            if col not in data.columns:
                logger.warning(f"Feature {col} not found in data")
                continue

            series = data[col].ffill().bfill()
            tokens = self._transform_series(col, series)
            tokens_dict[col] = tokens

        # Create combined token sequences
        combined_tokens = self._create_combined_sequences(tokens_dict)

        result: dict[str, dict[str, np.ndarray] | np.ndarray] = {
            "individual": tokens_dict,
            "combined": combined_tokens,
        }
        return result

    def fit_transform(
        self, data: pd.DataFrame
    ) -> dict[str, dict[str, np.ndarray] | np.ndarray]:  # Returns tokenized outputs
        """Fit and transform in one step."""
        return self.fit(data).transform(data)

    def inverse_transform(self, tokens: np.ndarray, feature_name: str) -> np.ndarray:
        """
        Convert tokens back to approximate original values.

        Args:
            tokens: Token sequence
            feature_name: Name of the feature

        Returns:
            Approximate original values
        """
        if not self.is_fitted:
            raise ValueError("Tokenizer must be fitted before inverse transform")

        if feature_name not in self.scalers:
            raise ValueError(f"Feature {feature_name} not found in fitted tokenizer")

        # Convert tokens to bin centers
        if self.method == "uniform" or self.method == "quantile":
            bin_edges = self.bin_edges[feature_name]
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            scaled_values = bin_centers[tokens - self.special_tokens]
        elif self.method == "kmeans":
            kmeans = self.kmeans_models[feature_name]
            scaled_values = kmeans.cluster_centers_[
                tokens - self.special_tokens
            ].flatten()
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Inverse scale
        scaler = self.scalers[feature_name]
        original_values = scaler.inverse_transform(
            scaled_values.reshape(-1, 1)
        ).flatten()

        return original_values

    def _fit_uniform_bins(self, feature_name: str, values: np.ndarray) -> None:
        """Fit uniform bins for a feature."""
        min_val, max_val = values.min(), values.max()
        bin_edges = np.linspace(
            min_val, max_val, self.num_bins - self.special_tokens + 1
        )
        self.bin_edges[feature_name] = bin_edges

    def _fit_quantile_bins(self, feature_name: str, values: np.ndarray) -> None:
        """Fit quantile-based bins for a feature."""
        quantiles = np.linspace(0, 1, self.num_bins - self.special_tokens + 1)
        bin_edges = np.quantile(values, quantiles)
        self.bin_edges[feature_name] = bin_edges

    def _fit_kmeans_bins(self, feature_name: str, values: np.ndarray) -> None:
        """Fit K-means based bins for a feature."""
        kmeans = KMeans(
            n_clusters=self.num_bins - self.special_tokens, random_state=42, n_init=10
        )
        kmeans.fit(values.reshape(-1, 1))
        self.kmeans_models[feature_name] = kmeans

    def _transform_series(self, feature_name: str, series: pd.Series) -> np.ndarray:
        """Transform a single series to tokens."""
        # Scale the series
        scaler = self.scalers[feature_name]
        scaled_values = scaler.transform(series.values.reshape(-1, 1)).flatten()

        # Convert to tokens
        if self.method in ["uniform", "quantile"]:
            bin_edges = self.bin_edges[feature_name]
            tokens = np.digitize(scaled_values, bin_edges) + self.special_tokens - 1
            # Clip to valid range
            tokens = np.clip(tokens, self.special_tokens, self.num_bins - 1)
        elif self.method == "kmeans":
            kmeans = self.kmeans_models[feature_name]
            cluster_labels = kmeans.predict(scaled_values.reshape(-1, 1))
            tokens = cluster_labels + self.special_tokens

        return tokens.astype(np.int32)

    def _create_combined_sequences(
        self, tokens_dict: dict[str, np.ndarray]
    ) -> np.ndarray:
        """Create combined token sequences from individual feature tokens."""
        if not tokens_dict:
            return np.array([])

        # Get the length of sequences (should be same for all features)
        seq_length = len(next(iter(tokens_dict.values())))

        # Create combined sequences by interleaving tokens
        combined = []

        for i in range(seq_length):
            # Add start token at beginning of each timestep
            if i == 0:
                combined.append(self.start_token)

            # Add tokens from all features for this timestep
            for feature_name in self.feature_names:
                if feature_name in tokens_dict:
                    combined.append(tokens_dict[feature_name][i])

            # Add end token at the end
            if i == seq_length - 1:
                combined.append(self.end_token)

        # Truncate to context length if necessary
        if len(combined) > self.context_length:
            combined = combined[-self.context_length :]

        return np.array(combined, dtype=np.int32)


class AdvancedTokenizer(FinancialDataTokenizer):
    """Advanced tokenizer with additional features and preprocessing."""

    def __init__(
        self,
        num_bins: int = 1024,
        method: str = "quantile",
        context_length: int = 512,
        include_technical_indicators: bool = True,
        include_time_features: bool = True,
    ) -> None:
        """
        Initialize the advanced tokenizer.

        Args:
            num_bins: Number of bins for quantization
            method: Tokenization method
            context_length: Maximum context length
            include_technical_indicators: Whether to include technical indicators
            include_time_features: Whether to include time-based features
        """
        super().__init__(num_bins, method, context_length)
        self.include_technical_indicators = include_technical_indicators
        self.include_time_features = include_time_features

    def fit(self, data: pd.DataFrame) -> "AdvancedTokenizer":
        """Fit the advanced tokenizer with feature engineering."""
        logger.info("Fitting advanced tokenizer with feature engineering")

        # Create additional features
        enhanced_data = self._create_features(data)

        # Fit the base tokenizer
        return super().fit(enhanced_data)

    def transform(self, data: pd.DataFrame) -> dict[str, dict[str, np.ndarray] | np.ndarray]:
        """Transform data with feature engineering."""
        # Create additional features
        enhanced_data = self._create_features(data)

        # Transform using base tokenizer
        return super().transform(enhanced_data)

    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create additional features for tokenization."""
        enhanced_data = data.copy()

        if self.include_technical_indicators:
            enhanced_data = self._add_technical_indicators(enhanced_data)

        if self.include_time_features:
            enhanced_data = self._add_time_features(enhanced_data)

        return enhanced_data

    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataset."""
        # Simple moving averages
        for col in data.select_dtypes(include=[np.number]).columns:
            if "Close" in col or "Price" in col:
                data[f"{col}_SMA_5"] = data[col].rolling(5).mean()
                data[f"{col}_SMA_20"] = data[col].rolling(20).mean()

                # Relative Strength Index (simplified)
                delta = data[col].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                data[f"{col}_RSI"] = 100 - (100 / (1 + rs))

        return data

    def _add_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features to the dataset."""
        if isinstance(data.index, pd.DatetimeIndex):
            data["DayOfWeek"] = data.index.dayofweek
            data["Month"] = data.index.month
            data["Quarter"] = data.index.quarter
            data["DayOfYear"] = data.index.dayofyear

        return data


def create_sliding_windows(
    tokens: np.ndarray, window_size: int, prediction_length: int, stride: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows for training.

    Args:
        tokens: Token sequence
        window_size: Size of input window
        prediction_length: Length of prediction horizon
        stride: Stride for sliding window

    Returns:
        Tuple of (input_sequences, target_sequences)
    """
    if len(tokens) < window_size + prediction_length:
        raise ValueError(
            "Token sequence too short for given window and prediction sizes"
        )

    inputs = []
    targets = []

    for i in range(0, len(tokens) - window_size - prediction_length + 1, stride):
        input_seq = tokens[i : i + window_size]
        target_seq = tokens[i + window_size : i + window_size + prediction_length]

        inputs.append(input_seq)
        targets.append(target_seq)

    return np.array(inputs), np.array(targets)
