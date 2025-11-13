"""Tokenization strategies for converting numerical data to tokens."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


class FinancialDataTokenizer:
    """Basic tokenizer for financial time series data."""

    def __init__(
        self,
        num_bins: int = 1024,
        method: str = "quantile",
        context_length: int = 512,
    ):
        """Initialize tokenizer.

        Args:
            num_bins: Number of tokens in vocabulary
            method: Tokenization method (uniform, quantile, kmeans)
            context_length: Length of context window
        """
        self.num_bins = num_bins
        self.method = method
        self.context_length = context_length
        self.bin_edges: dict[str, np.ndarray] = {}
        self.column_stats: dict[str, dict] = {}
        self.kmeans_models: dict[str, KMeans] = {}

    def fit(self, data: pd.DataFrame) -> None:
        """Fit tokenizer on training data.

        Args:
            data: Training DataFrame
        """
        for col in data.columns:
            col_data = data[col].dropna().values

            if len(col_data) == 0:
                continue

            # Ensure col_data is a numpy array
            if not isinstance(col_data, np.ndarray):
                col_data = np.array(col_data)

            # Convert to float array for safe operations
            col_data_float = col_data.astype(np.float64)

            self.column_stats[col] = {
                "min": float(col_data_float.min()),
                "max": float(col_data_float.max()),
                "mean": float(col_data_float.mean()),
                "std": float(col_data_float.std()),
            }

            if self.method == "uniform":
                self._fit_uniform(col, col_data_float)
            elif self.method == "quantile":
                self._fit_quantile(col, col_data_float)
            elif self.method == "kmeans":
                self._fit_kmeans(col, col_data_float)

    def _fit_uniform(self, col: str, data: np.ndarray) -> None:
        """Fit uniform binning."""
        min_val = data.min()
        max_val = data.max()
        self.bin_edges[col] = np.linspace(min_val, max_val, self.num_bins + 1)

    def _fit_quantile(self, col: str, data: np.ndarray) -> None:
        """Fit quantile binning."""
        quantiles = np.linspace(0, 1, self.num_bins + 1)
        self.bin_edges[col] = np.quantile(data, quantiles)

    def _fit_kmeans(self, col: str, data: np.ndarray) -> None:
        """Fit k-means binning."""
        kmeans = KMeans(n_clusters=self.num_bins, random_state=42)
        kmeans.fit(data.reshape(-1, 1))
        self.kmeans_models[col] = kmeans
        centers = kmeans.cluster_centers_.flatten()
        self.bin_edges[col] = np.sort(centers)

    def transform(self, data: pd.DataFrame) -> dict[str, np.ndarray]:
        """Transform data to tokens.

        Args:
            data: Input DataFrame

        Returns:
            Dictionary with tokenized data
        """
        tokens = {}

        for col in data.columns:
            if col not in self.bin_edges:
                continue

            col_data = data[col].values

            # Convert to numpy array if it's an ExtensionArray
            if not isinstance(col_data, np.ndarray):
                col_data = np.array(col_data, dtype=np.float64)
            else:
                col_data = col_data.astype(np.float64)

            if self.method == "kmeans":
                token_indices = self.kmeans_models[col].predict(col_data.reshape(-1, 1))
            else:
                token_indices = np.digitize(col_data, self.bin_edges[col]) - 1
                token_indices = np.clip(token_indices, 0, self.num_bins - 1)

            tokens[col] = token_indices

        return tokens

    def inverse_transform(
        self,
        tokens: np.ndarray,
        col: str,
    ) -> np.ndarray:
        """Convert tokens back to approximate values.

        Args:
            tokens: Token indices
            col: Column name

        Returns:
            Approximate values
        """
        if col not in self.bin_edges:
            raise ValueError(f"Column {col} not fitted")

        if self.method == "kmeans":
            values = self.kmeans_models[col].cluster_centers_[tokens].flatten()
        else:
            bin_edges = self.bin_edges[col]
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            values = bin_centers[tokens]

        return values

    def get_vocabulary_size(self) -> int:
        """Get vocabulary size.

        Returns:
            Number of tokens
        """
        return self.num_bins


class AdvancedTokenizer(FinancialDataTokenizer):
    """Advanced tokenizer with technical indicators and time features."""

    def __init__(
        self,
        num_bins: int = 1024,
        method: str = "quantile",
        context_length: int = 512,
        include_technical_indicators: bool = True,
        include_time_features: bool = True,
    ):
        """Initialize advanced tokenizer.

        Args:
            num_bins: Number of tokens
            method: Tokenization method
            context_length: Context length
            include_technical_indicators: Whether to include technical indicators
            include_time_features: Whether to include time features
        """
        super().__init__(num_bins, method, context_length)
        self.include_technical_indicators = include_technical_indicators
        self.include_time_features = include_time_features

    def fit(self, data: pd.DataFrame) -> None:
        """Fit advanced tokenizer with feature engineering.

        Args:
            data: Training DataFrame
        """
        # Add technical indicators
        if self.include_technical_indicators:
            data = self._add_technical_indicators(data)

        # Add time features
        if self.include_time_features:
            data = self._add_time_features(data)

        super().fit(data)

    def _add_technical_indicators(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add technical indicators to data.

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with technical indicators
        """
        result = data.copy()

        # Get closing prices
        close_cols = [col for col in data.columns if "Close" in str(col)]

        # Collect all new columns to add at once for better performance
        new_cols = {}

        for col in close_cols:
            if col not in result.columns:
                continue

            # Simple moving averages
            for window in [5, 20, 50]:
                new_cols[f"{col}_SMA_{window}"] = result[col].rolling(window).mean()

            # RSI (Relative Strength Index)
            delta = result[col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            # Handle division by zero safely
            loss_safe = loss.where(loss != 0, 1e-10)
            rs = gain / loss_safe
            new_cols[f"{col}_RSI"] = 100 - (100 / (1 + rs))

        # Add all new columns at once to avoid fragmentation
        if new_cols:
            result = pd.concat([result, pd.DataFrame(new_cols, index=result.index)], axis=1)

        return result.ffill().bfill()

    def _add_time_features(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add time-based features.

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with time features
        """
        result = data.copy()

        # Add index if not datetime
        if not isinstance(result.index, pd.DatetimeIndex):
            result.index = pd.to_datetime(result.index)

        # Day of week
        result["day_of_week"] = result.index.day_of_week

        # Month
        result["month"] = result.index.month

        # Quarter
        result["quarter"] = result.index.quarter

        # Days since epoch
        result["days_since_epoch"] = (result.index - pd.Timestamp("1970-01-01")).days

        return result

    def create_token_sequences(
        self,
        data: pd.DataFrame,
        stride: int = 1,
    ) -> list[np.ndarray]:
        """Create sliding window token sequences.

        Args:
            data: Input DataFrame
            stride: Stride for sliding window

        Returns:
            List of token sequences
        """
        tokens_dict = self.transform(data)

        sequences = []

        # Stack all tokens
        token_matrix = np.column_stack(list(tokens_dict.values()))

        # Create sliding windows
        for i in range(0, len(token_matrix) - self.context_length, stride):
            seq = token_matrix[i : i + self.context_length]
            sequences.append(seq)

        return sequences
