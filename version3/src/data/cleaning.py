"""Data cleaning and preprocessing utilities."""

from typing import Optional

import numpy as np
import pandas as pd


class DataCleaner:
    """Clean and preprocess financial data."""

    def __init__(self, max_missing_ratio: float = 0.1):
        """Initialize data cleaner.

        Args:
            max_missing_ratio: Maximum allowed missing data ratio
        """
        self.max_missing_ratio = max_missing_ratio

    def clean_market_data(
        self,
        data: pd.DataFrame,
        forward_fill: bool = True,
    ) -> pd.DataFrame:
        """Clean market price data.

        Args:
            data: Raw market data
            forward_fill: Whether to forward-fill missing values

        Returns:
            Cleaned market data
        """
        cleaned = data.copy()

        # Remove columns with too many missing values
        missing_ratio = cleaned.isna().sum() / len(cleaned)
        cols_to_keep = missing_ratio[
            missing_ratio <= self.max_missing_ratio
        ].index
        cleaned = cleaned[cols_to_keep]

        # Forward-fill missing values for market data
        if forward_fill:
            cleaned = cleaned.ffill()

        # Fill remaining NaNs with backward fill
        cleaned = cleaned.bfill()

        return cleaned

    def clean_economic_data(
        self,
        data: pd.DataFrame,
        interpolate: bool = True,
    ) -> pd.DataFrame:
        """Clean economic indicator data.

        Args:
            data: Raw economic data
            interpolate: Whether to interpolate missing values

        Returns:
            Cleaned economic data
        """
        cleaned = data.copy()

        # Remove columns with too many missing values
        missing_ratio = cleaned.isna().sum() / len(cleaned)
        cols_to_keep = missing_ratio[
            missing_ratio <= self.max_missing_ratio
        ].index
        cleaned = cleaned[cols_to_keep]

        # Interpolate for economic data
        if interpolate:
            cleaned = cleaned.interpolate(method="linear", limit_direction="both")

        # Fill remaining NaNs
        cleaned = cleaned.fillna(method="ffill").fillna(method="bfill")

        return cleaned

    def align_frequencies(
        self,
        market_data: Optional[pd.DataFrame],
        economic_data: Optional[pd.DataFrame],
        target_freq: str = "D",
    ) -> pd.DataFrame:
        """Align different frequency data sources.

        Args:
            market_data: Daily/high frequency data
            economic_data: Monthly/low frequency data
            target_freq: Target frequency (D, M, Q, Y)

        Returns:
            Aligned DataFrame with all data
        """
        dataframes = []

        if market_data is not None:
            dataframes.append(market_data)

        if economic_data is not None:
            # Resample economic data to target frequency
            economic_resampled = economic_data.resample(target_freq).ffill()
            dataframes.append(economic_resampled)

        if not dataframes:
            raise ValueError("No data provided")

        # Merge all data
        aligned = dataframes[0]
        for df in dataframes[1:]:
            aligned = aligned.join(df, how="outer")

        # Fill remaining NaNs
        aligned = aligned.fillna(method="ffill").fillna(method="bfill")

        return aligned

    def create_combined_dataset(
        self,
        data_dict: dict[str, Optional[pd.DataFrame]],
        target_freq: str = "D",
    ) -> pd.DataFrame:
        """Create combined dataset from market and economic data.

        Args:
            data_dict: Dictionary with 'market' and 'economic' DataFrames
            target_freq: Target frequency

        Returns:
            Combined DataFrame
        """
        return self.align_frequencies(
            data_dict.get("market"),
            data_dict.get("economic"),
            target_freq,
        )


def create_features(
    data: pd.DataFrame,
    lag_periods: Optional[list[int]] = None,
    rolling_windows: Optional[list[int]] = None,
) -> pd.DataFrame:
    """Create technical and lagged features from price data.

    Args:
        data: Input DataFrame
        lag_periods: Periods for lagged features
        rolling_windows: Windows for rolling statistics

    Returns:
        DataFrame with additional features
    """
    lag_periods = lag_periods or [1, 5, 20]
    rolling_windows = rolling_windows or [5, 10, 20]

    features = data.copy()

    # Get closing prices (look for Close columns)
    close_cols = [col for col in features.columns if "Close" in str(col)]

    # Create lagged features
    for col in close_cols:
        for period in lag_periods:
            features[f"{col}_lag_{period}"] = features[col].shift(period)

    # Create rolling statistics
    for col in close_cols:
        for window in rolling_windows:
            features[f"{col}_rolling_mean_{window}"] = features[col].rolling(
                window=window
            ).mean()
            features[f"{col}_rolling_std_{window}"] = features[col].rolling(
                window=window
            ).std()

    # Calculate returns
    for col in close_cols:
        features[f"{col}_returns"] = features[col].pct_change()
        features[f"{col}_log_returns"] = np.log(
            features[col] / features[col].shift(1)
        )

    # Drop rows with NaNs created by lagging
    features = features.dropna()

    return features
