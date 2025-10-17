"""Data cleaning and alignment utilities."""

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataCleaner:
    """Utility class for cleaning and aligning financial time series data."""

    def __init__(self, max_missing_ratio: float = 0.1) -> None:
        """
        Initialize the data cleaner.

        Args:
            max_missing_ratio: Maximum ratio of missing values allowed per series
        """
        self.max_missing_ratio = max_missing_ratio

    def clean_market_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean market data from Yahoo Finance.

        Args:
            data: Raw market data DataFrame

        Returns:
            Cleaned market data
        """
        logger.info("Cleaning market data")

        if data.empty:
            return data

        # Remove weekends and holidays (keep only business days)
        data = data[data.index.dayofweek < 5]

        # Handle missing values
        data = self._handle_missing_values(data)

        # Remove outliers (basic approach)
        data = self._remove_outliers(data)

        # Ensure proper data types
        data = data.astype(float, errors="ignore")

        logger.info(f"Cleaned market data shape: {data.shape}")
        return data

    def clean_economic_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean economic data from FRED.

        Args:
            data: Raw economic data DataFrame

        Returns:
            Cleaned economic data
        """
        logger.info("Cleaning economic data")

        if data.empty:
            return data

        # Handle missing values
        data = self._handle_missing_values(data)

        # Forward fill for economic indicators (common practice)
        data = data.fillna(method="ffill")

        # Ensure proper data types
        data = data.astype(float, errors="ignore")

        logger.info(f"Cleaned economic data shape: {data.shape}")
        return data

    def align_frequencies(
        self,
        market_data: pd.DataFrame,
        economic_data: pd.DataFrame,
        target_freq: str = "D",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Align data frequencies between market and economic data.

        Args:
            market_data: Daily market data
            economic_data: Economic data (potentially different frequency)
            target_freq: Target frequency for alignment

        Returns:
            Tuple of aligned market and economic data
        """
        logger.info(f"Aligning data frequencies to {target_freq}")

        if market_data.empty or economic_data.empty:
            return market_data, economic_data

        # Resample economic data to target frequency
        if target_freq == "D":
            # Forward fill economic data to daily frequency
            economic_aligned = economic_data.resample("D").ffill()
        elif target_freq == "W":
            # Resample to weekly
            market_aligned = market_data.resample("W").last()
            economic_aligned = economic_data.resample("W").last()
        elif target_freq == "M":
            # Resample to monthly
            market_aligned = market_data.resample("M").last()
            economic_aligned = economic_data.resample("M").last()
        else:
            raise ValueError(f"Unsupported target frequency: {target_freq}")

        # Align indices
        if target_freq == "D":
            # For daily, keep market data as is and align economic
            common_index = market_data.index.intersection(economic_aligned.index)
            market_aligned = market_data.loc[common_index]
            economic_aligned = economic_aligned.loc[common_index]
        else:
            # For other frequencies, both are resampled
            common_index = market_aligned.index.intersection(economic_aligned.index)
            market_aligned = market_aligned.loc[common_index]
            economic_aligned = economic_aligned.loc[common_index]

        logger.info(
            f"Aligned data shape - Market: {market_aligned.shape}, Economic: {economic_aligned.shape}"
        )
        return market_aligned, economic_aligned

    def create_combined_dataset(
        self, data_dict: dict[str, pd.DataFrame], target_freq: str = "D"
    ) -> pd.DataFrame:
        """
        Create a combined dataset from multiple data sources.

        Args:
            data_dict: Dictionary of DataFrames from different sources
            target_freq: Target frequency for the combined dataset

        Returns:
            Combined DataFrame with all features
        """
        logger.info("Creating combined dataset")

        if not data_dict:
            return pd.DataFrame()

        # Start with the first dataset
        datasets = list(data_dict.values())
        combined = datasets[0].copy()

        # Align and merge other datasets
        for i, dataset in enumerate(datasets[1:], 1):
            source_name = list(data_dict.keys())[i]
            logger.debug(f"Merging dataset from {source_name}")

            # Align frequencies if needed
            if target_freq == "D" and dataset.index.freq != "D":
                dataset = dataset.resample("D").ffill()

            # Merge on index (outer join to keep all dates)
            combined = combined.join(dataset, how="outer")

        # Final cleaning
        combined = self._handle_missing_values(combined)

        logger.info(f"Combined dataset shape: {combined.shape}")
        return combined

    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        if data.empty:
            return data

        # Calculate missing ratio per column
        missing_ratios = data.isnull().sum() / len(data)

        # Remove columns with too many missing values
        cols_to_keep = missing_ratios[missing_ratios <= self.max_missing_ratio].index
        data = data[cols_to_keep]

        if len(cols_to_keep) < len(missing_ratios):
            dropped_cols = len(missing_ratios) - len(cols_to_keep)
            logger.warning(
                f"Dropped {dropped_cols} columns due to excessive missing values"
            )

        # Forward fill remaining missing values
        data = data.fillna(method="ffill")

        # Backward fill any remaining missing values at the beginning
        data = data.fillna(method="bfill")

        return data

    def _remove_outliers(
        self, data: pd.DataFrame, method: str = "iqr", threshold: float = 3.0
    ) -> pd.DataFrame:
        """Remove outliers from the dataset."""
        if data.empty:
            return data

        numeric_cols = data.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if method == "iqr":
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Cap outliers instead of removing them
                data[col] = data[col].clip(lower_bound, upper_bound)

            elif method == "zscore":
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                data.loc[z_scores > threshold, col] = np.nan

        # Fill any new NaN values created by outlier removal
        return data.fillna(method="ffill").fillna(method="bfill")


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional features from the base dataset.

    Args:
        data: Base dataset

    Returns:
        Dataset with additional features
    """
    logger.info("Creating additional features")

    if data.empty:
        return data

    features_data = data.copy()

    # Create returns for price series
    price_cols = [col for col in data.columns if "Close" in col or "Price" in col]
    for col in price_cols:
        # Daily returns
        features_data[f"{col}_Return"] = data[col].pct_change()

        # Log returns
        features_data[f"{col}_LogReturn"] = np.log(data[col] / data[col].shift(1))

        # Volatility (rolling standard deviation of returns)
        features_data[f"{col}_Volatility"] = (
            features_data[f"{col}_Return"].rolling(20).std()
        )

    # Create moving averages
    for col in data.select_dtypes(include=[np.number]).columns:
        features_data[f"{col}_MA5"] = data[col].rolling(5).mean()
        features_data[f"{col}_MA20"] = data[col].rolling(20).mean()
        features_data[f"{col}_MA50"] = data[col].rolling(50).mean()

    # Create yield spreads if we have multiple yield series
    yield_cols = [col for col in data.columns if "DGS" in col or "yield" in col.lower()]
    if len(yield_cols) >= 2:
        for i, col1 in enumerate(yield_cols):
            for col2 in yield_cols[i + 1 :]:
                features_data[f"{col1}_{col2}_Spread"] = data[col1] - data[col2]

    # Remove infinite values
    features_data = features_data.replace([np.inf, -np.inf], np.nan)

    # Forward fill any new NaN values
    features_data = features_data.fillna(method="ffill").fillna(method="bfill")

    logger.info(f"Created features dataset with shape: {features_data.shape}")
    return features_data
