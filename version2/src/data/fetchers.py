"""Data fetching utilities for financial and economic data."""

import warnings

import pandas as pd
import yfinance as yf
from fredapi import Fred

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Suppress yfinance warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")


class YahooFinanceFetcher:
    """Fetcher for market data from Yahoo Finance."""

    def __init__(self) -> None:
        """Initialize the Yahoo Finance fetcher."""
        self.name = "yahoo_finance"

    def fetch_data(
        self, symbols: list[str], start_date: str, end_date: str, interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch market data for given symbols.

        Args:
            symbols: List of ticker symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval (1d, 1wk, 1mo, etc.)

        Returns:
            DataFrame with market data
        """
        logger.info(f"Fetching Yahoo Finance data for {len(symbols)} symbols")

        try:
            # Download data
            data = yf.download(
                symbols,
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True,
                progress=False,
            )

            if data.empty:
                logger.warning("No data retrieved from Yahoo Finance")
                return pd.DataFrame()

            # Handle single vs multiple symbols
            if len(symbols) == 1:
                # Single symbol - add symbol level to columns
                data.columns = pd.MultiIndex.from_product(
                    [data.columns, symbols], names=["metric", "symbol"]
                )

            # Flatten column names for easier processing
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [f"{col[1]}_{col[0]}" for col in data.columns]

            logger.info(f"Retrieved {len(data)} rows of Yahoo Finance data")
            return data

        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance data: {e}")
            return pd.DataFrame()


class FredFetcher:
    """Fetcher for economic data from FRED (Federal Reserve Economic Data)."""

    def __init__(self, api_key: str | None = None) -> None:
        """
        Initialize the FRED fetcher.

        Args:
            api_key: FRED API key. If None, will try to get from environment.
        """
        self.name = "fred"
        self.api_key = api_key

        if api_key:
            self.fred = Fred(api_key=api_key)
        else:
            try:
                self.fred = Fred()  # Will try to get API key from environment
            except ValueError:
                logger.warning(
                    "FRED API key not provided. FRED data will not be available."
                )
                self.fred = None

    def fetch_data(
        self, series_ids: list[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Fetch economic data for given FRED series.

        Args:
            series_ids: List of FRED series IDs
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with economic data
        """
        if self.fred is None:
            logger.error("FRED API not available")
            return pd.DataFrame()

        logger.info(f"Fetching FRED data for {len(series_ids)} series")

        data_dict = {}

        for series_id in series_ids:
            try:
                series_data = self.fred.get_series(
                    series_id, observation_start=start_date, observation_end=end_date
                )

                if not series_data.empty:
                    data_dict[series_id] = series_data
                    logger.debug(
                        f"Retrieved {len(series_data)} observations for {series_id}"
                    )
                else:
                    logger.warning(f"No data for FRED series {series_id}")

            except Exception as e:
                logger.error(f"Error fetching FRED series {series_id}: {e}")

        if not data_dict:
            logger.warning("No FRED data retrieved")
            return pd.DataFrame()

        # Combine all series into a single DataFrame
        data = pd.DataFrame(data_dict)
        logger.info(f"Retrieved FRED data with shape {data.shape}")

        return data


class DataFetcher:
    """Main data fetcher that coordinates multiple data sources."""

    def __init__(self, fred_api_key: str | None = None) -> None:
        """
        Initialize the main data fetcher.

        Args:
            fred_api_key: FRED API key
        """
        self.yahoo_fetcher = YahooFinanceFetcher()
        self.fred_fetcher = FredFetcher(fred_api_key)

    def fetch_all_data(
        self,
        market_symbols: list[str],
        fred_series: list[str],
        start_date: str,
        end_date: str,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch data from all sources.

        Args:
            market_symbols: List of market ticker symbols
            fred_series: List of FRED series IDs
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            Dictionary with data from each source
        """
        logger.info("Fetching data from all sources")

        data = {}

        # Fetch market data
        if market_symbols:
            market_data = self.yahoo_fetcher.fetch_data(
                market_symbols, start_date, end_date
            )
            if not market_data.empty:
                data["market"] = market_data

        # Fetch economic data
        if fred_series:
            fred_data = self.fred_fetcher.fetch_data(fred_series, start_date, end_date)
            if not fred_data.empty:
                data["economic"] = fred_data

        logger.info(f"Fetched data from {len(data)} sources")
        return data


def get_sample_data() -> dict[str, pd.DataFrame]:
    """Get sample data for testing purposes."""
    logger.info("Generating sample data")

    # Create sample market data
    dates = pd.date_range("2020-01-01", "2023-12-31", freq="D")
    n_days = len(dates)

    # Simulate S&P 500 returns
    import numpy as np

    np.random.seed(42)

    returns = np.random.normal(0.0005, 0.02, n_days)  # Daily returns
    prices = 3000 * np.exp(np.cumsum(returns))  # Price series

    market_data = pd.DataFrame(
        {
            "^GSPC_Close": prices,
            "^GSPC_Volume": np.random.randint(int(1e9), int(5e9), n_days),
            "^VIX_Close": 20 + 10 * np.random.exponential(0.5, n_days),
        },
        index=dates,
    )

    # Create sample economic data (monthly)
    monthly_dates = pd.date_range("2020-01-01", "2023-12-31", freq="MS")
    n_months = len(monthly_dates)

    economic_data = pd.DataFrame(
        {
            "DGS10": 2.0 + np.random.normal(0, 0.5, n_months),  # 10-year yield
            "UNRATE": 5.0 + np.random.normal(0, 1.0, n_months),  # Unemployment
            "FEDFUNDS": 1.0 + np.random.normal(0, 0.3, n_months),  # Fed funds rate
        },
        index=monthly_dates,
    )

    return {"market": market_data, "economic": economic_data}
