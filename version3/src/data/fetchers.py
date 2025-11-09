"""Data fetching from various sources."""

import os
from datetime import datetime
from typing import Optional

import pandas as pd
import yfinance as yf


class YahooFinanceFetcher:
    """Fetch financial data from Yahoo Finance."""

    @staticmethod
    def fetch_data(
        tickers: list[str],
        start_date: str,
        end_date: str,
        interval: str = "1d",
        auto_adjust: bool = True,
    ) -> pd.DataFrame:
        """Fetch historical price data.

        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval (1d, 1wk, 1mo)
            auto_adjust: Automatically adjust OHLC for splits/dividends

        Returns:
            DataFrame with OHLCV data
        """
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=auto_adjust,
            progress=False,
        )

        if len(tickers) == 1:
            # yfinance returns (Price, Ticker) format, swap to (Ticker, Price)
            data.columns = data.columns.swaplevel(0, 1)

        return data


class FredFetcher:
    """Fetch economic data from FRED."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize FRED fetcher.

        Args:
            api_key: FRED API key. If None, uses FRED_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("FRED_API_KEY")

        if not self.api_key:
            self.fred = None
        else:
            try:
                from fredapi import Fred  # type: ignore

                self.fred = Fred(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "fredapi not installed. Install with: pip install fredapi"
                )

    def fetch_data(
        self,
        series_ids: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch economic data from FRED.

        Args:
            series_ids: List of FRED series IDs
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with economic indicators
        """
        if self.fred is None:
            raise ValueError(
                "FRED API key not configured. Set FRED_API_KEY environment"
                " variable or pass api_key to FredFetcher."
            )

        data_dict = {}

        for series_id in series_ids:
            try:
                series = self.fred.get_series(
                    series_id,
                    observation_start=start_date,
                    observation_end=end_date,
                )
                data_dict[series_id] = series
            except Exception as e:
                print(f"Error fetching {series_id}: {e}")
                continue

        return pd.DataFrame(data_dict)


class DataFetcher:
    """Main data fetcher combining multiple sources."""

    def __init__(self, fred_api_key: Optional[str] = None):
        """Initialize data fetcher.

        Args:
            fred_api_key: FRED API key
        """
        self.yahoo = YahooFinanceFetcher()
        self.fred = FredFetcher(fred_api_key)

    def fetch_all_data(
        self,
        market_symbols: list[str],
        fred_series: list[str],
        start_date: str,
        end_date: str,
    ) -> dict[str, pd.DataFrame]:
        """Fetch data from all sources.

        Args:
            market_symbols: List of market symbols
            fred_series: List of FRED series IDs
            start_date: Start date
            end_date: End date

        Returns:
            Dictionary with 'market' and 'economic' DataFrames
        """
        market_data = None
        economic_data = None

        # Fetch market data
        if market_symbols:
            try:
                market_data = self.yahoo.fetch_data(
                    market_symbols, start_date, end_date
                )
            except Exception as e:
                print(f"Error fetching market data: {e}")

        # Fetch economic data
        if fred_series and self.fred.fred is not None:
            try:
                economic_data = self.fred.fetch_data(
                    fred_series, start_date, end_date
                )
            except Exception as e:
                print(f"Error fetching economic data: {e}")

        return {
            "market": market_data,
            "economic": economic_data,
        }
