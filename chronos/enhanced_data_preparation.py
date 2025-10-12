"""
Enhanced Data Preparation Scripts for Chronos Forecasting
Supports: Sentiment scores, User clickstreams, Engagement metrics, IMF macroeconomic data, etc.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import json
from imf_data_loader import IMFDataLoader


class EnhancedBehavioralDataLoader:
    """
    Enhanced data loader with IMF macroeconomic data integration
    """

    def __init__(self):
        """Initialize data loader with IMF data capability"""
        self.imf_loader = None

    def _get_imf_loader(self) -> IMFDataLoader:
        """Lazy initialization of IMF loader"""
        if self.imf_loader is None:
            self.imf_loader = IMFDataLoader()
        return self.imf_loader

    # IMF Macroeconomic Data Loading Methods
    def load_imf_inflation_data(
        self,
        countries: List[str] | None = None,
        start_year: int = 2000,
        end_year: int = 2023,
        country_to_use: str | None = None,
    ) -> List[float]:
        """
        Load inflation data from IMF for forecasting

        Args:
            countries: List of country codes to fetch
            start_year: Start year for data
            end_year: End year for data
            country_to_use: Specific country to return (if None, returns first available)

        Returns:
            List of inflation values for time series forecasting
        """
        if countries is None:
            countries = ["US"]  # Default to US data

        imf_loader = self._get_imf_loader()
        inflation_data = imf_loader.get_inflation_data(countries, start_year, end_year)

        # Return data for specified country or first available
        if country_to_use and country_to_use in inflation_data:
            return inflation_data[country_to_use]
        elif inflation_data:
            return list(inflation_data.values())[0]
        else:
            print("Warning: No inflation data available, generating synthetic data")
            return self.generate_synthetic_inflation(end_year - start_year + 1)

    def load_imf_gdp_growth_data(
        self,
        countries: List[str] | None = None,
        start_year: int = 2000,
        end_year: int = 2023,
        country_to_use: str | None = None,
    ) -> List[float]:
        """
        Load GDP growth data from IMF for forecasting

        Args:
            countries: List of country codes to fetch
            start_year: Start year for data
            end_year: End year for data
            country_to_use: Specific country to return (if None, returns first available)

        Returns:
            List of GDP growth values for time series forecasting
        """
        if countries is None:
            countries = ["US"]  # Default to US data

        imf_loader = self._get_imf_loader()
        gdp_data = imf_loader.get_gdp_growth_data(countries, start_year, end_year)

        # Return data for specified country or first available
        if country_to_use and country_to_use in gdp_data:
            return gdp_data[country_to_use]
        elif gdp_data:
            return list(gdp_data.values())[0]
        else:
            print("Warning: No GDP data available, generating synthetic data")
            return self.generate_synthetic_gdp_growth(end_year - start_year + 1)

    def load_imf_exchange_rate_data(
        self,
        base_currency: str = "USD",
        target_currency: str = "EUR",
        start_year: int = 2000,
        end_year: int = 2023,
    ) -> List[float]:
        """
        Load exchange rate data from IMF for forecasting

        Args:
            base_currency: Base currency code
            target_currency: Target currency code
            start_year: Start year for data
            end_year: End year for data

        Returns:
            List of exchange rate values for time series forecasting
        """
        imf_loader = self._get_imf_loader()
        fx_data = imf_loader.fetch_exchange_rates(
            base_currency, [target_currency], start_year, end_year
        )

        pair_key = f"{base_currency}/{target_currency}"
        if pair_key in fx_data:
            return fx_data[pair_key]
        else:
            print(
                f"Warning: No exchange rate data for {pair_key}, generating synthetic data"
            )
            return self.generate_synthetic_exchange_rate(end_year - start_year + 1)

    def load_imf_multivariate_data(
        self,
        country: str = "US",
        indicators: List[str] | None = None,
        start_year: int = 2000,
        end_year: int = 2023,
        target_indicator: str | None = None,
    ) -> Dict[str, List[float]]:
        """
        Load multiple IMF indicators for a country

        Args:
            country: Country code
            indicators: List of indicator names
            start_year: Start year for data
            end_year: End year for data
            target_indicator: Primary indicator to forecast (others as features)

        Returns:
            Dictionary with indicator names as keys and time series as values
        """
        if indicators is None:
            indicators = ["gdp_growth", "inflation"]  # Default indicators

        imf_loader = self._get_imf_loader()

        try:
            df = imf_loader.create_multivariate_dataset(
                country, indicators, start_year, end_year
            )

            if df.empty:
                print(
                    f"Warning: No multivariate data for {country}, using synthetic data"
                )
                result = {}
                for indicator in indicators:
                    if indicator == "gdp_growth":
                        result[indicator] = self.generate_synthetic_gdp_growth(
                            end_year - start_year + 1
                        )
                    elif indicator == "inflation":
                        result[indicator] = self.generate_synthetic_inflation(
                            end_year - start_year + 1
                        )
                    else:
                        result[indicator] = self.generate_synthetic_economic_indicator(
                            end_year - start_year + 1
                        )
                return result

            # Convert DataFrame to dictionary of lists
            result = {}
            for col in df.columns:
                if col != "year":
                    result[col] = df[col].tolist()

            return result

        except Exception as e:
            print(f"Error loading multivariate data: {e}")
            # Fallback to synthetic data
            result = {}
            for indicator in indicators:
                if indicator == "gdp_growth":
                    result[indicator] = self.generate_synthetic_gdp_growth(
                        end_year - start_year + 1
                    )
                elif indicator == "inflation":
                    result[indicator] = self.generate_synthetic_inflation(
                        end_year - start_year + 1
                    )
                else:
                    result[indicator] = self.generate_synthetic_economic_indicator(
                        end_year - start_year + 1
                    )
            return result

    # Legacy behavioral data methods (from original script)
    @staticmethod
    def load_sentiment_data(
        file_path: str | None = None, format: str = "csv"
    ) -> List[float] | None:
        """Load sentiment score data from various formats"""
        if file_path is None:
            return EnhancedBehavioralDataLoader.generate_synthetic_sentiment(500)

        if format.lower() == "csv":
            df = pd.read_csv(file_path)
            sentiment_cols = ["sentiment", "sentiment_score", "score", "rating"]
            for col in sentiment_cols:
                if col in df.columns:
                    return df[col].tolist()
            raise ValueError(
                f"No sentiment column found. Available columns: {df.columns.tolist()}"
            )

        elif format.lower() == "json":
            with open(file_path, "r") as f:
                data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                if isinstance(data[0], dict):
                    sentiment_keys = ["sentiment", "sentiment_score", "score"]
                    for key in sentiment_keys:
                        if key in data[0]:
                            return [item[key] for item in data]
                elif isinstance(data[0], (int, float)):
                    return data
            raise ValueError("Could not extract sentiment scores from JSON data")

    @staticmethod
    def load_clickstream_data(file_path: str | None = None) -> List[float]:
        """Convert clickstream data to numeric sequence"""
        if file_path is None:
            return EnhancedBehavioralDataLoader.generate_synthetic_clickstream(300)

        df = pd.read_csv(file_path)

        if "event_type" in df.columns:
            event_mapping = {
                "click": 1.0,
                "scroll": 0.5,
                "hover": 0.3,
                "page_view": 0.8,
                "purchase": 2.0,
                "add_to_cart": 1.5,
                "search": 0.7,
                "exit": 0.0,
            }

            numeric_sequence = []
            for event in df["event_type"]:
                numeric_sequence.append(event_mapping.get(event.lower(), 0.5))

            return numeric_sequence

        raise ValueError("No 'event_type' column found in clickstream data")

    @staticmethod
    def load_engagement_data(file_path: str | None = None) -> List[float]:
        """Load user engagement metrics"""
        if file_path is None:
            return EnhancedBehavioralDataLoader.generate_synthetic_engagement(400)

        df = pd.read_csv(file_path)
        engagement_cols = ["likes", "shares", "comments", "views", "engagement_rate"]
        available_cols = [col for col in engagement_cols if col in df.columns]

        if not available_cols:
            raise ValueError(
                f"No engagement columns found. Available: {df.columns.tolist()}"
            )

        engagement_scores = []
        for _, row in df.iterrows():
            score = sum(row[col] for col in available_cols if pd.notna(row[col]))
            engagement_scores.append(score)

        # Normalize to [0, 1] range
        min_score, max_score = min(engagement_scores), max(engagement_scores)
        if max_score > min_score:
            engagement_scores = [
                (score - min_score) / (max_score - min_score)
                for score in engagement_scores
            ]

        return engagement_scores

    # Enhanced synthetic data generators
    @staticmethod
    def generate_synthetic_sentiment(n_samples: int = 500) -> List[float]:
        """Generate synthetic sentiment score data"""
        np.random.seed(42)
        t = np.linspace(0, 4 * np.pi, n_samples)

        sentiment = (
            0.5  # Baseline neutral sentiment
            + 0.2 * np.sin(t)  # Daily pattern
            + 0.1 * np.sin(5 * t)  # Shorter cycles
            + 0.05 * np.cumsum(np.random.randn(n_samples)) / n_samples  # Random walk
            + 0.1 * np.random.randn(n_samples)  # Noise
        )

        sentiment = np.clip(sentiment, 0, 1)
        return sentiment.tolist()

    @staticmethod
    def generate_synthetic_clickstream(n_samples: int = 300) -> List[float]:
        """Generate synthetic clickstream data"""
        np.random.seed(123)
        events = []
        session_length = np.random.poisson(20, n_samples // 20)

        for length in session_length:
            session = [0.8]  # Start with page view

            for _ in range(length - 1):
                last_event = session[-1]

                if last_event == 0.8:  # After page view
                    next_event = np.random.choice([1.0, 0.5, 0.7], p=[0.4, 0.4, 0.2])
                elif last_event == 1.0:  # After click
                    next_event = np.random.choice([0.8, 1.5, 0.0], p=[0.5, 0.3, 0.2])
                else:  # Other events
                    next_event = np.random.choice([1.0, 0.5, 0.0], p=[0.5, 0.3, 0.2])

                session.append(next_event)

            events.extend(session)
            if len(events) >= n_samples:
                break

        return events[:n_samples]

    @staticmethod
    def generate_synthetic_engagement(n_samples: int = 400) -> List[float]:
        """Generate synthetic engagement data"""
        np.random.seed(456)
        engagement = []
        current_level = 0.3

        for i in range(n_samples):
            change = np.random.normal(0, 0.02)

            # Occasional viral spike
            if np.random.random() < 0.05:
                change += np.random.exponential(0.3)

            current_level += change
            current_level = max(0, min(1, current_level))
            engagement.append(current_level)

        return engagement

    @staticmethod
    def generate_synthetic_inflation(n_samples: int = 24) -> List[float]:
        """Generate realistic synthetic inflation data"""
        np.random.seed(42)

        # Generate inflation with cycles and shocks
        time_trend = np.linspace(0, 4 * np.pi, n_samples)
        cyclical = 1.0 * np.sin(time_trend) + 0.5 * np.sin(2 * time_trend)

        # Add occasional inflation shocks
        shocks = np.zeros(n_samples)
        shock_years = np.random.choice(
            n_samples, size=max(1, n_samples // 8), replace=False
        )
        shocks[shock_years] = np.random.normal(0, 3, len(shock_years))

        base_inflation = 2.5  # Target inflation
        inflation = (
            base_inflation + cyclical + shocks + 0.5 * np.random.randn(n_samples)
        )
        inflation = np.maximum(inflation, -2.0)  # Prevent extreme deflation

        return inflation.tolist()

    @staticmethod
    def generate_synthetic_gdp_growth(n_samples: int = 24) -> List[float]:
        """Generate realistic synthetic GDP growth data"""
        np.random.seed(123)

        # Generate growth with business cycles and recessions
        time_trend = np.linspace(0, 2 * np.pi, n_samples)
        business_cycle = 1.5 * np.sin(time_trend)

        # Add recession shocks
        recession_prob = 0.1  # 10% chance per year
        recessions = np.random.random(n_samples) < recession_prob
        recession_impact = np.where(recessions, np.random.normal(-4, 2, n_samples), 0)

        base_growth = 2.5  # Long-term growth rate
        growth = (
            base_growth
            + business_cycle
            + recession_impact
            + 0.8 * np.random.randn(n_samples)
        )

        return growth.tolist()

    @staticmethod
    def generate_synthetic_exchange_rate(n_samples: int = 24) -> List[float]:
        """Generate realistic synthetic exchange rate data"""
        np.random.seed(789)

        base_rate = 1.1  # EUR/USD approximate

        # Generate with trend and volatility
        time_trend = np.linspace(0, 2 * np.pi, n_samples)
        trend = 0.02 * np.sin(time_trend)  # Cyclical trend
        volatility = 0.1 * np.random.randn(n_samples)  # Random volatility

        rates = base_rate * (1 + trend + volatility)
        rates = np.maximum(rates, 0.01)  # Ensure positive rates

        return rates.tolist()

    @staticmethod
    def generate_synthetic_economic_indicator(n_samples: int = 24) -> List[float]:
        """Generate generic economic indicator data"""
        np.random.seed(456)

        # Generate with trend and cycles
        time_trend = np.linspace(0, 3 * np.pi, n_samples)
        trend = 0.1 * np.sin(time_trend)
        noise = 0.2 * np.random.randn(n_samples)

        base_value = 50  # Neutral level
        indicator = base_value + 10 * trend + 5 * noise

        return indicator.tolist()


def run_imf_data_demo():
    """
    Demonstrate IMF data loading and forecasting
    """
    print("=" * 60)
    print("IMF MACROECONOMIC DATA FORECASTING DEMO")
    print("=" * 60)

    # Initialize enhanced data loader
    loader = EnhancedBehavioralDataLoader()

    # Test different IMF data types
    datasets = {}

    print("\n1. Loading US Inflation Data...")
    try:
        inflation_data = loader.load_imf_inflation_data(["US"], 2010, 2023, "US")
        datasets["us_inflation"] = inflation_data
        print(f"   ✓ Loaded {len(inflation_data)} inflation data points")
        print(f"   Range: [{min(inflation_data):.2f}%, {max(inflation_data):.2f}%]")
    except Exception as e:
        print(f"   ✗ Error loading inflation data: {e}")

    print("\n2. Loading US GDP Growth Data...")
    try:
        gdp_data = loader.load_imf_gdp_growth_data(["US"], 2010, 2023, "US")
        datasets["us_gdp_growth"] = gdp_data
        print(f"   ✓ Loaded {len(gdp_data)} GDP growth data points")
        print(f"   Range: [{min(gdp_data):.2f}%, {max(gdp_data):.2f}%]")
    except Exception as e:
        print(f"   ✗ Error loading GDP data: {e}")

    print("\n3. Loading USD/EUR Exchange Rate...")
    try:
        fx_data = loader.load_imf_exchange_rate_data("USD", "EUR", 2010, 2023)
        datasets["usd_eur"] = fx_data
        print(f"   ✓ Loaded {len(fx_data)} exchange rate data points")
        print(f"   Range: [{min(fx_data):.4f}, {max(fx_data):.4f}]")
    except Exception as e:
        print(f"   ✗ Error loading exchange rate data: {e}")

    print("\n4. Loading Multivariate Economic Data...")
    try:
        mv_data = loader.load_imf_multivariate_data(
            "US", ["gdp_growth", "inflation"], 2010, 2023
        )
        print(f"   ✓ Loaded multivariate data with indicators: {list(mv_data.keys())}")
        for indicator, data in mv_data.items():
            print(
                f"     {indicator}: {len(data)} points, range: [{min(data):.2f}, {max(data):.2f}]"
            )
        datasets.update(mv_data)
    except Exception as e:
        print(f"   ✗ Error loading multivariate data: {e}")

    print(f"\n5. Summary: Loaded {len(datasets)} datasets for forecasting")

    # Run forecasting on one dataset
    if datasets:
        print("\n6. Running Chronos Forecasting on Inflation Data...")
        try:
            from chronos_behavioral_framework import (
                ChronosBehavioralForecaster,
                BenchmarkRunner,
            )

            # Use inflation data if available, otherwise first available dataset
            forecast_data = datasets.get("us_inflation", list(datasets.values())[0])

            forecaster = ChronosBehavioralForecaster(
                model_name="amazon/chronos-bolt-small", device="cpu"
            )

            benchmark = BenchmarkRunner(forecaster)
            results = benchmark.run_benchmark(
                data=forecast_data, test_split=0.3, prediction_length=3, window_size=8
            )

            benchmark.print_results()

            print("\n✓ IMF Data Forecasting Demo Completed Successfully!")
            return results

        except Exception as e:
            print(f"   ✗ Error running forecasting: {e}")

    return None


if __name__ == "__main__":
    # Run the IMF data demo
    results = run_imf_data_demo()

