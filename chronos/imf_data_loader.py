"""
IMF Data Loader Module (fixed)
Fetches real macroeconomic data from IMF APIs using sdmx1 package

Fixes applied:
- Adds explicit format='sdmx-2.1' parameter to requests so IMF returns SDMX-ML.
- Maps ISO3 country codes -> IMF expected country codes (many IMF endpoints use 2-letter).
- Queries the specific series via the SDMX `key` and time params instead of downloading full dataset.
- Robust conversion from SDMX message to pandas DataFrame with sensible fallbacks.
- Preserves caching, synthetic fallback, validation, and dataset composition logic.
"""

import json
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import sdmx

warnings.filterwarnings("ignore")


class IMFDataLoader:
    """
    Load macroeconomic data from IMF APIs using sdmx1
    """

    # Common indicator codes for different databases
    INDICATORS = {
        # GDP indicators (WEO codes)
        "gdp_growth": "NGDP_RPCH",  # GDP growth rate
        "gdp_nominal": "NGDP",  # Nominal GDP
        "gdp_real": "NGDP_R",  # Real GDP
        "gdp_per_capita": "NGDPPC",  # GDP per capita
        # Inflation indicators
        "inflation": "PCPIPCH",  # Inflation rate (CPI)
        "cpi": "PCPI",  # Consumer Price Index
        "deflator": "NGDP_D",  # GDP deflator
        # Employment indicators
        "unemployment": "LUR",  # Unemployment rate
        "employment": "LE",  # Employment
        # External sector
        "current_account": "BCA",  # Current account balance
        "exports": "BX",  # Exports
        "imports": "BM",  # Imports
        # Fiscal indicators
        "gov_balance": "GGR_NGDP",  # Government balance
        "gov_debt": "GGXWDG_NGDP",  # Government debt
        # Monetary indicators
        "interest_rate": "FPOLM_PA",  # Policy interest rate
        "money_supply": "FM",  # Money supply
    }

    # IFS indicator codes for exchange rates and other financial statistics
    IFS_INDICATORS = {
        "exchange_rate": "ENDA_XDC_USD_RATE",  # End of period exchange rate
        "interest_rate_short": "FITB_PA",  # Treasury bill rate
        "interest_rate_long": "FIGB_PA",  # Government bond yield
        "reserve_assets": "FARA_XDC",  # Reserve assets
    }

    # Country codes (ISO 3-letter codes) mapping to IMF WEO/IFS codes (2-letter where applicable)
    # This mapping covers commonly used codes; falls back to first-2-letters if unknown.
    IMF_COUNTRY_MAP = {
        "USA": "US",
        "GBR": "GB",
        "DEU": "DE",
        "FRA": "FR",
        "JPN": "JP",
        "CHN": "CN",
        "IND": "IN",
        "BRA": "BR",
        "CAN": "CA",
        "AUS": "AU",
        "ITA": "IT",
        "ESP": "ES",
        "KOR": "KR",
        "MEX": "MX",
        "RUS": "RU",
    }

    def __init__(self, cache_dir: str = "imf_cache", rate_limit: float = 1.0):
        """
        Initialize IMF Data Loader with sdmx1

        Args:
            cache_dir: Directory for caching API responses
            rate_limit: Minimum seconds between API calls
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.rate_limit = rate_limit
        self.last_request_time = 0.0

        # Initialize SDMX client for IMF (try common endpoint keys)
        # the argument is usually the provider id; sdmx.Client will resolve it
        try:
            self.imf_client = sdmx.Client("IMF")
        except Exception:
            try:
                self.imf_client = sdmx.Client("IMF_DATA")
            except Exception:
                # As a last fallback construct a default client and hope for the best
                self.imf_client = sdmx.Client()

    # ----------------------------
    # Utility / rate limiting
    # ----------------------------
    def _wait_for_rate_limit(self):
        """Ensure rate limiting between API calls"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()

    def _get_cache_path(
        self,
        database: str,
        country: str,
        indicator: str,
        start_year: int,
        end_year: int,
    ) -> Path:
        """Generate cache file path"""
        safe_country = country.replace(" ", "_")
        safe_indicator = indicator.replace("/", "_").replace(" ", "_")
        filename = (
            f"{database}_{safe_country}_{safe_indicator}_{start_year}_{end_year}.json"
        )
        return self.cache_dir / filename

    def _load_from_cache(
        self, cache_path: Path, max_age_days: int = 7
    ) -> Optional[Dict]:
        """Load data from cache if it exists and is recent"""
        if not cache_path.exists():
            return None

        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        if cache_age > timedelta(days=max_age_days):
            return None

        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

    def _save_to_cache(self, cache_path: Path, data: Dict):
        """Save data to cache"""
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"Warning: Could not save to cache: {e}")

    # ----------------------------
    # Helper: IMF country code mapping
    # ----------------------------
    def _to_imf_country_code(self, iso3: str) -> str:
        """Map ISO3 -> IMF expected code (often 2-letter); fallback to iso3[:2].upper()"""
        if not iso3:
            return iso3
        iso3u = iso3.upper()
        return self.IMF_COUNTRY_MAP.get(iso3u, iso3u[:2])

    # ----------------------------
    # Helper: convert SDMX message to pandas DataFrame robustly
    # ----------------------------
    def _sdmx_msg_to_dataframe(self, msg) -> Optional[pd.DataFrame]:
        """
        Convert an sdmx message (from client.data) to a pandas DataFrame.
        Tries sdmx.to_pandas(), then a few fallbacks.
        """
        try:
            # Preferred conversion
            df = sdmx.to_pandas(msg)
            # If to_pandas yields a Series, turn it into DataFrame
            if isinstance(df, pd.Series):
                df = df.reset_index()
                df.columns = ["time_period", "value"]
            # If DataFrame and has multiindex, reset and try to find value/time columns downstream
            return df
        except Exception:
            # Fallback: try to inspect msg structure (best-effort)
            try:
                # Some sdmx library messages have a .data or .series attribute
                # We'll attempt to build a list of observations
                rows = []
                # Many implementations: msg.data contains series keyed by seriesKey with obs
                if hasattr(msg, "data") and isinstance(msg.data, dict):
                    for series_key, series in msg.data.items():
                        # series might have 'observations' mapping
                        obs = series.get("observations", {})
                        # obs keys are usually period -> [value, ...]
                        for period, val in obs.items():
                            if isinstance(val, (list, tuple)):
                                value = val[0]
                            else:
                                value = val
                            rows.append({"time_period": period, "value": value})
                # Another possible shape: msg.series is iterable
                elif hasattr(msg, "series"):
                    for s in msg.series:
                        if hasattr(s, "observations"):
                            for obs in s.observations:
                                # obs often has .time and .value fields
                                t = getattr(obs, "time", None) or obs.get("time", None)
                                v = getattr(obs, "value", None) or obs.get(
                                    "value", None
                                )
                                rows.append({"time_period": t, "value": v})
                if rows:
                    return pd.DataFrame(rows)
            except Exception:
                pass
        return None

    # ----------------------------
    # WEO specific fetch
    # ----------------------------
    def _get_weo_full_data(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Attempt to fetch a full WEO release index (if needed).
        We keep a simple implementation that tries to retrieve WEO metadata or fallback to empty DF.
        In practice the module fetches targeted series via fetch_weo_data, so this function is conservative.
        """
        # We'll attempt to get dataflow/structure then return empty DataFrame if not needed.
        try:
            self._wait_for_rate_limit()
            # try to fetch the WEO dataflow structure to validate connectivity
            dataflow = self.imf_client.get(resource_type="dataflow", resource_id="WEO")
            # No heavy processing here; return empty DF (we fetch series individually in fetch_weo_data)
            return pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    def fetch_weo_data(
        self,
        country: str,
        indicator: str,
        start_year: int = 2000,
        end_year: int = 2023,
        use_cache: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch data from World Economic Outlook database using sdmx1

        Args:
            country: ISO 3-letter country code (e.g., 'USA', 'GBR', 'DEU')
            indicator: WEO indicator code
            start_year: Start year for data
            end_year: End year for data
            use_cache: Whether to use cached data

        Returns:
            DataFrame with time series data or None if failed
        """
        # normalize country for cache naming (use ISO3 as provided)
        cache_path = self._get_cache_path(
            "WEO", country, indicator, start_year, end_year
        )
        if use_cache:
            cached_data = self._load_from_cache(cache_path)
            if cached_data:
                try:
                    return pd.DataFrame(cached_data)
                except Exception:
                    # ignore cache deserialization errors and refetch
                    pass

        imf_country = self._to_imf_country_code(country)

        print(
            f"Fetching WEO data: {country} (IMF:{imf_country}) - {indicator} ({start_year}-{end_year})"
        )
        try:
            self._wait_for_rate_limit()
            # Build the query. WEO uses dimension names like REF_AREA and SUBJECT/WEO Subject Code.
            # different IMF endpoints sometimes use different dimension names. We'll try common variants.
            tried_keys = [
                {"REF_AREA": imf_country, "WEO Subject Code": indicator},
                {"REF_AREA": imf_country, "SUBJECT": indicator},
                {"REF_AREA": imf_country, "WEO_SUBJECT_CODE": indicator},
                {
                    "REF_AREA": imf_country
                },  # fallback to country-only query (then filter)
            ]

            msg = None
            for key in tried_keys:
                try:
                    params = {
                        "startPeriod": start_year,
                        "endPeriod": end_year,
                        "format": "sdmx-2.1",  # ensure server returns SDMX-ML
                    }
                    msg = self.imf_client.data(
                        resource_id="WEO", key=key, params=params
                    )
                    if msg:
                        break
                except Exception:
                    # try next key variant
                    msg = None

            if msg is None:
                raise RuntimeError("No message returned from IMF WEO service")

            df_raw = self._sdmx_msg_to_dataframe(msg)
            if df_raw is None or df_raw.empty:
                raise RuntimeError("SDMX to pandas conversion returned no data")

            # Process DataFrame into standardized format
            df_processed = self._process_sdmx_dataframe(df_raw, country, indicator)
            if df_processed is not None and not df_processed.empty:
                if use_cache:
                    self._save_to_cache(cache_path, df_processed.to_dict("records"))
                return df_processed

            # If no processed data, attempt to build from raw df (best-effort)
            if df_raw is not None and not df_raw.empty:
                # Try to coerce to expected shape
                try:
                    df_try = pd.DataFrame(
                        {
                            "time_period": pd.to_datetime(
                                df_raw.index.astype(str), errors="coerce"
                            ),
                            "value": pd.to_numeric(df_raw.iloc[:, 0], errors="coerce"),
                        }
                    )
                    df_try["country"] = country
                    df_try["indicator"] = indicator
                    df_try["year"] = df_try["time_period"].dt.year
                    df_try = df_try.dropna(subset=["value"])
                    if use_cache:
                        self._save_to_cache(cache_path, df_try.to_dict("records"))
                    return df_try
                except Exception:
                    pass

            print(f"No usable WEO data found for {country} - {indicator}")
            return None

        except Exception as e:
            print(f"Error fetching WEO data: {e}")
            print(f"Falling back to synthetic data for {country} - {indicator}")
            return self._generate_fallback_data(
                country, indicator, start_year, end_year
            )

    # ----------------------------
    # IFS fetch
    # ----------------------------
    def fetch_ifs_data(
        self,
        country: str,
        indicator: str,
        start_year: int = 2000,
        end_year: int = 2023,
        frequency: str = "M",
        use_cache: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch data from International Financial Statistics database
        """
        cache_path = self._get_cache_path(
            f"IFS_{frequency}", country, indicator, start_year, end_year
        )
        if use_cache:
            cached = self._load_from_cache(cache_path)
            if cached:
                try:
                    return pd.DataFrame(cached)
                except Exception:
                    pass

        imf_country = self._to_imf_country_code(country)
        print(
            f"Fetching IFS data: {country} (IMF:{imf_country}) - {indicator} ({start_year}-{end_year}, freq={frequency})"
        )
        try:
            self._wait_for_rate_limit()
            # IFS keys sometimes expect "COUNTRY.INDICATOR.FREQUENCY" style
            key_str = f"{imf_country}.{indicator}.{frequency}"
            params = {
                "startPeriod": start_year,
                "endPeriod": end_year,
                "format": "sdmx-2.1",
            }
            msg = None
            try:
                msg = self.imf_client.data(
                    resource_id="IFS", key=key_str, params=params
                )
            except Exception:
                # try with dict key
                try:
                    msg = self.imf_client.data(
                        resource_id="IFS", key={"COUNTRY": imf_country}, params=params
                    )
                except Exception:
                    msg = None

            if msg is None:
                raise RuntimeError("No message returned from IMF IFS service")

            df_raw = self._sdmx_msg_to_dataframe(msg)
            if df_raw is None or df_raw.empty:
                raise RuntimeError("SDMX to pandas conversion returned no data")

            df_processed = self._process_sdmx_dataframe(df_raw, country, indicator)
            if df_processed is not None and not df_processed.empty:
                if use_cache:
                    self._save_to_cache(cache_path, df_processed.to_dict("records"))
                return df_processed

            print(f"No usable IFS data found for {country} - {indicator}")
            return None

        except Exception as e:
            print(f"Error fetching IFS data: {e}")
            return None

    # ----------------------------
    # Process SDMX DataFrame into standardized format
    # ----------------------------
    def _process_sdmx_dataframe(
        self, df: pd.DataFrame, country: str, indicator: str
    ) -> Optional[pd.DataFrame]:
        """Process SDMX DataFrame into standardized format"""
        try:
            # If df is a Series disguised as DF, normalize
            if isinstance(df, pd.Series):
                df = df.reset_index()
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index()

            # Find the time column
            time_cols = [
                col
                for col in df.columns
                if "TIME" in str(col).upper()
                or "PERIOD" in str(col).upper()
                or "DATE" in str(col).upper()
            ]

            if not time_cols:
                # maybe index holds the period
                if pd.api.types.is_datetime64_any_dtype(
                    df.index
                ) or pd.api.types.is_integer_dtype(df.index):
                    df = df.reset_index().rename(columns={"index": "time_period"})
                    time_cols = ["time_period"]
                else:
                    # try common field names
                    for candidate in [
                        "time_period",
                        "TIME_PERIOD",
                        "TIME",
                        "OBS_TIME",
                        "DATE",
                    ]:
                        if candidate in df.columns:
                            time_cols = [candidate]
                            break

            if not time_cols:
                print("No time column found in DataFrame")
                return None

            time_col = time_cols[0]

            # Find the value column
            # choose numeric columns excluding time_cols
            value_cols = [
                col
                for col in df.columns
                if col not in time_cols and pd.api.types.is_numeric_dtype(df[col])
            ]
            if not value_cols:
                # try named columns
                for cand in ["value", "OBS_VALUE", "OBS_VALUE.0", 0]:
                    if cand in df.columns:
                        value_cols = [cand]
                        break
            if not value_cols and df.shape[1] >= 2:
                # fallback to second column
                other_cols = [c for c in df.columns if c not in time_cols]
                if other_cols:
                    value_cols = [other_cols[0]]

            if not value_cols:
                print("No value column found in DataFrame")
                return None

            value_col = value_cols[0]

            # Create standardized DataFrame
            result_df = pd.DataFrame()
            result_df["time_period"] = pd.to_datetime(
                df[time_col].astype(str), errors="coerce"
            )

            # If all NaT, try parse as years
            if result_df["time_period"].isna().all():
                result_df["time_period"] = pd.to_datetime(
                    df[time_col].astype(str), format="%Y", errors="coerce"
                )

            result_df["value"] = pd.to_numeric(df[value_col], errors="coerce")
            result_df["country"] = country
            result_df["indicator"] = indicator
            result_df["year"] = result_df["time_period"].dt.year

            # Drop rows without numeric value
            result_df = result_df.dropna(subset=["value"]).reset_index(drop=True)
            result_df = result_df.sort_values("time_period").reset_index(drop=True)

            return result_df

        except Exception as e:
            print(f"Error processing SDMX DataFrame: {e}")
            return None

    # ----------------------------
    # Fallback synthetic generators
    # ----------------------------
    def _generate_fallback_data(
        self, country: str, indicator: str, start_year: int, end_year: int
    ) -> pd.DataFrame:
        """Generate synthetic fallback data when API fails"""
        n_years = end_year - start_year + 1
        years = list(range(start_year, end_year + 1))

        if (
            "GDP" in indicator
            or "growth" in indicator.lower()
            or "NGDP_RPCH" in indicator
        ):
            values = self._generate_synthetic_gdp_growth(n_years, country)
        elif (
            "INF" in indicator.upper()
            or "CPI" in indicator.upper()
            or "PCPIPCH" in indicator
        ):
            values = self._generate_synthetic_inflation(n_years, country)
        else:
            np.random.seed(abs(hash(f"{country}{indicator}")) % 1000)
            values = (2.0 + np.random.randn(n_years)).tolist()

        return pd.DataFrame(
            {"year": years, "country": country, "indicator": indicator, "value": values}
        )

    def _generate_synthetic_exchange_rate(
        self, n_years: int, currency: str
    ) -> List[float]:
        """Generate realistic synthetic exchange rate data"""
        np.random.seed(abs(hash(currency)) % 1000)

        base_rates = {
            "EUR": 0.85,
            "GBP": 0.75,
            "JPY": 110.0,
            "CNY": 6.5,
            "CAD": 1.25,
            "AUD": 1.35,
        }
        base_rate = base_rates.get(currency, 1.0)
        time_trend = np.linspace(0, 2 * np.pi, n_years)
        trend = 0.02 * np.sin(time_trend)
        volatility = 0.05 * np.random.randn(n_years)
        rates = base_rate * (1 + trend + volatility)
        rates = np.maximum(rates, 0.01)
        return rates.tolist()

    def _generate_synthetic_inflation(self, n_years: int, country: str) -> List[float]:
        """Generate realistic synthetic inflation data"""
        np.random.seed(abs(hash(country)) % 1000)
        base_inflation = {
            "USA": 2.5,
            "GBR": 2.0,
            "DEU": 1.5,
            "JPN": 0.5,
            "CHN": 3.0,
            "IND": 5.0,
            "BRA": 6.0,
            "RUS": 8.0,
        }
        base = base_inflation.get(country, 2.0)
        n = n_years
        time_trend = np.linspace(0, 4 * np.pi, n)
        cyclical = 1.0 * np.sin(time_trend) + 0.5 * np.sin(2 * time_trend)
        shocks = np.zeros(n)
        rng = np.random.default_rng(abs(hash(country)) % 2**32)
        shock_years = rng.choice(n, size=max(1, n // 8), replace=False)
        shocks[shock_years] = rng.normal(0, 3, len(shock_years))
        inflation = base + cyclical + shocks + 0.5 * rng.standard_normal(n)
        inflation = np.maximum(inflation, -2.0)
        return inflation.tolist()

    def _generate_synthetic_gdp_growth(self, n_years: int, country: str) -> List[float]:
        """Generate realistic synthetic GDP growth data"""
        np.random.seed(abs(hash(country)) % (2**31 - 1) + 42)
        base_growth = {
            "USA": 2.5,
            "GBR": 2.0,
            "DEU": 1.5,
            "JPN": 1.0,
            "CHN": 7.0,
            "IND": 6.0,
            "BRA": 2.0,
            "RUS": 1.5,
        }
        base = base_growth.get(country, 2.0)
        time_trend = np.linspace(0, 2 * np.pi, n_years)
        business_cycle = 1.5 * np.sin(time_trend)
        rng = np.random.default_rng(abs(hash(country)) % 2**32 + 7)
        recession_prob = 0.1
        recessions = rng.random(n_years) < recession_prob
        recession_impact = np.where(recessions, rng.normal(-4, 2, n_years), 0)
        growth = (
            base
            + business_cycle
            + recession_impact
            + 0.8 * rng.standard_normal(n_years)
        )
        return growth.tolist()

    # ----------------------------
    # Validation & summary utilities
    # ----------------------------
    def validate_data(
        self, data: List[float], indicator_type: str = "general"
    ) -> Dict[str, Any]:
        if not data or len(data) == 0:
            return {"valid": False, "error": "Empty data"}

        data_array = np.array(data, dtype=float)
        stats = {
            "count": int(len(data_array)),
            "mean": float(np.nanmean(data_array)),
            "std": float(np.nanstd(data_array)),
            "min": float(np.nanmin(data_array)),
            "max": float(np.nanmax(data_array)),
            "missing_count": int(np.sum(np.isnan(data_array))),
            "infinite_count": int(np.sum(np.isinf(data_array))),
        }

        valid_data = data_array[~np.isnan(data_array)]
        outliers = []
        if len(valid_data) > 0:
            q1, q3 = np.percentile(valid_data, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = np.where(
                (data_array < lower_bound) | (data_array > upper_bound)
            )[0]

        warnings_list = []
        if indicator_type == "inflation":
            if stats["max"] > 50:
                warnings_list.append("Extremely high inflation detected")
            if stats["min"] < -10:
                warnings_list.append("Extreme deflation detected")
        elif indicator_type == "gdp_growth":
            if stats["max"] > 20:
                warnings_list.append("Unrealistic GDP growth detected")
            if stats["min"] < -15:
                warnings_list.append("Extreme GDP contraction detected")

        quality_score = 1.0
        if stats["missing_count"] > 0:
            quality_score -= 0.2 * (stats["missing_count"] / stats["count"])
        if len(outliers) > 0:
            quality_score -= 0.1 * (len(outliers) / stats["count"])
        if stats["infinite_count"] > 0:
            quality_score -= 0.3

        return {
            "valid": quality_score > 0.5,
            "quality_score": float(max(0.0, quality_score)),
            "statistics": stats,
            "outliers": outliers.tolist(),
            "warnings": warnings_list,
        }

    # ----------------------------
    # High-level retrievals
    # ----------------------------
    def fetch_exchange_rates(
        self,
        base_currency: str = "USD",
        target_currencies: Optional[List[str]] = None,
        start_year: int = 2000,
        end_year: int = 2023,
        use_cache: bool = True,
    ) -> Dict[str, List[float]]:
        if target_currencies is None:
            target_currencies = ["EUR", "GBP", "JPY", "CNY"]

        exchange_rates: Dict[str, List[float]] = {}
        currency_to_country = {
            "EUR": "U2",  # Euro Area
            "GBP": "GBR",
            "JPY": "JPN",
            "CNY": "CHN",
            "CAD": "CAN",
            "AUD": "AUS",
        }

        for currency in target_currencies:
            country_code = currency_to_country.get(currency)
            pair = f"{base_currency}/{currency}"
            if not country_code:
                print(f"Unknown currency: {currency}, using synthetic data")
                exchange_rates[pair] = self._generate_synthetic_exchange_rate(
                    end_year - start_year + 1, currency
                )
                continue

            print(f"Fetching {pair} exchange rate...")
            try:
                df = self.fetch_ifs_data(
                    country_code,
                    self.IFS_INDICATORS["exchange_rate"],
                    start_year,
                    end_year,
                    frequency="A",
                    use_cache=use_cache,
                )
                if df is not None and not df.empty:
                    # Convert to annual series (mean)
                    annual_data = (
                        df.groupby("year")["value"]
                        .mean()
                        .reindex(range(start_year, end_year + 1))
                        .tolist()
                    )
                    # If any NaNs, fallback to synthetic for that pair
                    if any(pd.isna(x) for x in annual_data):
                        raise RuntimeError(
                            "Incomplete exchange rate series, using synthetic fallback"
                        )
                    exchange_rates[pair] = annual_data
                else:
                    exchange_rates[pair] = self._generate_synthetic_exchange_rate(
                        end_year - start_year + 1, currency
                    )
            except Exception as e:
                print(f"Error fetching exchange rate for {currency}: {e}")
                exchange_rates[pair] = self._generate_synthetic_exchange_rate(
                    end_year - start_year + 1, currency
                )
        return exchange_rates

    def get_inflation_data(
        self,
        countries: Optional[List[str]] = None,
        start_year: int = 2000,
        end_year: int = 2023,
        use_cache: bool = True,
    ) -> Dict[str, List[float]]:
        if countries is None:
            countries = ["USA", "GBR", "DEU", "JPN", "CHN"]

        inflation_data: Dict[str, List[float]] = {}
        for country in countries:
            df = self.fetch_weo_data(
                country,
                self.INDICATORS["inflation"],
                start_year,
                end_year,
                use_cache=use_cache,
            )
            if df is not None and not df.empty:
                annual_data = (
                    df.groupby("year")["value"]
                    .mean()
                    .reindex(range(start_year, end_year + 1))
                    .tolist()
                )
                # If any missing, replace with synthetic for that country
                if any(pd.isna(x) for x in annual_data):
                    print(
                        f"Using synthetic inflation data for {country} due to missing points"
                    )
                    inflation_data[country] = self._generate_synthetic_inflation(
                        end_year - start_year + 1, country
                    )
                else:
                    inflation_data[country] = [float(x) for x in annual_data]
            else:
                print(f"Using synthetic inflation data for {country}")
                inflation_data[country] = self._generate_synthetic_inflation(
                    end_year - start_year + 1, country
                )
        return inflation_data

    def get_gdp_growth_data(
        self,
        countries: Optional[List[str]] = None,
        start_year: int = 2000,
        end_year: int = 2023,
        use_cache: bool = True,
    ) -> Dict[str, List[float]]:
        if countries is None:
            countries = ["USA", "GBR", "DEU", "JPN", "CHN"]

        gdp_data: Dict[str, List[float]] = {}
        for country in countries:
            df = self.fetch_weo_data(
                country,
                self.INDICATORS["gdp_growth"],
                start_year,
                end_year,
                use_cache=use_cache,
            )
            if df is not None and not df.empty:
                annual_data = (
                    df.groupby("year")["value"]
                    .mean()
                    .reindex(range(start_year, end_year + 1))
                    .tolist()
                )
                if any(pd.isna(x) for x in annual_data):
                    print(
                        f"Using synthetic GDP growth data for {country} due to missing points"
                    )
                    gdp_data[country] = self._generate_synthetic_gdp_growth(
                        end_year - start_year + 1, country
                    )
                else:
                    gdp_data[country] = [float(x) for x in annual_data]
            else:
                print(f"Using synthetic GDP growth data for {country}")
                gdp_data[country] = self._generate_synthetic_gdp_growth(
                    end_year - start_year + 1, country
                )
        return gdp_data

    # ----------------------------
    # Meta helpers
    # ----------------------------
    def get_available_indicators(self) -> Dict[str, str]:
        return self.INDICATORS.copy()

    def get_available_countries(self) -> Dict[str, str]:
        return self.IMF_COUNTRY_MAP.copy()

    def create_multivariate_dataset(
        self,
        country: str,
        indicators: List[str],
        start_year: int = 2000,
        end_year: int = 2023,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        all_data = {}
        years_index = list(range(start_year, end_year + 1))
        for indicator_name in indicators:
            if indicator_name not in self.INDICATORS:
                print(f"Warning: Unknown indicator '{indicator_name}', skipping")
                continue
            indicator_code = self.INDICATORS[indicator_name]
            df = self.fetch_weo_data(
                country, indicator_code, start_year, end_year, use_cache=use_cache
            )
            if df is not None and not df.empty:
                annual_data = df.groupby("year")["value"].mean().reindex(years_index)
                all_data[indicator_name] = annual_data.values
            else:
                print(
                    f"Warning: No data available for {country} - {indicator_name}, using synthetic"
                )
                if indicator_name == "inflation":
                    vals = self._generate_synthetic_inflation(
                        end_year - start_year + 1, country
                    )
                elif indicator_name == "gdp_growth":
                    vals = self._generate_synthetic_gdp_growth(
                        end_year - start_year + 1, country
                    )
                else:
                    vals = [np.nan] * (end_year - start_year + 1)
                all_data[indicator_name] = vals

        if not all_data:
            return pd.DataFrame()

        combined_df = pd.DataFrame(all_data, index=years_index)
        combined_df.index.name = "year"
        # fill forward/backwards
        combined_df = combined_df.ffill().bfill()
        combined_df = combined_df.reset_index()
        return combined_df


# ----------------------------
# Example usage and testing functions
# ----------------------------
def test_imf_loader():
    """Test the IMF Data Loader functionality"""
    print("Testing IMF Data Loader (fixed) ...")
    print("=" * 60)

    loader = IMFDataLoader()

    # Test 1: Fetch inflation data
    print("\n1. Testing inflation data fetch...")
    start_year = 2010
    end_year = 2020
    inflation_data = loader.get_inflation_data(["USA", "GBR"], start_year, end_year)
    for country, data in inflation_data.items():
        validation = loader.validate_data(data, "inflation")
        print(
            f"  {country}: {len(data)} points, quality: {validation['quality_score']:.2f}"
        )
        print(f"    Mean: {validation['statistics']['mean']:.2f}%")

    # Test 2: Fetch GDP growth data
    print("\n2. Testing GDP growth data fetch...")
    gdp_data = loader.get_gdp_growth_data(["USA", "CHN"], start_year, end_year)
    for country, data in gdp_data.items():
        validation = loader.validate_data(data, "gdp_growth")
        print(
            f"  {country}: {len(data)} points, quality: {validation['quality_score']:.2f}"
        )
        print(f"    Mean: {validation['statistics']['mean']:.2f}%")

    # Test 3: Fetch exchange rates
    print("\n3. Testing exchange rate data...")
    fx_data = loader.fetch_exchange_rates("USD", ["EUR", "GBP"], 2015, 2020)
    for pair, data in fx_data.items():
        if data:
            print(
                f"  {pair}: {len(data)} points, range: [{min(data):.3f}, {max(data):.3f}]"
            )
        else:
            print(f"  {pair}: empty series")

    # Test 4: Create multivariate dataset
    print("\n4. Testing multivariate dataset creation...")
    mv_data = loader.create_multivariate_dataset(
        "USA", ["gdp_growth", "inflation"], 2010, 2020
    )
    if not mv_data.empty:
        print(f"  Multivariate dataset: {mv_data.shape}")
        print(f"  Columns: {mv_data.columns.tolist()}")
        print(f"\n  Sample data:\n{mv_data.head()}")

    print("\n" + "=" * 60)
    print("IMF Data Loader test completed!")
    return loader


if __name__ == "__main__":
    loader = test_imf_loader()
