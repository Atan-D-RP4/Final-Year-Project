"""
IMF Data Loader Module (revised)
Fetches real macroeconomic data directly from IMF DataMapper JSON API
instead of using the sdmx1 package.

- Uses direct HTTPS JSON endpoints (https://www.imf.org/external/datamapper/api/v1)
- Caches results locally to avoid rate limits
- Preserves fallback synthetic data generation and validation
- Maintains same public API: fetch_weo_data(), fetch_ifs_data(), etc.
"""

import json
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")


class IMFDataLoader:
    """Load macroeconomic data from IMF DataMapper API"""

    BASE_URL = "https://www.imf.org/external/datamapper/api/v1"

    INDICATORS = {
        "gdp_growth": "NGDP_RPCH",
        "gdp_nominal": "NGDP",
        "gdp_real": "NGDP_R",
        "gdp_per_capita": "NGDPPC",
        "inflation": "PCPIPCH",
        "cpi": "PCPI",
        "deflator": "NGDP_D",
        "unemployment": "LUR",
        "employment": "LE",
        "current_account": "BCA",
        "exports": "BX",
        "imports": "BM",
        "gov_balance": "GGR_NGDP",
        "gov_debt": "GGXWDG_NGDP",
        "interest_rate": "FPOLM_PA",
        "money_supply": "FM",
    }

    IFS_INDICATORS = {
        "exchange_rate": "ENDA_XDC_USD_RATE",
        "interest_rate_short": "FITB_PA",
        "interest_rate_long": "FIGB_PA",
        "reserve_assets": "FARA_XDC",
    }

    IMF_COUNTRY_MAP = {
        "US": "USA",
        "GB": "GBR",
        "DE": "DEU",
        "FR": "FRA",
        "JP": "JPN",
        "CN": "CHN",
        "IN": "IND",
        "BR": "BRA",
        "CA": "CAN",
        "AU": "AUS",
        "IT": "ITA",
        "ES": "ESP",
        "KR": "KOR",
        "MX": "MEX",
        "RU": "RUS",
    }

    def __init__(self, cache_dir: str = "imf_cache", rate_limit: float = 1.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.rate_limit = rate_limit
        self.last_request_time = 0.0

    # ------------------------
    # Utility / Cache
    # ------------------------
    def _wait_for_rate_limit(self):
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
        safe_country = country.replace(" ", "_")
        safe_indicator = indicator.replace("/", "_").replace(" ", "_")
        filename = (
            f"{database}_{safe_country}_{safe_indicator}_{start_year}_{end_year}.json"
        )
        return self.cache_dir / filename

    def _load_from_cache(
        self, path: Path, max_age_days: int = 7
    ) -> Optional[pd.DataFrame]:
        if not path.exists():
            return None
        if (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)) > timedelta(
            days=max_age_days
        ):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return pd.DataFrame(data)
        except Exception:
            return None

    def _save_to_cache(self, path: Path, df: pd.DataFrame):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(df.to_dict("records"), f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"Warning: Could not save cache: {e}")

    def _to_imf_country_code(self, iso3: str) -> str:
        if not iso3:
            return iso3
        iso3u = iso3.upper()
        return self.IMF_COUNTRY_MAP.get(iso3u, iso3u[:2])

    # ------------------------
    # WEO Data Fetch via JSON
    # ------------------------
    def fetch_weo_data(
        self,
        country: str,
        indicator: str,
        start_year: int = 2000,
        end_year: int = 2023,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch data from IMF WEO JSON API"""
        cache_path = self._get_cache_path(
            "WEO", country, indicator, start_year, end_year
        )

        if use_cache:
            cached = self._load_from_cache(cache_path)
            if cached is not None:
                return cached

        imf_country = self._to_imf_country_code(country)
        url = f"{self.BASE_URL}/{indicator}/{imf_country}"
        print(f"Fetching WEO data: {country} - {indicator} ({start_year}-{end_year})")

        try:
            self._wait_for_rate_limit()
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}")
            data = resp.json()

            values = data["values"][indicator][imf_country]
            df = pd.DataFrame(
                {"year": list(values.keys()), "value": list(values.values())}
            )
            df["year"] = pd.to_numeric(df["year"], errors="coerce")
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna(subset=["year", "value"])
            df = df[(df["year"] >= start_year) & (df["year"] <= end_year)]
            df["country"] = country
            df["indicator"] = indicator
            df = df.reset_index(drop=True)

            if use_cache:
                self._save_to_cache(cache_path, df)
            return df

        except Exception as e:
            print(f"Error fetching WEO data: {e}")
            print(f"Falling back to synthetic data for {country}-{indicator}")
            return self._generate_fallback_data(
                country, indicator, start_year, end_year
            )

    # ------------------------
    # IFS Fetch (placeholder fallback)
    # ------------------------
    def fetch_ifs_data(
        self,
        country: str,
        indicator: str,
        start_year: int = 2000,
        end_year: int = 2023,
        frequency: str = "M",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Currently no IMF datamapper endpoint for IFS, so fallback."""
        print(f"IFS data not available via JSON API, generating synthetic data.")
        return self._generate_fallback_data(country, indicator, start_year, end_year)

    # ------------------------
    # Synthetic fallback data
    # ------------------------
    def _generate_fallback_data(
        self, country: str, indicator: str, start_year: int, end_year: int
    ) -> pd.DataFrame:
        n_years = end_year - start_year + 1
        years = list(range(start_year, end_year + 1))

        if "GDP" in indicator or "growth" in indicator.lower():
            values = self._generate_synthetic_gdp_growth(n_years, country)
        elif "INF" in indicator.upper() or "CPI" in indicator.upper():
            values = self._generate_synthetic_inflation(n_years, country)
        else:
            np.random.seed(abs(hash(f"{country}{indicator}")) % 1000)
            values = (2.0 + np.random.randn(n_years)).tolist()

        return pd.DataFrame(
            {"year": years, "country": country, "indicator": indicator, "value": values}
        )

    def _generate_synthetic_inflation(self, n_years: int, country: str) -> List[float]:
        np.random.seed(abs(hash(country)) % 1000)
        base_inflation = {"USA": 2.5, "IND": 5.0, "CHN": 3.0}
        base = base_inflation.get(country, 2.0)
        time_trend = np.linspace(0, 4 * np.pi, n_years)
        cyclical = np.sin(time_trend)
        shocks = np.random.normal(0, 0.5, n_years)
        return (base + cyclical + shocks).tolist()

    def _generate_synthetic_gdp_growth(self, n_years: int, country: str) -> List[float]:
        np.random.seed(abs(hash(country)) % 1000)
        base_growth = {"USA": 2.5, "IND": 6.0, "CHN": 7.0}
        base = base_growth.get(country, 2.0)
        cycle = np.sin(np.linspace(0, 2 * np.pi, n_years))
        noise = np.random.normal(0, 0.8, n_years)
        return (base + cycle + noise).tolist()

    # ------------------------
    # Validation
    # ------------------------
    def validate_data(
        self, data: List[float], indicator_type: str = "general"
    ) -> Dict[str, Any]:
        if not data:
            return {"valid": False, "error": "Empty data"}
        arr = np.array(data, dtype=float)
        stats = {
            "count": len(arr),
            "mean": float(np.nanmean(arr)),
            "std": float(np.nanstd(arr)),
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
        }
        valid = np.isfinite(arr).all() and not np.isnan(arr).all()
        return {"valid": valid, "statistics": stats}

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


# Example usage:
if __name__ == "__main__":
    loader = IMFDataLoader()
    df = loader.fetch_weo_data("USA", "PCPIPCH", 2010, 2023)
    print(df.head())
