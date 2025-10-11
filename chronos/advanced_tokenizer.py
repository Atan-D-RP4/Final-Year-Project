"""
Advanced Tokenizer for Chronos Forecasting
Supports multivariate inputs, feature engineering, and economic-specific preprocessing
"""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from scipy import stats
from scipy.signal import detrend
from statsmodels.tsa.seasonal import seasonal_decompose
from typing import List, Dict, Any, Optional, Tuple, Union
import warnings

warnings.filterwarnings("ignore")


class AdvancedTokenizer:
    """
    Advanced tokenizer with multivariate support and feature engineering
    """
    
    def __init__(self, 
                 window_size: int = 10,
                 quantization_levels: int = 1000,
                 scaling_method: str = 'standard',
                 feature_selection: bool = True,
                 max_features: int = 10,
                 economic_features: bool = True):
        """
        Initialize advanced tokenizer
        
        Args:
            window_size: Size of sliding window
            quantization_levels: Number of quantization levels
            scaling_method: 'standard', 'robust', 'minmax', or 'none'
            feature_selection: Whether to perform feature selection
            max_features: Maximum number of features to select
            economic_features: Whether to add economic-specific features
        """
        self.window_size = window_size
        self.quantization_levels = quantization_levels
        self.scaling_method = scaling_method
        self.feature_selection = feature_selection
        self.max_features = max_features
        self.economic_features = economic_features
        
        # Initialize scalers and selectors
        self.scaler = self._get_scaler()
        self.feature_selector = None
        self.is_fitted = False
        
        # Store feature names for interpretability
        self.feature_names = []
        self.selected_features = []
    
    def _get_scaler(self):
        """Get the appropriate scaler based on method"""
        if self.scaling_method == 'standard':
            return StandardScaler()
        elif self.scaling_method == 'robust':
            return RobustScaler()
        elif self.scaling_method == 'minmax':
            return MinMaxScaler(feature_range=(0, self.quantization_levels - 1))
        else:
            return None
    
    def _create_lag_features(self, data: np.ndarray, max_lags: int = 5) -> np.ndarray:
        """Create lag features"""
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        n_samples, n_features = data.shape
        lag_features = []
        
        for lag in range(1, max_lags + 1):
            if lag < n_samples:
                lagged = np.roll(data, lag, axis=0)
                lagged[:lag] = np.nan  # Set initial values to NaN
                lag_features.append(lagged)
        
        if lag_features:
            return np.concatenate(lag_features, axis=1)
        else:
            return np.array([]).reshape(n_samples, 0)
    
    def _create_moving_average_features(self, data: np.ndarray, 
                                      windows: List[int] = [3, 5, 10]) -> np.ndarray:
        """Create moving average features"""
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        n_samples, n_features = data.shape
        ma_features = []
        
        for window in windows:
            if window < n_samples:
                ma_data = np.full_like(data, np.nan)
                for i in range(window - 1, n_samples):
                    ma_data[i] = np.mean(data[i - window + 1:i + 1], axis=0)
                ma_features.append(ma_data)
        
        if ma_features:
            return np.concatenate(ma_features, axis=1)
        else:
            return np.array([]).reshape(n_samples, 0)
    
    def _create_volatility_features(self, data: np.ndarray, 
                                  windows: List[int] = [5, 10]) -> np.ndarray:
        """Create volatility (rolling standard deviation) features"""
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        n_samples, n_features = data.shape
        vol_features = []
        
        for window in windows:
            if window < n_samples:
                vol_data = np.full_like(data, np.nan)
                for i in range(window - 1, n_samples):
                    vol_data[i] = np.std(data[i - window + 1:i + 1], axis=0)
                vol_features.append(vol_data)
        
        if vol_features:
            return np.concatenate(vol_features, axis=1)
        else:
            return np.array([]).reshape(n_samples, 0)
    
    def _create_rate_of_change_features(self, data: np.ndarray, 
                                      periods: List[int] = [1, 3, 5]) -> np.ndarray:
        """Create rate of change features"""
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        n_samples, n_features = data.shape
        roc_features = []
        
        for period in periods:
            if period < n_samples:
                roc_data = np.full_like(data, np.nan)
                for i in range(period, n_samples):
                    roc_data[i] = (data[i] - data[i - period]) / (data[i - period] + 1e-8)
                roc_features.append(roc_data)
        
        if roc_features:
            return np.concatenate(roc_features, axis=1)
        else:
            return np.array([]).reshape(n_samples, 0)
    
    def _create_economic_features(self, data: np.ndarray) -> np.ndarray:
        """Create economic-specific features"""
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        n_samples, n_features = data.shape
        econ_features = []
        
        # Trend features
        for i in range(n_features):
            series = data[:, i]
            
            # Linear trend
            x = np.arange(len(series))
            valid_mask = ~np.isnan(series)
            if np.sum(valid_mask) > 2:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    x[valid_mask], series[valid_mask]
                )
                trend_feature = slope * x + intercept
            else:
                trend_feature = np.full_like(series, np.nan)
            
            econ_features.append(trend_feature.reshape(-1, 1))
            
            # Detrended series
            try:
                detrended_series = detrend(series[valid_mask])
                detrended_full = np.full_like(series, np.nan)
                detrended_full[valid_mask] = detrended_series
                econ_features.append(detrended_full.reshape(-1, 1))
            except:
                econ_features.append(np.full_like(series, np.nan).reshape(-1, 1))
        
        # Seasonal decomposition (if enough data)
        if n_samples >= 24:  # Need at least 2 years of monthly data
            for i in range(n_features):
                series = data[:, i]
                valid_mask = ~np.isnan(series)
                
                if np.sum(valid_mask) >= 24:
                    try:
                        # Create a pandas series for seasonal decomposition
                        ts = pd.Series(series[valid_mask])
                        decomposition = seasonal_decompose(ts, model='additive', period=12)
                        
                        # Extract seasonal and residual components
                        seasonal_full = np.full_like(series, np.nan)
                        residual_full = np.full_like(series, np.nan)
                        
                        seasonal_full[valid_mask] = decomposition.seasonal.values
                        residual_full[valid_mask] = decomposition.resid.values
                        
                        econ_features.append(seasonal_full.reshape(-1, 1))
                        econ_features.append(residual_full.reshape(-1, 1))
                    except:
                        # If seasonal decomposition fails, add NaN features
                        econ_features.append(np.full_like(series, np.nan).reshape(-1, 1))
                        econ_features.append(np.full_like(series, np.nan).reshape(-1, 1))
                else:
                    # Not enough data for seasonal decomposition
                    econ_features.append(np.full_like(series, np.nan).reshape(-1, 1))
                    econ_features.append(np.full_like(series, np.nan).reshape(-1, 1))
        
        if econ_features:
            return np.concatenate(econ_features, axis=1)
        else:
            return np.array([]).reshape(n_samples, 0)
    
    def _create_statistical_features(self, data: np.ndarray) -> np.ndarray:
        """Create statistical features from sliding windows"""
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        n_samples, n_features = data.shape
        features = []
        
        for i in range(self.window_size - 1, n_samples):
            window_data = data[i - self.window_size + 1:i + 1]
            
            window_features = []
            for j in range(n_features):
                series = window_data[:, j]
                valid_data = series[~np.isnan(series)]
                
                if len(valid_data) > 0:
                    # Basic statistics
                    window_features.extend([
                        np.mean(valid_data),
                        np.std(valid_data),
                        np.min(valid_data),
                        np.max(valid_data),
                        np.median(valid_data),
                    ])
                    
                    # Advanced statistics
                    if len(valid_data) > 2:
                        window_features.extend([
                            stats.skew(valid_data),
                            stats.kurtosis(valid_data),
                        ])
                    else:
                        window_features.extend([0.0, 0.0])
                    
                    # Trend features
                    if len(valid_data) > 1:
                        diff = np.diff(valid_data)
                        window_features.extend([
                            np.sum(diff > 0),  # Upward movements
                            np.sum(diff < 0),  # Downward movements
                            valid_data[-1] - valid_data[0],  # Total change
                        ])
                    else:
                        window_features.extend([0.0, 0.0, 0.0])
                else:
                    # If no valid data, fill with zeros
                    window_features.extend([0.0] * 10)
            
            features.append(window_features)
        
        return np.array(features)
    
    def _generate_feature_names(self, n_original_features: int) -> List[str]:
        """Generate feature names for interpretability"""
        names = []
        
        # Original features
        for i in range(n_original_features):
            names.append(f'original_{i}')
        
        # Statistical features from sliding windows
        for i in range(n_original_features):
            base_name = f'series_{i}'
            names.extend([
                f'{base_name}_mean', f'{base_name}_std', f'{base_name}_min',
                f'{base_name}_max', f'{base_name}_median', f'{base_name}_skew',
                f'{base_name}_kurtosis', f'{base_name}_up_moves', f'{base_name}_down_moves',
                f'{base_name}_total_change'
            ])\n        \n        # Lag features\n        for lag in range(1, 6):  # max_lags = 5\n            for i in range(n_original_features):\n                names.append(f'lag_{lag}_series_{i}')\n        \n        # Moving average features\n        for window in [3, 5, 10]:\n            for i in range(n_original_features):\n                names.append(f'ma_{window}_series_{i}')\n        \n        # Volatility features\n        for window in [5, 10]:\n            for i in range(n_original_features):\n                names.append(f'vol_{window}_series_{i}')\n        \n        # Rate of change features\n        for period in [1, 3, 5]:\n            for i in range(n_original_features):\n                names.append(f'roc_{period}_series_{i}')\n        \n        # Economic features\n        if self.economic_features:\n            for i in range(n_original_features):\n                names.extend([\n                    f'trend_series_{i}', f'detrended_series_{i}',\n                    f'seasonal_series_{i}', f'residual_series_{i}'\n                ])\n        \n        return names
    
    def fit_transform(self, data: Union[List[float], np.ndarray, Dict[str, List[float]]], \n                  target: Optional[Union[List[float], np.ndarray]] = None) -> torch.Tensor:\n        \"\"\"\n        Fit the tokenizer and transform data\n        \n        Args:\n            data: Input data (univariate list/array or multivariate dict)\n            target: Target variable for feature selection (optional)\n            \n        Returns:\n            Tokenized tensor ready for Chronos\n        \"\"\"\n        # Convert input to numpy array\n        if isinstance(data, dict):\n            # Multivariate case\n            data_arrays = []\n            for key, values in data.items():\n                data_arrays.append(np.array(values))\n            data_matrix = np.column_stack(data_arrays)\n        else:\n            # Univariate case\n            data_matrix = np.array(data).reshape(-1, 1)\n        \n        n_samples, n_features = data_matrix.shape\n        \n        # Generate feature names\n        self.feature_names = self._generate_feature_names(n_features)\n        \n        # Create all features\n        all_features = []\n        \n        # 1. Statistical features from sliding windows\n        stat_features = self._create_statistical_features(data_matrix)\n        if stat_features.size > 0:\n            all_features.append(stat_features)\n        \n        # 2. Lag features\n        lag_features = self._create_lag_features(data_matrix)\n        if lag_features.size > 0:\n            # Align with statistical features (remove first window_size-1 rows)\n            lag_features_aligned = lag_features[self.window_size - 1:]\n            all_features.append(lag_features_aligned)\n        \n        # 3. Moving average features\n        ma_features = self._create_moving_average_features(data_matrix)\n        if ma_features.size > 0:\n            ma_features_aligned = ma_features[self.window_size - 1:]\n            all_features.append(ma_features_aligned)\n        \n        # 4. Volatility features\n        vol_features = self._create_volatility_features(data_matrix)\n        if vol_features.size > 0:\n            vol_features_aligned = vol_features[self.window_size - 1:]\n            all_features.append(vol_features_aligned)\n        \n        # 5. Rate of change features\n        roc_features = self._create_rate_of_change_features(data_matrix)\n        if roc_features.size > 0:\n            roc_features_aligned = roc_features[self.window_size - 1:]\n            all_features.append(roc_features_aligned)\n        \n        # 6. Economic features\n        if self.economic_features:\n            econ_features = self._create_economic_features(data_matrix)\n            if econ_features.size > 0:\n                econ_features_aligned = econ_features[self.window_size - 1:]\n                all_features.append(econ_features_aligned)\n        \n        # Combine all features\n        if all_features:\n            combined_features = np.concatenate(all_features, axis=1)\n        else:\n            # Fallback to simple statistical features\n            combined_features = stat_features\n        \n        # Handle NaN values\n        combined_features = np.nan_to_num(combined_features, nan=0.0, posinf=0.0, neginf=0.0)\n        \n        # Feature selection\n        if self.feature_selection and target is not None and combined_features.shape[1] > self.max_features:\n            target_aligned = np.array(target)[self.window_size - 1:]\n            \n            self.feature_selector = SelectKBest(\n                score_func=f_regression, \n                k=min(self.max_features, combined_features.shape[1])\n            )\n            \n            combined_features = self.feature_selector.fit_transform(combined_features, target_aligned)\n            \n            # Update selected feature names\n            if hasattr(self.feature_selector, 'get_support'):\n                selected_mask = self.feature_selector.get_support()\n                self.selected_features = [name for name, selected in zip(self.feature_names, selected_mask) if selected]\n        \n        # Scaling\n        if self.scaler is not None:\n            combined_features = self.scaler.fit_transform(combined_features)\n        \n        # Quantization\n        if self.scaling_method != 'minmax':  # MinMaxScaler already handles this\n            combined_features = np.clip(combined_features, -3, 3)  # Clip outliers\n            combined_features = (combined_features + 3) / 6 * (self.quantization_levels - 1)\n            combined_features = np.round(combined_features).astype(int)\n        \n        # Create pseudo time series by taking mean across features\n        pseudo_timeseries = np.mean(combined_features, axis=1)\n        \n        self.is_fitted = True\n        return torch.tensor(pseudo_timeseries, dtype=torch.float32)\n    \n    def transform(self, data: Union[List[float], np.ndarray, Dict[str, List[float]]]) -> torch.Tensor:\n        \"\"\"\n        Transform new data using fitted tokenizer\n        \n        Args:\n            data: Input data in same format as fit_transform\n            \n        Returns:\n            Tokenized tensor\n        \"\"\"\n        if not self.is_fitted:\n            raise ValueError(\"Tokenizer must be fitted first\")\n        \n        # Convert input to numpy array\n        if isinstance(data, dict):\n            data_arrays = []\n            for key, values in data.items():\n                data_arrays.append(np.array(values))\n            data_matrix = np.column_stack(data_arrays)\n        else:\n            data_matrix = np.array(data).reshape(-1, 1)\n        \n        # Create features (same process as fit_transform)\n        all_features = []\n        \n        # Statistical features\n        stat_features = self._create_statistical_features(data_matrix)\n        if stat_features.size > 0:\n            all_features.append(stat_features)\n        \n        # Other features...\n        lag_features = self._create_lag_features(data_matrix)\n        if lag_features.size > 0:\n            lag_features_aligned = lag_features[self.window_size - 1:]\n            all_features.append(lag_features_aligned)\n        \n        ma_features = self._create_moving_average_features(data_matrix)\n        if ma_features.size > 0:\n            ma_features_aligned = ma_features[self.window_size - 1:]\n            all_features.append(ma_features_aligned)\n        \n        vol_features = self._create_volatility_features(data_matrix)\n        if vol_features.size > 0:\n            vol_features_aligned = vol_features[self.window_size - 1:]\n            all_features.append(vol_features_aligned)\n        \n        roc_features = self._create_rate_of_change_features(data_matrix)\n        if roc_features.size > 0:\n            roc_features_aligned = roc_features[self.window_size - 1:]\n            all_features.append(roc_features_aligned)\n        \n        if self.economic_features:\n            econ_features = self._create_economic_features(data_matrix)\n            if econ_features.size > 0:\n                econ_features_aligned = econ_features[self.window_size - 1:]\n                all_features.append(econ_features_aligned)\n        \n        # Combine features\n        if all_features:\n            combined_features = np.concatenate(all_features, axis=1)\n        else:\n            combined_features = stat_features\n        \n        # Handle NaN values\n        combined_features = np.nan_to_num(combined_features, nan=0.0, posinf=0.0, neginf=0.0)\n        \n        # Apply feature selection\n        if self.feature_selector is not None:\n            combined_features = self.feature_selector.transform(combined_features)\n        \n        # Apply scaling\n        if self.scaler is not None:\n            combined_features = self.scaler.transform(combined_features)\n        \n        # Quantization\n        if self.scaling_method != 'minmax':\n            combined_features = np.clip(combined_features, -3, 3)\n            combined_features = (combined_features + 3) / 6 * (self.quantization_levels - 1)\n            combined_features = np.round(combined_features).astype(int)\n        \n        # Create pseudo time series\n        pseudo_timeseries = np.mean(combined_features, axis=1)\n        \n        return torch.tensor(pseudo_timeseries, dtype=torch.float32)\n    \n    def get_feature_importance(self) -> Dict[str, float]:\n        \"\"\"\n        Get feature importance scores (if feature selection was used)\n        \n        Returns:\n            Dictionary mapping feature names to importance scores\n        \"\"\"\n        if self.feature_selector is None:\n            return {}\n        \n        if hasattr(self.feature_selector, 'scores_'):\n            scores = self.feature_selector.scores_\n            selected_mask = self.feature_selector.get_support()\n            \n            importance_dict = {}\n            for i, (name, selected) in enumerate(zip(self.feature_names, selected_mask)):\n                if selected and i < len(scores):\n                    importance_dict[name] = scores[i]\n            \n            return importance_dict\n        \n        return {}\n    \n    def get_feature_names(self) -> List[str]:\n        \"\"\"\n        Get names of selected features\n        \n        Returns:\n            List of feature names\n        \"\"\"\n        if self.selected_features:\n            return self.selected_features\n        else:\n            return self.feature_names[:self.max_features] if self.feature_names else []\n\n\ndef test_advanced_tokenizer():\n    \"\"\"\n    Test the advanced tokenizer with different data types\n    \"\"\"\n    print(\"Testing Advanced Tokenizer...\")\n    \n    # Test 1: Univariate data\n    print(\"\\n1. Testing univariate data...\")\n    np.random.seed(42)\n    univariate_data = np.cumsum(np.random.randn(50)) + 100\n    \n    tokenizer = AdvancedTokenizer(\n        window_size=10, \n        quantization_levels=1000,\n        scaling_method='standard',\n        feature_selection=True,\n        max_features=15,\n        economic_features=True\n    )\n    \n    tokenized = tokenizer.fit_transform(univariate_data.tolist())\n    print(f\"   Input shape: {univariate_data.shape}\")\n    print(f\"   Output shape: {tokenized.shape}\")\n    print(f\"   Feature names: {len(tokenizer.get_feature_names())}\")\n    \n    # Test 2: Multivariate data\n    print(\"\\n2. Testing multivariate data...\")\n    multivariate_data = {\n        'gdp_growth': (np.random.randn(50) * 2 + 2.5).tolist(),\n        'inflation': (np.random.randn(50) * 1 + 2.0).tolist(),\n        'unemployment': (np.random.randn(50) * 0.5 + 5.0).tolist()\n    }\n    \n    tokenizer_mv = AdvancedTokenizer(\n        window_size=8,\n        quantization_levels=1000,\n        scaling_method='robust',\n        feature_selection=True,\n        max_features=20,\n        economic_features=True\n    )\n    \n    tokenized_mv = tokenizer_mv.fit_transform(multivariate_data)\n    print(f\"   Input variables: {list(multivariate_data.keys())}\")\n    print(f\"   Output shape: {tokenized_mv.shape}\")\n    print(f\"   Selected features: {len(tokenizer_mv.get_feature_names())}\")\n    \n    # Test 3: Feature importance\n    print(\"\\n3. Testing feature importance...\")\n    target = np.random.randn(len(tokenized_mv))\n    \n    # Refit with target for feature selection\n    tokenized_with_target = tokenizer_mv.fit_transform(multivariate_data, target)\n    importance = tokenizer_mv.get_feature_importance()\n    \n    if importance:\n        print(\"   Top 5 most important features:\")\n        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]\n        for name, score in sorted_features:\n            print(f\"     {name}: {score:.3f}\")\n    \n    print(\"\\nAdvanced Tokenizer test completed!\")\n    return tokenizer, tokenizer_mv\n\n\nif __name__ == \"__main__\":\n    # Run tests\n    tokenizer, tokenizer_mv = test_advanced_tokenizer()