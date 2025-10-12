"""
Baseline Models for Time Series Forecasting Comparison
Implements ARIMA, Exponential Smoothing, Naive forecasts, and other classical methods
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Try to import statsmodels for ARIMA and exponential smoothing
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller

    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print(
        "Warning: statsmodels not available. Some baseline models will use simplified implementations."
    )

warnings.filterwarnings("ignore")


class BaselineForecaster:
    """
    Base class for baseline forecasting models
    """

    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
        self.model = None

    def fit(self, data: List[float]) -> "BaselineForecaster":
        """Fit the model to training data"""
        raise NotImplementedError

    def predict(self, steps: int) -> List[float]:
        """Make predictions for specified number of steps"""
        raise NotImplementedError

    def fit_predict(self, data: List[float], steps: int) -> List[float]:
        """Fit model and make predictions"""
        self.fit(data)
        return self.predict(steps)


class NaiveForecaster(BaselineForecaster):
    """
    Naive forecasting: repeat the last observed value
    """

    def __init__(self):
        super().__init__("Naive")
        self.last_value = None

    def fit(self, data: List[float]) -> "NaiveForecaster":
        """Fit naive model (just store last value)"""
        if not data:
            raise ValueError("Data cannot be empty")
        self.last_value = data[-1]
        self.is_fitted = True
        return self

    def predict(self, steps: int) -> List[float]:
        """Predict by repeating last value"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return [self.last_value] * steps


class SeasonalNaiveForecaster(BaselineForecaster):
    """
    Seasonal naive forecasting: repeat the pattern from the same season last year
    """

    def __init__(self, season_length: int = 12):
        super().__init__("Seasonal Naive")
        self.season_length = season_length
        self.data = None

    def fit(self, data: List[float]) -> "SeasonalNaiveForecaster":
        """Fit seasonal naive model"""
        if len(data) < self.season_length:
            raise ValueError(
                f"Data must have at least {self.season_length} observations"
            )
        self.data = data.copy()
        self.is_fitted = True
        return self

    def predict(self, steps: int) -> List[float]:
        """Predict using seasonal pattern"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        predictions = []
        for i in range(steps):
            # Use value from same season in previous cycle
            seasonal_index = (len(self.data) + i) % self.season_length
            if seasonal_index < len(self.data):
                predictions.append(self.data[-(self.season_length - seasonal_index)])
            else:
                # If not enough history, use last available value
                predictions.append(self.data[-1])

        return predictions


class DriftForecaster(BaselineForecaster):
    """
    Drift forecasting: linear trend extrapolation
    """

    def __init__(self):
        super().__init__("Drift")
        self.slope = None
        self.last_value = None

    def fit(self, data: List[float]) -> "DriftForecaster":
        """Fit drift model (calculate linear trend)"""
        if len(data) < 2:
            raise ValueError("Data must have at least 2 observations")

        # Calculate average drift (slope)
        self.slope = (data[-1] - data[0]) / (len(data) - 1)
        self.last_value = data[-1]
        self.is_fitted = True
        return self

    def predict(self, steps: int) -> List[float]:
        """Predict using linear drift"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        predictions = []
        for i in range(1, steps + 1):
            predictions.append(self.last_value + self.slope * i)

        return predictions


class MovingAverageForecaster(BaselineForecaster):
    """
    Moving average forecasting
    """

    def __init__(self, window_size: int = 5):
        super().__init__(f"Moving Average ({window_size})")
        self.window_size = window_size
        self.recent_values = None

    def fit(self, data: List[float]) -> "MovingAverageForecaster":
        """Fit moving average model"""
        if len(data) < self.window_size:
            raise ValueError(f"Data must have at least {self.window_size} observations")

        self.recent_values = data[-self.window_size :]
        self.is_fitted = True
        return self

    def predict(self, steps: int) -> List[float]:
        """Predict using moving average"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        predictions = []
        current_values = self.recent_values.copy()

        for _ in range(steps):
            # Predict as average of recent values
            prediction = np.mean(current_values)
            predictions.append(prediction)

            # Update window (add prediction, remove oldest)
            current_values = current_values[1:] + [prediction]

        return predictions


class LinearTrendForecaster(BaselineForecaster):
    """
    Linear trend forecasting using linear regression
    """

    def __init__(self):
        super().__init__("Linear Trend")
        self.model = LinearRegression()
        self.n_obs = None

    def fit(self, data: List[float]) -> "LinearTrendForecaster":
        """Fit linear trend model"""
        if len(data) < 2:
            raise ValueError("Data must have at least 2 observations")

        X = np.arange(len(data)).reshape(-1, 1)
        y = np.array(data)

        self.model.fit(X, y)
        self.n_obs = len(data)
        self.is_fitted = True
        return self

    def predict(self, steps: int) -> List[float]:
        """Predict using linear trend"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        future_X = np.arange(self.n_obs, self.n_obs + steps).reshape(-1, 1)
        predictions = self.model.predict(future_X)

        return predictions.tolist()


class ExponentialSmoothingForecaster(BaselineForecaster):
    """
    Exponential smoothing forecasting
    """

    def __init__(
        self,
        alpha: float = 0.3,
        trend: Optional[str] = None,
        seasonal: Optional[str] = None,
        seasonal_periods: int = 12,
    ):
        name = f"Exponential Smoothing (α={alpha}"
        if trend:
            name += f", trend={trend}"
        if seasonal:
            name += f", seasonal={seasonal}"
        name += ")"

        super().__init__(name)
        self.alpha = alpha
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.fitted_model = None

    def fit(self, data: List[float]) -> "ExponentialSmoothingForecaster":
        """Fit exponential smoothing model"""
        if len(data) < 2:
            raise ValueError("Data must have at least 2 observations")

        if STATSMODELS_AVAILABLE:
            try:
                # Use statsmodels for proper exponential smoothing
                ts = pd.Series(data)

                # Determine if we have enough data for seasonal models
                if self.seasonal and len(data) < 2 * self.seasonal_periods:
                    print(
                        "Warning: Not enough data for seasonal model. Using simple exponential smoothing."
                    )
                    self.seasonal = None

                self.fitted_model = ExponentialSmoothing(
                    ts,
                    trend=self.trend,
                    seasonal=self.seasonal,
                    seasonal_periods=self.seasonal_periods if self.seasonal else None,
                ).fit(smoothing_level=self.alpha)

                self.is_fitted = True
                return self

            except Exception as e:
                print(
                    f"Warning: statsmodels exponential smoothing failed: {e}. Using simple implementation."
                )

        # Fallback to simple exponential smoothing
        self._fit_simple_exponential_smoothing(data)
        return self

    def _fit_simple_exponential_smoothing(self, data: List[float]):
        """Simple exponential smoothing implementation"""
        self.level = data[0]

        # Update level for each observation
        for value in data[1:]:
            self.level = self.alpha * value + (1 - self.alpha) * self.level

        self.is_fitted = True

    def predict(self, steps: int) -> List[float]:
        """Predict using exponential smoothing"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        if self.fitted_model is not None:
            # Use statsmodels prediction
            forecast = self.fitted_model.forecast(steps)
            return forecast.tolist()
        else:
            # Use simple exponential smoothing prediction
            return [self.level] * steps


class ARIMAForecaster(BaselineForecaster):
    """
    ARIMA forecasting model
    """

    def __init__(
        self, order: Tuple[int, int, int] = (1, 1, 1), auto_order: bool = True
    ):
        super().__init__(f"ARIMA{order}")
        self.order = order
        self.auto_order = auto_order
        self.fitted_model = None
        self.data = None

    def fit(self, data: List[float]) -> "ARIMAForecaster":
        """Fit ARIMA model"""
        if len(data) < 10:
            raise ValueError("ARIMA requires at least 10 observations")

        self.data = data.copy()

        if STATSMODELS_AVAILABLE:
            try:
                ts = pd.Series(data)

                if self.auto_order:
                    # Simple auto-order selection
                    best_aic = np.inf
                    best_order = (1, 1, 1)

                    for p in range(3):
                        for d in range(2):
                            for q in range(3):
                                try:
                                    model = ARIMA(ts, order=(p, d, q))
                                    fitted = model.fit()
                                    if fitted.aic < best_aic:
                                        best_aic = fitted.aic
                                        best_order = (p, d, q)
                                except Exception as e:
                                    print(f"ARIMA order {(p,d,q)} failed: {e}")
                                    continue

                    self.order = best_order
                    self.name = f"ARIMA{best_order}"

                # Fit final model
                model = ARIMA(ts, order=self.order)
                self.fitted_model = model.fit()
                self.is_fitted = True
                return self

            except Exception as e:
                print(f"Warning: ARIMA fitting failed: {e}. Using simple AR model.")

        # Fallback to simple AR(1) model
        self._fit_simple_ar(data)
        return self

    def _fit_simple_ar(self, data: List[float]):
        """Simple AR(1) model implementation"""
        if len(data) < 2:
            raise ValueError("Need at least 2 observations for AR model")

        # Fit AR(1): y_t = c + φ * y_{t-1} + ε_t
        y = np.array(data[1:])
        X = np.array(data[:-1]).reshape(-1, 1)

        self.ar_model = LinearRegression()
        self.ar_model.fit(X, y)
        self.last_value = data[-1]
        self.is_fitted = True

    def predict(self, steps: int) -> List[float]:
        """Predict using ARIMA model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        if self.fitted_model is not None:
            # Use statsmodels prediction
            forecast = self.fitted_model.forecast(steps)
            return forecast.tolist()
        else:
            # Use simple AR(1) prediction
            predictions = []
            current_value = self.last_value

            for _ in range(steps):
                prediction = self.ar_model.predict([[current_value]])[0]
                predictions.append(prediction)
                current_value = prediction

            return predictions


class BaselineComparison:
    """
    Compare multiple baseline models
    """

    def __init__(self):
        self.models = {}
        self.results = {}

    def add_model(self, model: BaselineForecaster):
        """Add a baseline model to comparison"""
        self.models[model.name] = model

    def add_standard_models(self, seasonal_periods: int = 12):
        """Add standard set of baseline models"""
        self.add_model(NaiveForecaster())
        self.add_model(SeasonalNaiveForecaster(seasonal_periods))
        self.add_model(DriftForecaster())
        self.add_model(MovingAverageForecaster(3))
        self.add_model(MovingAverageForecaster(5))
        self.add_model(LinearTrendForecaster())
        self.add_model(ExponentialSmoothingForecaster(alpha=0.3))
        self.add_model(ExponentialSmoothingForecaster(alpha=0.3, trend="add"))

        if STATSMODELS_AVAILABLE:
            self.add_model(ARIMAForecaster(auto_order=True))

    def compare_models(
        self,
        train_data: List[float],
        test_data: List[float],
        show_progress: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare all models on given train/test split

        Args:
            train_data: Training data
            test_data: Test data for evaluation
            show_progress: Whether to show progress

        Returns:
            Dictionary with model performance metrics
        """
        if show_progress:
            print(f"Comparing {len(self.models)} baseline models...")

        results = {}
        prediction_length = len(test_data)

        for model_name, model in self.models.items():
            if show_progress:
                print(f"  Testing {model_name}...")

            try:
                # Fit model and make predictions
                predictions = model.fit_predict(train_data, prediction_length)

                # Calculate metrics
                mse = mean_squared_error(test_data, predictions)
                mae = mean_absolute_error(test_data, predictions)
                rmse = np.sqrt(mse)
                mape = (
                    np.mean(
                        np.abs(
                            (np.array(test_data) - np.array(predictions))
                            / (np.array(test_data) + 1e-8)
                        )
                    )
                    * 100
                )

                # MASE calculation
                naive_error = np.mean(np.abs(np.diff(train_data)))
                mase = mae / naive_error if naive_error > 0 else np.inf

                # Directional accuracy
                if len(test_data) > 1:
                    true_direction = np.sign(np.diff(test_data))
                    pred_direction = np.sign(np.diff(predictions))
                    directional_accuracy = (
                        np.mean(true_direction == pred_direction) * 100
                    )
                else:
                    directional_accuracy = 0.0

                results[model_name] = {
                    "mse": mse,
                    "rmse": rmse,
                    "mae": mae,
                    "mape": mape,
                    "mase": mase,
                    "directional_accuracy": directional_accuracy,
                    "predictions": predictions,
                }

            except Exception as e:
                if show_progress:
                    print(f"    Error: {e}")
                results[model_name] = {
                    "mse": np.inf,
                    "rmse": np.inf,
                    "mae": np.inf,
                    "mape": np.inf,
                    "mase": np.inf,
                    "directional_accuracy": 0.0,
                    "predictions": [np.nan] * prediction_length,
                }

        self.results = results
        return results

    def print_comparison(self):
        """Print formatted comparison results"""
        if not self.results:
            print("No results to display. Run comparison first.")
            return

        print(" " + "=" * 100)
        print("BASELINE MODEL COMPARISON")
        print("=" * 100)

        # Sort models by MASE (lower is better)
        sorted_models = sorted(self.results.items(), key=lambda x: x[1]["mase"])

        print(
            f"{'Model':<25} {'MSE':<10} {'RMSE':<10} {'MAE':<10} {'MAPE':<10} {'MASE':<10} {'Dir.Acc':<10}"
        )
        print("-" * 100)

        for model_name, metrics in sorted_models:
            print(
                f"{model_name:<25} {metrics['mse']:<10.4f} {metrics['rmse']:<10.4f} "
                f"{metrics['mae']:<10.4f} {metrics['mape']:<10.2f} {metrics['mase']:<10.4f} "
                f"{metrics['directional_accuracy']:<10.2f}"
            )

        # Highlight best model
        best_model = sorted_models[0][0]
        print(f"Best Model (by MASE): {best_model}")

    def plot_comparison(self, test_data: List[float], top_n: int = 5):
        """Plot predictions from top N models"""
        if not self.results:
            print("No results to plot. Run comparison first.")
            return

        import matplotlib.pyplot as plt

        # Sort models by MASE and take top N
        sorted_models = sorted(self.results.items(), key=lambda x: x[1]["mase"])[:top_n]

        plt.figure(figsize=(12, 8))

        # Plot actual data
        plt.plot(
            range(len(test_data)),
            test_data,
            "k-",
            linewidth=2,
            label="Actual",
            alpha=0.8,
        )

        # Plot predictions from top models
        colors = ["red", "blue", "green", "orange", "purple"]
        for i, (model_name, metrics) in enumerate(sorted_models):
            if not np.any(np.isnan(metrics["predictions"])):
                plt.plot(
                    range(len(test_data)),
                    metrics["predictions"],
                    color=colors[i % len(colors)],
                    linestyle="--",
                    label=f"{model_name} (MASE: {metrics['mase']:.3f})",
                    alpha=0.7,
                )

        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.title(f"Baseline Model Comparison - Top {top_n} Models")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def test_baseline_models():
    """
    Test baseline models with synthetic data
    """
    print("Testing Baseline Models...")

    # Generate synthetic time series
    np.random.seed(42)
    n_samples = 60
    trend = np.linspace(0, 5, n_samples)
    seasonal = 2 * np.sin(2 * np.pi * np.arange(n_samples) / 12)
    noise = np.random.normal(0, 0.5, n_samples)
    data = (trend + seasonal + noise + 10).tolist()

    # Split into train/test
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    test_data = data[train_size:]

    print(
        f"Generated data: {len(data)} points, train: {len(train_data)}, test: {len(test_data)}"
    )

    # Test individual models
    print("Testing individual models:")

    models = [
        NaiveForecaster(),
        DriftForecaster(),
        MovingAverageForecaster(5),
        LinearTrendForecaster(),
        ExponentialSmoothingForecaster(alpha=0.3),
    ]

    for model in models:
        try:
            predictions = model.fit_predict(train_data, len(test_data))
            mae = mean_absolute_error(test_data, predictions)
            print(f"  {model.name}: MAE = {mae:.4f}")
        except Exception as e:
            print(f"  {model.name}: Error - {e}")

    # Test comparison framework
    print("Testing comparison framework:")
    comparison = BaselineComparison()
    comparison.add_standard_models(seasonal_periods=12)

    _ = comparison.compare_models(train_data, test_data)
    comparison.print_comparison()

    print("Baseline models test completed!")
    return comparison, train_data, test_data


if __name__ == "__main__":
    # Run tests
    comparison, train_data, test_data = test_baseline_models()

