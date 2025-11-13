"""Attribution and causality analysis for forecasting models."""

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
import pandas as pd


class AttributionMethod(ABC):
    """Base class for attribution methods."""

    @abstractmethod
    def compute_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        predict_fn: Callable,
    ) -> np.ndarray:
        """Compute feature importance.

        Args:
            X: Feature matrix
            y: Target values
            predict_fn: Prediction function

        Returns:
            Feature importance scores
        """


class AblationImportance(AttributionMethod):
    """Ablation-based feature importance.

    Measure importance by ablating each feature and measuring
    performance degradation.
    """

    def __init__(self, metric_fn: Callable | None = None):
        """Initialize ablation importance.

        Args:
            metric_fn: Function to compute performance metric
        """
        if metric_fn is None:
            from src.eval.metrics import mae

            metric_fn = mae

        self.metric_fn = metric_fn

    def compute_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        predict_fn: Callable,
    ) -> np.ndarray:
        """Compute ablation-based importance.

        Args:
            X: Feature matrix (shape: n_samples, n_features)
            y: Target values
            predict_fn: Prediction function

        Returns:
            Feature importance scores
        """
        n_features = X.shape[1]
        baseline_pred = predict_fn(X)
        baseline_score = self.metric_fn(y, baseline_pred)

        importances = np.zeros(n_features)

        for feature_idx in range(n_features):
            # Create ablated version
            X_ablated = X.copy()
            X_ablated[:, feature_idx] = 0  # Ablate feature

            # Compute degraded performance
            ablated_pred = predict_fn(X_ablated)
            ablated_score = self.metric_fn(y, ablated_pred)

            # Importance is performance degradation
            importances[feature_idx] = ablated_score - baseline_score

        # Normalize
        if np.sum(np.abs(importances)) > 0:
            importances = importances / np.sum(np.abs(importances))

        return importances


class PermutationImportance(AttributionMethod):
    """Permutation-based feature importance.

    Measure importance by randomly permuting each feature and measuring
    performance degradation.
    """

    def __init__(
        self,
        metric_fn: Callable | None = None,
        n_repeats: int = 10,
        random_state: int = 42,
    ):
        """Initialize permutation importance.

        Args:
            metric_fn: Function to compute performance metric
            n_repeats: Number of permutation repeats
            random_state: Random state for reproducibility
        """
        if metric_fn is None:
            from src.eval.metrics import mae

            metric_fn = mae

        self.metric_fn = metric_fn
        self.n_repeats = n_repeats
        self.random_state = random_state

    def compute_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        predict_fn: Callable,
    ) -> np.ndarray:
        """Compute permutation-based importance.

        Args:
            X: Feature matrix (shape: n_samples, n_features)
            y: Target values
            predict_fn: Prediction function

        Returns:
            Feature importance scores
        """
        np.random.seed(self.random_state)
        n_features = X.shape[1]

        baseline_pred = predict_fn(X)
        baseline_score = self.metric_fn(y, baseline_pred)

        importances = np.zeros(n_features)

        for feature_idx in range(n_features):
            scores = []

            for _ in range(self.n_repeats):
                # Create permuted version
                X_permuted = X.copy()
                X_permuted[:, feature_idx] = np.random.permutation(X_permuted[:, feature_idx])

                # Compute degraded performance
                permuted_pred = predict_fn(X_permuted)
                permuted_score = self.metric_fn(y, permuted_pred)

                # Importance is performance degradation
                scores.append(permuted_score - baseline_score)

            importances[feature_idx] = np.mean(scores)

        # Normalize
        if np.sum(np.abs(importances)) > 0:
            importances = importances / np.sum(np.abs(importances))

        return importances


class ShapleyImportance(AttributionMethod):
    """Shapley value-based feature importance.

    Approximates Shapley values using Monte Carlo sampling.
    """

    def __init__(
        self,
        metric_fn: Callable | None = None,
        n_samples: int = 100,
        random_state: int = 42,
    ):
        """Initialize Shapley importance.

        Args:
            metric_fn: Function to compute performance metric
            n_samples: Number of Monte Carlo samples
            random_state: Random state for reproducibility
        """
        if metric_fn is None:
            from src.eval.metrics import mae

            metric_fn = mae

        self.metric_fn = metric_fn
        self.n_samples = n_samples
        self.random_state = random_state

    def compute_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        predict_fn: Callable,
    ) -> np.ndarray:
        """Compute Shapley value importance.

        Args:
            X: Feature matrix (shape: n_samples, n_features)
            y: Target values
            predict_fn: Prediction function

        Returns:
            Shapley value importance scores
        """
        np.random.seed(self.random_state)
        n_features = X.shape[1]

        shapley_values = np.zeros(n_features)

        for _ in range(self.n_samples):
            # Random feature ordering
            feature_order = np.random.permutation(n_features)

            # Compute marginal contributions
            for _i, feature_idx in enumerate(feature_order):
                # Prediction without feature
                X_without = X.copy()
                X_without[:, feature_idx] = np.mean(X[:, feature_idx])
                pred_without = predict_fn(X_without)
                score_without = self.metric_fn(y, pred_without)

                # Prediction with feature
                pred_with = predict_fn(X)
                score_with = self.metric_fn(y, pred_with)

                # Marginal contribution
                marginal_contrib = score_with - score_without
                shapley_values[feature_idx] += marginal_contrib

        # Average over samples and normalize
        shapley_values = shapley_values / self.n_samples

        if np.sum(np.abs(shapley_values)) > 0:
            shapley_values = shapley_values / np.sum(np.abs(shapley_values))

        return shapley_values


class LagImportance:
    """Importance of lagged features for time series."""

    def __init__(self, metric_fn: Callable | None = None):
        """Initialize lag importance.

        Args:
            metric_fn: Function to compute performance metric
        """
        if metric_fn is None:
            from src.eval.metrics import mae

            metric_fn = mae

        self.metric_fn = metric_fn

    def compute_lag_importance(
        self,
        series: np.ndarray,
        max_lag: int = 50,
        predict_fn: Callable | None = None,
    ) -> dict:
        """Compute importance of different lags.

        Args:
            series: Time series data
            max_lag: Maximum lag to consider
            predict_fn: Optional prediction function for model-based importance

        Returns:
            Dictionary with lag importance scores
        """
        lag_importance = {}

        # Correlation-based importance (if no predict_fn)
        if predict_fn is None:
            for lag in range(1, min(max_lag + 1, len(series) // 2)):
                correlation = np.corrcoef(series[:-lag], series[lag:])[0, 1]
                lag_importance[lag] = abs(correlation)
        else:
            # Model-based importance
            baseline_pred = predict_fn(series)
            baseline_score = self.metric_fn(series[1:], baseline_pred)

            for lag in range(1, min(max_lag + 1, len(series) // 2)):
                series_ablated = series.copy()
                series_ablated[:-lag] = 0  # Ablate lag

                ablated_pred = predict_fn(series_ablated)
                ablated_score = self.metric_fn(series[1:], ablated_pred)

                lag_importance[lag] = ablated_score - baseline_score

        return lag_importance


class AttributionAnalyzer:
    """Comprehensive attribution analyzer."""

    def __init__(self):
        """Initialize attribution analyzer."""
        self.results = {}

    def analyze(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        predict_fn: Callable,
        methods: list[str] | None = None,
        feature_names: list[str] | None = None,
    ) -> dict:
        """Perform attribution analysis.

        Args:
            X: Feature DataFrame
            y: Target values
            predict_fn: Prediction function
            methods: Attribution methods to use
            feature_names: Names of features

        Returns:
            Dictionary with attribution results
        """
        if methods is None:
            methods = ["ablation", "permutation"]

        if feature_names is None:
            feature_names = (
                list(X.columns)
                if hasattr(X, "columns")
                else [f"Feature_{i}" for i in range(X.shape[1])]
            )

        X_array = X.values if hasattr(X, "values") else X
        # Ensure X_array is an ndarray
        if isinstance(X_array, pd.DataFrame):
            X_array = X_array.to_numpy()
        elif not isinstance(X_array, np.ndarray):
            X_array = np.asarray(X_array)

        results = {}

        if "ablation" in methods:
            analyzer = AblationImportance()
            importances = analyzer.compute_importance(X_array, y, predict_fn)
            results["ablation"] = pd.Series(importances, index=feature_names)

        if "permutation" in methods:
            analyzer = PermutationImportance()
            importances = analyzer.compute_importance(X_array, y, predict_fn)
            results["permutation"] = pd.Series(importances, index=feature_names)

        if "shapley" in methods:
            analyzer = ShapleyImportance()
            importances = analyzer.compute_importance(X_array, y, predict_fn)
            results["shapley"] = pd.Series(importances, index=feature_names)

        self.results = results
        return results

    def summary(self) -> pd.DataFrame:
        """Get summary of attribution results.

        Returns:
            DataFrame with attribution results
        """
        return pd.DataFrame(self.results)

    def plot_importance(
        self,
        method: str = "ablation",
        top_n: int = 10,
    ) -> None:
        """Plot feature importance.

        Args:
            method: Attribution method to plot
            top_n: Number of top features to show
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available")
            return

        if method not in self.results:
            print(f"Method {method} not found in results")
            return

        importances = self.results[method].sort_values(ascending=False)[:top_n]

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(importances)), importances.values)
        plt.yticks(range(len(importances)), importances.index)
        plt.xlabel("Importance Score")
        plt.title(f"Feature Importance - {method}")
        plt.tight_layout()
        plt.show()
