"""Fine-tuning iteration module with comprehensive observability and analysis.

This module enables systematic iteration on model fine-tuning with detailed tracking
of hyperparameters, metrics, and data insights to identify what helps and hurts
model performance.

Key features:
- Grid search over hyperparameters
- Experiment tracking and logging
- Data insights (regime detection, feature analysis)
- Comparative analysis across experiments
- Visualization-ready output
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from src.eval.metrics import ForecastEvaluator
from src.models.chronos_wrapper import ChronosFineTuner
from src.utils.logger import setup_logger


@dataclass
class HyperparameterConfig:
    """Hyperparameter configuration for fine-tuning."""

    learning_rate: float = 1e-4
    epochs: int = 5
    context_length: int = 512
    prediction_length: int = 24
    batch_size: int = 32
    warmup_steps: int = 0
    weight_decay: float = 0.01
    adapter_type: str = "lora"
    lora_rank: int = 8
    lora_alpha: float = 16.0
    dropout_rate: float = 0.1


@dataclass
class ExperimentResult:
    """Results from a single fine-tuning experiment."""

    experiment_id: str
    timestamp: str
    hyperparams: dict[str, Any]
    train_metrics: dict[str, float]
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]
    data_insights: dict[str, Any]
    training_history: list[dict[str, float]]
    notes: str = ""
    model_path: str = ""
    error: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class DataAnalyzer:
    """Analyzes data characteristics and regimes."""

    @staticmethod
    def compute_volatility(data: pd.Series, window: int = 20) -> np.ndarray:
        """Compute rolling volatility."""
        returns = data.pct_change().dropna()
        volatility = returns.rolling(window=window).std() * np.sqrt(252)
        return cast(np.ndarray, volatility.values)

    @staticmethod
    def detect_regimes(data: pd.Series, n_regimes: int = 2) -> dict[str, Any]:
        """Detect market regimes based on volatility."""
        volatility = DataAnalyzer.compute_volatility(data)

        # Simple regime detection: high vol vs low vol
        vol_threshold = np.median(volatility)

        high_vol_indices = volatility > vol_threshold
        low_vol_indices = volatility <= vol_threshold

        return {
            "volatility_threshold": float(vol_threshold),
            "high_vol_periods": int(np.sum(high_vol_indices)),
            "low_vol_periods": int(np.sum(low_vol_indices)),
            "volatility_mean": float(np.mean(volatility)),
            "volatility_std": float(np.std(volatility)),
            "volatility_min": float(np.min(volatility)),
            "volatility_max": float(np.max(volatility)),
        }

    @staticmethod
    def compute_correlations(data: pd.DataFrame) -> dict[str, float]:
        """Compute feature correlations with target."""
        if data.shape[1] < 2:
            return {}

        target_col = data.columns[0]
        correlations = {}

        for col in data.columns[1:]:
            if data[col].dtype in [np.float64, np.float32, int, np.int64]:
                try:
                    target_series = data[target_col]
                    col_series = data[col]
                    if isinstance(target_series, pd.Series) and isinstance(col_series, pd.Series):
                        corr = target_series.corr(col_series)  # type: ignore
                        if not np.isnan(corr):
                            correlations[f"corr_with_{col}"] = float(corr)
                except Exception:
                    pass

        return correlations

    @staticmethod
    def analyze_data_quality(data: pd.DataFrame) -> dict[str, Any]:
        """Analyze data quality metrics."""
        return {
            "n_rows": len(data),
            "n_cols": len(data.columns),
            "missing_values": int(data.isnull().sum().sum()),  # type: ignore
            "missing_pct": float(data.isnull().sum().sum() / (len(data) * len(data.columns))),  # type: ignore
            "numeric_cols": int(data.select_dtypes(include=[np.number]).shape[1]),
            "date_range": f"{data.index[0]} to {data.index[-1]}"
            if hasattr(data.index[0], "isoformat")
            else "N/A",
        }

    @staticmethod
    def analyze_target_distribution(data: pd.Series) -> dict[str, float]:
        """Analyze target variable distribution."""
        returns = data.pct_change().dropna()

        return {
            "mean": float(returns.mean()),  # type: ignore
            "std": float(returns.std()),  # type: ignore
            "skewness": float(returns.skew()),  # type: ignore
            "kurtosis": float(returns.kurtosis()),  # type: ignore
            "min": float(returns.min()),  # type: ignore
            "max": float(returns.max()),  # type: ignore
            "median": float(returns.median()),  # type: ignore
        }


class ExperimentTracker:
    """Tracks and manages fine-tuning experiments."""

    def __init__(self, base_dir: str = "results/fine_tuning_experiments"):
        """Initialize experiment tracker.

        Args:
            base_dir: Base directory for storing experiment results
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger("fine_tuning_iterator")
        self.experiments: list[ExperimentResult] = []

    def create_experiment_id(self, base_name: str = "exp") -> str:
        """Create unique experiment ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base_name}_{timestamp}"

    def log_experiment(self, result: ExperimentResult) -> None:
        """Log experiment result.

        Args:
            result: ExperimentResult to log
        """
        self.experiments.append(result)
        self.logger.info(f"Logged experiment {result.experiment_id}")

    def save_experiment(self, result: ExperimentResult) -> Path:
        """Save experiment to disk.

        Args:
            result: ExperimentResult to save

        Returns:
            Path to saved result
        """
        exp_dir = self.base_dir / result.experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Save main results
        results_file = exp_dir / "results.json"
        with open(results_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

        self.logger.info(f"Saved experiment results to {results_file}")
        return results_file

    def load_experiments(self) -> list[ExperimentResult]:
        """Load all experiments from disk.

        Returns:
            List of ExperimentResult objects
        """
        experiments = []

        for exp_dir in self.base_dir.glob("exp_*"):
            results_file = exp_dir / "results.json"
            if results_file.exists():
                with open(results_file) as f:
                    data = json.load(f)
                    experiments.append(ExperimentResult(**data))

        self.experiments = experiments
        self.logger.info(f"Loaded {len(experiments)} experiments")
        return experiments

    def to_dataframe(self) -> pd.DataFrame:
        """Convert experiments to DataFrame for analysis.

        Returns:
            DataFrame with experiment results
        """
        rows = []

        for exp in self.experiments:
            row: dict[str, Any] = {"experiment_id": exp.experiment_id, "timestamp": exp.timestamp}

            # Flatten hyperparams
            for key, value in exp.hyperparams.items():
                row[f"hp_{key}"] = value

            # Flatten metrics
            for key, value in exp.test_metrics.items():
                row[f"test_{key}"] = value

            for key, value in exp.data_insights.items():
                row[f"insight_{key}"] = value

            if exp.error:
                row["error"] = exp.error

            rows.append(row)

        return pd.DataFrame(rows) if rows else pd.DataFrame()


class FineTuningIterator:
    """Main class for iterating on model fine-tuning."""

    def __init__(
        self,
        fine_tuner: ChronosFineTuner | None = None,
        base_dir: str = "results/fine_tuning_experiments",
    ):
        """Initialize fine-tuning iterator.

        Args:
            fine_tuner: ChronosFineTuner instance. If None, creates one.
            base_dir: Base directory for saving results
        """
        self.fine_tuner = fine_tuner or ChronosFineTuner(results_dir=base_dir)
        self.tracker = ExperimentTracker(base_dir)
        self.data_analyzer = DataAnalyzer()
        self.evaluator = ForecastEvaluator()
        self.logger = setup_logger("fine_tuning_iterator")

    def run_experiment(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        test_data: pd.DataFrame,
        target_col: str,
        hyperparams: HyperparameterConfig | dict | None = None,
        experiment_name: str = "exp",
        notes: str = "",
    ) -> ExperimentResult:
        """Run a single fine-tuning experiment.

        Args:
            train_data: Training data
            val_data: Validation data
            test_data: Test data
            target_col: Target column name
            hyperparams: Hyperparameter configuration
            experiment_name: Base name for experiment
            notes: Experiment notes

        Returns:
            ExperimentResult with metrics and insights
        """
        # Setup hyperparameters
        if hyperparams is None:
            config = HyperparameterConfig()
        elif isinstance(hyperparams, dict):
            config = HyperparameterConfig(**hyperparams)
        else:
            config = hyperparams

        # Create experiment ID
        exp_id = self.tracker.create_experiment_id(experiment_name)
        timestamp = datetime.now().isoformat()

        self.logger.info(f"Starting experiment {exp_id}")
        self.logger.info(f"Hyperparameters: {asdict(config)}")

        try:
            # Analyze data before fine-tuning
            self.logger.info("Analyzing data...")
            data_insights = self._analyze_data(train_data, val_data, test_data, target_col)

            # Load base model
            self.fine_tuner.load_base_model()
            self.fine_tuner.prepare_data(train_data, val_data, target_col)

            # Fine-tune
            self.logger.info("Starting fine-tuning...")
            ft_results = self.fine_tuner.fine_tune(
                train_data,
                val_data,
                target_col=target_col,
            )

            # Evaluate
            self.logger.info("Evaluating model...")
            test_metrics, train_metrics, val_metrics = self._evaluate_model(
                self.fine_tuner, train_data, val_data, test_data, target_col
            )

            # Extract training history
            training_history = self._extract_training_history(ft_results)

            # Create result
            result = ExperimentResult(
                experiment_id=exp_id,
                timestamp=timestamp,
                hyperparams=asdict(config),
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
                data_insights=data_insights,
                training_history=training_history,
                notes=notes,
                model_path=ft_results.get("model_path", ""),
            )

            # Log and save
            self.tracker.log_experiment(result)
            self.tracker.save_experiment(result)

            self.logger.info(f"Experiment {exp_id} completed successfully")
            self.logger.info(f"Test metrics: {test_metrics}")

            return result

        except Exception as e:
            self.logger.error(f"Experiment {exp_id} failed: {e}")
            result = ExperimentResult(
                experiment_id=exp_id,
                timestamp=timestamp,
                hyperparams=asdict(config),
                train_metrics={},
                val_metrics={},
                test_metrics={},
                data_insights={},
                training_history=[],
                notes=notes,
                error=str(e),
            )
            self.tracker.log_experiment(result)
            return result

    def grid_search(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        test_data: pd.DataFrame,
        target_col: str,
        param_grid: dict[str, list],
        experiment_name: str = "grid_search",
    ) -> list[ExperimentResult]:
        """Run grid search over hyperparameters.

        Args:
            train_data: Training data
            val_data: Validation data
            test_data: Test data
            target_col: Target column name
            param_grid: Dictionary mapping parameter names to lists of values
            experiment_name: Base name for experiments

        Returns:
            List of ExperimentResult objects
        """
        self.logger.info(
            f"Starting grid search with {self._count_combinations(param_grid)} combinations"
        )

        # Generate all combinations
        import itertools

        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]

        results = []

        for i, values in enumerate(itertools.product(*param_values)):
            hyperparams = dict(zip(param_names, values, strict=True))
            notes = f"Grid search iteration {i + 1}"

            self.logger.info(f"Running iteration {i + 1}: {hyperparams}")

            result = self.run_experiment(
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                target_col=target_col,
                hyperparams=hyperparams,
                experiment_name=f"{experiment_name}_{i + 1:03d}",
                notes=notes,
            )

            results.append(result)

        self.logger.info(f"Grid search completed. Results saved to {self.tracker.base_dir}")
        return results

    def compare_experiments(self, experiment_ids: list[str] | None = None) -> pd.DataFrame:
        """Compare multiple experiments.

        Args:
            experiment_ids: List of experiment IDs to compare. If None, compares all.

        Returns:
            DataFrame with comparison
        """
        if experiment_ids is None:
            exps_to_compare = self.tracker.experiments
        else:
            exps_to_compare = [
                e for e in self.tracker.experiments if e.experiment_id in experiment_ids
            ]

        comparison_data = []

        for exp in exps_to_compare:
            row: dict[str, Any] = {
                "experiment_id": exp.experiment_id,
                "timestamp": exp.timestamp,
            }

            # Add hyperparameters
            for key, value in exp.hyperparams.items():
                row[f"hp_{key}"] = value

            # Add test metrics
            for key, value in exp.test_metrics.items():
                row[f"test_{key}"] = value

            # Add best metric indicator
            if "mae" in exp.test_metrics:
                row["test_mae"] = exp.test_metrics["mae"]

            if exp.error:
                row["error"] = exp.error

            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)
        return df.sort_values("test_mae") if "test_mae" in df.columns else df

    def identify_best_experiment(self, metric: str = "mae") -> ExperimentResult | None:
        """Find best experiment by metric.

        Args:
            metric: Metric to optimize (default: mae)

        Returns:
            Best ExperimentResult or None if no valid experiments
        """
        valid_exps = [e for e in self.tracker.experiments if not e.error]

        if not valid_exps:
            return None

        # Find best by minimizing metric (or maximizing if directional_accuracy)
        best = min(
            valid_exps,
            key=lambda e: -e.test_metrics.get(metric, float("inf"))
            if metric == "directional_accuracy"
            else e.test_metrics.get(metric, float("inf")),
        )

        return best

    def _analyze_data(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        test_data: pd.DataFrame,
        target_col: str,
    ) -> dict[str, Any]:
        """Analyze data characteristics.

        Args:
            train_data: Training data
            val_data: Validation data
            test_data: Test data
            target_col: Target column name

        Returns:
            Dictionary with data insights
        """
        insights: dict[str, Any] = {}

        # Overall data quality
        insights["train_quality"] = self.data_analyzer.analyze_data_quality(train_data)
        insights["val_quality"] = self.data_analyzer.analyze_data_quality(val_data)
        insights["test_quality"] = self.data_analyzer.analyze_data_quality(test_data)

        # Target distribution
        if target_col in train_data.columns:
            target_series = train_data[target_col]
            if isinstance(target_series, pd.Series):
                insights["target_distribution"] = self.data_analyzer.analyze_target_distribution(
                    target_series
                )

        # Regime detection
        if target_col in train_data.columns:
            train_series = train_data[target_col]
            test_series = test_data[target_col]
            if isinstance(train_series, pd.Series) and isinstance(test_series, pd.Series):
                insights["train_regimes"] = self.data_analyzer.detect_regimes(train_series)
                insights["test_regimes"] = self.data_analyzer.detect_regimes(test_series)

        # Correlations
        if train_data.shape[1] > 1:
            insights["correlations"] = self.data_analyzer.compute_correlations(train_data)

        return insights

    def _evaluate_model(
        self,
        fine_tuner: ChronosFineTuner,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        test_data: pd.DataFrame,
        target_col: str,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        """Evaluate model on train/val/test sets.

        Args:
            fine_tuner: Trained fine-tuner
            train_data: Training data
            val_data: Validation data
            test_data: Test data
            target_col: Target column name

        Returns:
            Tuple of (test_metrics, train_metrics, val_metrics)
        """
        metrics_to_compute = ["mae", "rmse", "mase", "directional_accuracy"]

        test_metrics = {}
        train_metrics = {}
        val_metrics = {}

        try:
            # Test evaluation
            test_forecasts = fine_tuner.forecast_fine_tuned(test_data, target_col, num_samples=100)
            test_median = test_forecasts.get("median") or test_forecasts.get("forecast") or []
            test_median = cast(list, test_median)
            test_actual = cast(np.ndarray, test_data[target_col].values[-len(test_median) :])
            test_pred = test_median

            if len(test_actual) > 0 and len(test_pred) > 0:
                test_metrics = self.evaluator.evaluate(
                    test_actual, np.array(test_pred), metrics=metrics_to_compute
                )

            # Train evaluation
            train_forecasts = fine_tuner.forecast_fine_tuned(
                train_data, target_col, num_samples=100
            )
            train_median = train_forecasts.get("median") or train_forecasts.get("forecast") or []
            train_median = cast(list, train_median)
            train_actual = cast(np.ndarray, train_data[target_col].values[-len(train_median) :])
            train_pred = train_median

            if len(train_actual) > 0 and len(train_pred) > 0:
                train_metrics = self.evaluator.evaluate(
                    train_actual, np.array(train_pred), metrics=metrics_to_compute
                )

            # Val evaluation
            val_forecasts = fine_tuner.forecast_fine_tuned(val_data, target_col, num_samples=100)
            val_median = val_forecasts.get("median") or val_forecasts.get("forecast") or []
            val_median = cast(list, val_median)
            val_actual = cast(np.ndarray, val_data[target_col].values[-len(val_median) :])
            val_pred = val_median

            if len(val_actual) > 0 and len(val_pred) > 0:
                val_metrics = self.evaluator.evaluate(
                    val_actual, np.array(val_pred), metrics=metrics_to_compute
                )

        except Exception as e:
            self.logger.warning(f"Evaluation failed: {e}")

        return test_metrics, train_metrics, val_metrics

    def _extract_training_history(self, ft_results: dict) -> list[dict[str, float]]:
        """Extract training history from fine-tuning results.

        Args:
            ft_results: Fine-tuning results dictionary

        Returns:
            List of per-epoch metrics
        """
        history = []

        if isinstance(ft_results.get("history"), list):
            history = ft_results["history"]
        elif isinstance(ft_results.get("history"), dict):
            # Convert dict to list of dicts
            keys = list(ft_results["history"].keys())
            if keys:
                values = list(ft_results["history"].values())
                if values and isinstance(values[0], (list, np.ndarray)):
                    for i in range(len(values[0])):
                        epoch_data: dict[str, float] = {
                            key: values[j][i] for j, key in enumerate(keys)
                        }
                        history.append(epoch_data)

        return history

    @staticmethod
    def _count_combinations(param_grid: dict) -> int:
        """Count number of combinations in parameter grid.

        Args:
            param_grid: Parameter grid dictionary

        Returns:
            Number of combinations
        """
        count = 1
        for values in param_grid.values():
            count *= len(values)
        return count


class ExperimentAnalyzer:
    """Analyzes fine-tuning experiments to identify patterns."""

    def __init__(self, experiments: list[ExperimentResult]):
        """Initialize analyzer.

        Args:
            experiments: List of ExperimentResult objects
        """
        self.experiments = experiments
        self.df = self._to_dataframe()

    def _to_dataframe(self) -> pd.DataFrame:
        """Convert experiments to DataFrame.

        Returns:
            DataFrame with flattened experiment data
        """
        rows = []

        for exp in self.experiments:
            row: dict[str, Any] = {
                "experiment_id": exp.experiment_id,
                "timestamp": exp.timestamp,
            }

            # Hyperparameters
            for key, value in exp.hyperparams.items():
                row[f"hp_{key}"] = value

            # Test metrics
            for key, value in exp.test_metrics.items():
                row[f"test_{key}"] = value

            rows.append(row)

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def identify_important_hyperparameters(
        self, metric: str = "mae", top_n: int = 5
    ) -> pd.DataFrame:
        """Identify most important hyperparameters by correlation with metric.

        Args:
            metric: Metric to analyze
            top_n: Number of top hyperparameters to return

        Returns:
            DataFrame with hyperparameter importance
        """
        metric_col = f"test_{metric}"

        if metric_col not in self.df.columns:
            return pd.DataFrame()

        hp_cols = [col for col in self.df.columns if col.startswith("hp_")]

        correlations: list[dict[str, Any]] = []

        for col in hp_cols:
            if self.df[col].dtype in [np.float64, np.float32, int, np.int64]:
                corr = float(self.df[metric_col].corr(self.df[col]))  # type: ignore
                if not np.isnan(corr):
                    correlations.append(
                        {"hyperparameter": col, "correlation": corr, "abs_correlation": abs(corr)}
                    )

        df_corr = pd.DataFrame(correlations).sort_values("abs_correlation", ascending=False)

        return df_corr.head(top_n)

    def detect_overfitting(self) -> pd.DataFrame:
        """Detect experiments with overfitting (large train-val gap).

        Returns:
            DataFrame with overfitting indicators
        """
        results = []

        for exp in self.experiments:
            if not exp.error and exp.train_metrics and exp.val_metrics:
                train_mae = exp.train_metrics.get("mae", 0)
                val_mae = exp.val_metrics.get("mae", 0)

                if train_mae > 0:
                    overfitting_ratio = val_mae / train_mae if train_mae > 0 else 0
                    results.append(
                        {
                            "experiment_id": exp.experiment_id,
                            "train_mae": train_mae,
                            "val_mae": val_mae,
                            "overfitting_ratio": overfitting_ratio,
                            "overfitting_flag": overfitting_ratio > 1.3,  # 30% threshold
                        }
                    )

        return pd.DataFrame(results) if results else pd.DataFrame()

    def identify_convergence_issues(self) -> pd.DataFrame:
        """Identify experiments with convergence issues.

        Returns:
            DataFrame with convergence indicators
        """
        results = []

        for exp in self.experiments:
            if exp.training_history:
                history_df = pd.DataFrame(exp.training_history)

                if "loss" in history_df.columns:
                    losses = cast(np.ndarray, history_df["loss"].values)
                    # Check if loss is decreasing
                    diffs = np.diff(losses)
                    increasing_pct = float(np.sum(diffs > 0) / len(diffs)) if len(diffs) > 0 else 0

                    results.append(
                        {
                            "experiment_id": exp.experiment_id,
                            "final_loss": float(losses[-1]),
                            "initial_loss": float(losses[0]),
                            "loss_reduction": float(losses[0] - losses[-1]),
                            "increasing_pct": increasing_pct,
                            "convergence_issue": increasing_pct > 0.3,
                        }
                    )

        return pd.DataFrame(results) if results else pd.DataFrame()

    def get_summary_statistics(self) -> dict[str, Any]:
        """Get summary statistics across all experiments.

        Returns:
            Dictionary with summary statistics
        """
        summary: dict[str, Any] = {
            "total_experiments": len(self.experiments),
            "successful_experiments": sum(1 for e in self.experiments if not e.error),
            "failed_experiments": sum(1 for e in self.experiments if e.error),
        }

        # Best metrics
        valid_exps = [e for e in self.experiments if not e.error and e.test_metrics]

        if valid_exps:
            maes = [e.test_metrics.get("mae", float("inf")) for e in valid_exps]
            dir_accs = [e.test_metrics.get("directional_accuracy", 0) for e in valid_exps]

            summary["best_mae"] = float(min(maes))
            summary["worst_mae"] = float(max(maes))
            summary["mean_mae"] = float(np.mean([m for m in maes if m != float("inf")]))
            summary["best_directional_accuracy"] = float(max(dir_accs))
            summary["mean_directional_accuracy"] = float(np.mean(dir_accs))

        return summary

    def export_results(self, output_file: str) -> None:
        """Export results to CSV.

        Args:
            output_file: Path to output CSV file
        """
        self.df.to_csv(output_file, index=False)
