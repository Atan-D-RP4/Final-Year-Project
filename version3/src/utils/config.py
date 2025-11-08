"""Configuration management for financial forecasting project."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class DataConfig:
    """Configuration for data sources and processing."""

    fred_api_key: Optional[str] = None
    start_date: str = "2010-01-01"
    end_date: str = "2024-12-31"

    # Target variables
    targets: list[str] = field(default_factory=lambda: ["^GSPC", "^VIX"])

    # Market data sources
    market_symbols: list[str] = field(
        default_factory=lambda: [
            "^GSPC",  # S&P 500
            "^VIX",   # VIX
            "^TNX",   # 10-Year Treasury
            "GLD",    # Gold ETF
            "DXY",    # US Dollar Index
        ]
    )

    # Economic indicators from FRED
    fred_series: list[str] = field(
        default_factory=lambda: [
            "DGS10",     # 10-Year Treasury Rate
            "DGS2",      # 2-Year Treasury Rate
            "UNRATE",    # Unemployment Rate
            "CPIAUCSL",  # Consumer Price Index
            "FEDFUNDS",  # Federal Funds Rate
        ]
    )

    # Processing parameters
    frequency: str = "D"  # Daily
    max_missing_ratio: float = 0.1
    cache_dir: str = "data/raw"


@dataclass
class PreprocessingConfig:
    """Configuration for tokenization and feature engineering."""

    # Tokenization settings
    num_bins: int = 1024
    tokenization_method: str = "quantile"  # uniform, quantile, kmeans
    context_length: int = 512

    # Feature engineering
    include_technical_indicators: bool = True
    include_time_features: bool = True
    lag_features: list[int] = field(default_factory=lambda: [1, 5, 20])
    rolling_windows: list[int] = field(default_factory=lambda: [5, 10, 20])


@dataclass
class ModelConfig:
    """Configuration for model training and inference."""

    # Model selection
    model_name: str = "amazon/chronos-t5-small"
    chronos_variants: list[str] = field(
        default_factory=lambda: ["small", "base"]
    )
    prediction_length: int = 24
    context_length: int = 512

    # Tokenization for model
    num_bins: int = 1024
    tokenization_method: str = "quantile"

    # Training hyperparameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 10
    early_stopping_patience: int = 3
    validation_split: float = 0.1

    # Fine-tuning settings
    fine_tune: bool = True
    fine_tune_epochs: int = 5
    fine_tune_lr: float = 1e-5

    # Device settings
    device: str = "cuda"  # cuda, cpu, mps
    mixed_precision: bool = True


@dataclass
class EvalConfig:
    """Configuration for evaluation settings."""

    # Metrics to compute
    metrics: list[str] = field(
        default_factory=lambda: [
            "mae",
            "rmse",
            "mase",
            "smape",
            "directional_accuracy",
        ]
    )

    # Train/test split
    test_size: float = 0.2
    validation_size: float = 0.1

    # Walk-forward validation
    use_walk_forward: bool = True
    walk_forward_steps: int = 12

    # Prediction horizons to evaluate
    horizons: list[int] = field(default_factory=lambda: [1, 5, 10, 20])


@dataclass
class AttributionConfig:
    """Configuration for attribution and analysis."""

    # Attribution methods
    methods: list[str] = field(
        default_factory=lambda: [
            "ablation",
            "permutation",
        ]
    )

    # Settings
    num_samples: int = 100
    random_seed: int = 42
    include_shap: bool = False


@dataclass
class ExperimentConfig:
    """Configuration for experiment execution."""

    # Tracking
    experiment_name: str = "multivariate_forecasting"
    use_wandb: bool = False
    wandb_project: str = "financial-forecasting"
    log_level: str = "INFO"

    # Data
    data: DataConfig = field(default_factory=DataConfig)

    # Preprocessing
    preprocessing: PreprocessingConfig = field(
        default_factory=PreprocessingConfig
    )

    # Model
    model: ModelConfig = field(default_factory=ModelConfig)

    # Evaluation
    eval: EvalConfig = field(default_factory=EvalConfig)

    # Attribution
    attribution: AttributionConfig = field(
        default_factory=AttributionConfig
    )

    # Paths
    data_dir: str = "data"
    results_dir: str = "results"
    checkpoint_dir: str = "checkpoints"

    # Reproducibility
    random_seed: int = 42
    deterministic: bool = True


def load_config(config_path: Optional[str] = None) -> ExperimentConfig:
    """Load configuration from file or use defaults.

    Args:
        config_path: Path to YAML config file. If None, uses defaults.

    Returns:
        ExperimentConfig instance
    """
    if config_path is None or not Path(config_path).exists():
        return ExperimentConfig()

    with open(config_path) as f:
        config_dict = yaml.safe_load(f) or {}

    return ExperimentConfig(**config_dict)


def save_config(config: ExperimentConfig, output_path: str) -> None:
    """Save configuration to YAML file.

    Args:
        config: ExperimentConfig to save
        output_path: Path to save YAML file
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    config_dict = {
        "experiment_name": config.experiment_name,
        "data": {
            "fred_api_key": config.data.fred_api_key,
            "start_date": config.data.start_date,
            "end_date": config.data.end_date,
            "targets": config.data.targets,
            "frequency": config.data.frequency,
            "max_missing_ratio": config.data.max_missing_ratio,
        },
        "model": {
            "model_name": config.model.model_name,
            "prediction_length": config.model.prediction_length,
            "num_bins": config.model.num_bins,
        },
        "eval": {
            "test_size": config.eval.test_size,
            "use_walk_forward": config.eval.use_walk_forward,
            "walk_forward_steps": config.eval.walk_forward_steps,
        },
    }

    with open(output_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)
