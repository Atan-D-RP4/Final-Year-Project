"""Configuration management for the forecasting system."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataConfig:
    """Configuration for data sources and processing."""

    # Data sources
    fred_api_key: str | None = None
    start_date: str = "2010-01-01"
    end_date: str = "2024-12-31"

    # Target variables
    targets: list[str] | None = None

    # Covariate groups
    market_symbols: list[str] | None = None
    fred_series: list[str] | None = None

    # Data processing
    frequency: str = "D"  # Daily
    max_missing_ratio: float = 0.1

    def __post_init__(self) -> None:
        """Set default values after initialization."""
        if self.targets is None:
            self.targets = ["^GSPC", "^VIX"]  # S&P 500, VIX

        if self.market_symbols is None:
            self.market_symbols = [
                "^GSPC",  # S&P 500
                "^VIX",  # VIX
                "^TNX",  # 10-Year Treasury Note Yield
                "GLD",  # Gold ETF
                "DXY",  # US Dollar Index
            ]

        if self.fred_series is None:
            self.fred_series = [
                "DGS10",  # 10-Year Treasury Constant Maturity Rate
                "DGS2",  # 2-Year Treasury Constant Maturity Rate
                "UNRATE",  # Unemployment Rate
                "CPIAUCSL",  # Consumer Price Index
                "FEDFUNDS",  # Federal Funds Rate
            ]


@dataclass
class ModelConfig:
    """Configuration for model training and evaluation."""

    # Model parameters
    model_name: str = "chronos-t5-small"
    context_length: int = 512
    prediction_length: int = 24

    # Tokenization
    num_bins: int = 1024
    tokenization_method: str = "uniform"  # uniform, quantile, kmeans

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 10
    early_stopping_patience: int = 3

    # Evaluation
    test_size: float = 0.2
    validation_size: float = 0.1
    walk_forward_steps: int = 12


@dataclass
class ExperimentConfig:
    """Configuration for experiments and evaluation."""

    # Experiment tracking
    experiment_name: str = "multivariate_forecasting"
    use_wandb: bool = False
    wandb_project: str = "financial-forecasting"

    # Evaluation metrics
    metrics: list[str] | None = None

    # Attribution methods
    attribution_methods: list[str] | None = None

    def __post_init__(self) -> None:
        """Set default values after initialization."""
        if self.metrics is None:
            self.metrics = [
                "mae",
                "rmse",
                "mase",
                "smape",
                "crps",
                "directional_accuracy",
            ]

        if self.attribution_methods is None:
            self.attribution_methods = ["ablation", "permutation", "shap"]


@dataclass
class Config:
    """Main configuration class."""

    # Sub-configurations
    data: DataConfig
    model: ModelConfig
    experiment: ExperimentConfig

    # Paths
    project_root: Path
    data_dir: Path
    models_dir: Path
    results_dir: Path

    def __init__(self, project_root: Path | None = None, **kwargs: dict) -> None:
        """Initialize configuration."""
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent

        self.project_root = project_root
        self.data_dir = project_root / "data"
        self.models_dir = project_root / "models"
        self.results_dir = project_root / "results"

        # Initialize sub-configurations
        self.data = DataConfig(**kwargs.get("data", {}))
        self.model = ModelConfig(**kwargs.get("model", {}))
        self.experiment = ExperimentConfig(**kwargs.get("experiment", {}))

        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)


def load_config(config_path: Path | None = None) -> Config:
    """Load configuration from file or create default."""
    if config_path and config_path.exists():
        # TODO: Implement config loading from file (JSON/YAML)
        pass

    return Config()
