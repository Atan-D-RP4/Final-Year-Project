# Financial Forecasting with Chronos

A comprehensive Python ML project for multivariate financial forecasting using Chronos models, with support for zero-shot and fine-tuned approaches, baseline comparisons, and attribution analysis.

## Features

- **Chronos Models**: Zero-shot and fine-tuned Chronos T5 models for financial time series forecasting
- **Baseline Models**: Multiple baseline forecasters including ARIMA, VAR, LSTM, Linear Regression, and ensemble methods
- **Evaluation Metrics**: Comprehensive evaluation suite including point forecast and probabilistic metrics
- **Attribution Analysis**: Feature importance analysis using ablation, permutation, and Shapley value methods
- **Tokenization**: Multiple tokenization strategies (uniform, quantile, k-means) for financial data
- **Data Integration**: Automatic data fetching from Yahoo Finance and FRED API

## Project Structure

```
src/
├── data/           # Data fetching and preprocessing
├── preprocessing/  # Tokenization strategies
├── models/         # Forecasting models (Chronos, baselines)
├── eval/           # Evaluation metrics and utilities
├── analysis/       # Attribution methods
└── utils/          # Configuration and logging

experiments/
├── phase3/         # Zero-shot vs baseline comparisons
└── phase4/         # Fine-tuning and attribution analysis
```

## Installation

Install dependencies using `uv`:

```bash
uv sync
```

## Usage

### Phase 3: Zero-Shot Forecasting

Run zero-shot comparison between Chronos and baseline models:

```bash
cd experiments/phase3
python zero_shot.py
```

Results are saved to `results/phase3/` including:
- Metrics comparison (CSV)
- Forecast visualizations (PNG)
- Detailed results (JSON)

### Basic Forecasting

```python
from src.models.chronos_wrapper import ChronosFinancialForecaster
from src.data.fetchers import DataFetcher
from src.data.cleaning import DataCleaner, create_features
from src.utils.config import DataConfig

# Fetch and prepare data
fetcher = DataFetcher()
data = fetcher.fetch_all_data(['AAPL'], [], '2023-01-01', '2024-01-01')
cleaner = DataCleaner()
data = cleaner.clean(data['market'])
data = create_features(data)

# Train and forecast
forecaster = ChronosFinancialForecaster()
forecaster.fit(data, 'Close')
forecasts = forecaster.forecast_zero_shot(data, 'Close', num_samples=100)
```

## Configuration

Configuration is managed through dataclasses in `src/utils/config.py`:

- `DataConfig`: Data sources, date ranges, and features
- `PreprocessingConfig`: Tokenization and feature engineering settings
- `ModelConfig`: Model selection and hyperparameters
- `EvalConfig`: Evaluation metrics and validation strategies
- `AttributionConfig`: Attribution analysis settings

## Citation

This project implements Chronos models as described in:
- [Chronos: Pretrained (Language) Models for Time Series Forecasting](https://arxiv.org/abs/2403.07690)

## License

MIT License
