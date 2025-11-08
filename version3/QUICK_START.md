# Quick Reference Guide

## Running the Project

### Installation

```bash
# Install dependencies
uv sync

# Activate environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows
```

### Running Phase 3 Zero-Shot Experiments

```bash
cd experiments/phase3
python zero_shot.py
```

Results will be saved to `results/phase3/`

### Running Tests

```bash
# Run all smoke tests
pytest tests/test_smoke.py -v

# Run specific test class
pytest tests/test_smoke.py::TestBaselineModels -v

# Run specific test
pytest tests/test_smoke.py::TestBaselineModels::test_naive_forecaster -v

# Run with coverage
pytest tests/test_smoke.py --cov=src --cov-report=html
```

---

## Code Structure Quick Reference

### 1. Using Baseline Models

```python
from src.models.baselines import NaiveForecaster, ARIMAForecaster
import pandas as pd

# Create model
model = NaiveForecaster()

# Fit on training data
model.fit(train_df, 'Close')

# Forecast
predictions = model.forecast(test_df, 'Close', horizon=20)
```

### 2. Using Chronos Model

```python
from src.models.chronos_wrapper import ChronosFinancialForecaster

# Initialize
model = ChronosFinancialForecaster(prediction_length=20, device='cpu')

# Fit tokenizer
model.fit(train_df, 'Close')

# Zero-shot forecast
forecasts = model.forecast_zero_shot(test_df, 'Close', num_samples=100)
median_forecast = forecasts['median']
```

### 3. Evaluation

```python
from src.eval.metrics import ForecastEvaluator, calculate_all_metrics

# Method 1: Using evaluator class
evaluator = ForecastEvaluator()
metrics = evaluator.evaluate(actual, predicted, metrics=['mae', 'rmse'])

# Method 2: Direct function
metrics = calculate_all_metrics(actual, predicted)

# Compare models
comparison = evaluator.compare_models(
    {'Model A': pred_a, 'Model B': pred_b},
    actual
)
```

### 4. Attribution Analysis

```python
from src.analysis import AttributionAnalyzer

analyzer = AttributionAnalyzer()
results = analyzer.analyze(
    X=features_df,
    y=targets,
    predict_fn=model.predict,
    methods=['ablation', 'permutation']
)

summary = analyzer.summary()
```

### 5. Data Fetching

```python
from src.data.fetchers import DataFetcher
from src.data.cleaning import DataCleaner, create_features

# Fetch data
fetcher = DataFetcher()
data_dict = fetcher.fetch_all_data(
    market_symbols=['AAPL', 'MSFT'],
    fred_series=['DEXUSEU'],  # EUR/USD
    start_date='2023-01-01',
    end_date='2024-01-01'
)

# Clean and prepare
cleaner = DataCleaner()
data = cleaner.clean(data_dict['market'])
data = create_features(data)
```

### 6. Tokenization

```python
from src.preprocessing.tokenizer import FinancialDataTokenizer, AdvancedTokenizer

# Basic tokenizer
tokenizer = FinancialDataTokenizer(method='quantile', num_bins=100)
tokenizer.fit(train_df[['Close']])
tokens = tokenizer.transform(test_df[['Close']])

# Advanced tokenizer with features
tokenizer = AdvancedTokenizer(
    include_technical_indicators=True,
    include_time_features=True
)
```

### 7. Configuration

```python
from src.utils.config import DataConfig, PreprocessingConfig, ModelConfig

# Data config
data_cfg = DataConfig(
    market_symbols=['AAPL'],
    start_date='2023-01-01',
    end_date='2024-01-01'
)

# Preprocessing config
prep_cfg = PreprocessingConfig(
    tokenization_method='quantile',
    num_bins=1024
)

# Model config
model_cfg = ModelConfig(
    model_type='chronos',
    prediction_length=20
)
```

### 8. Logging

```python
from src.utils.logger import setup_logger

logger = setup_logger('my_script', 'logs/script.log')
logger.info('Starting process')
logger.error('An error occurred')
```

---

## File Organization

### Core Modules

**Data** (`src/data/`)
- `fetchers.py`: Data retrieval
- `cleaning.py`: Preprocessing

**Preprocessing** (`src/preprocessing/`)
- `tokenizer.py`: Tokenization strategies

**Models** (`src/models/`)
- `chronos_wrapper.py`: Chronos integration
- `baselines.py`: 10+ baseline models

**Evaluation** (`src/eval/`)
- `metrics.py`: 15+ metrics

**Analysis** (`src/analysis/`)
- `__init__.py`: Attribution methods

**Utils** (`src/utils/`)
- `config.py`: Configuration
- `logger.py`: Logging

### Experiments

**Phase 3** (`experiments/phase3/`)
- `zero_shot.py`: Zero-shot comparison experiments

---

## Key Classes

### Models
- `ChronosFinancialForecaster`: Main Chronos wrapper
- `NaiveForecaster`, `MeanForecaster`, `SeasonalNaiveForecaster`
- `ARIMAForecaster`, `VARForecaster`
- `LinearRegressionForecaster`, `LSTMForecaster`
- `EnsembleForecaster`

### Evaluation
- `ForecastEvaluator`: Main evaluation orchestrator
- Functions: `mae()`, `rmse()`, `mase()`, `mape()`, `smape()`, `directional_accuracy()`

### Attribution
- `AttributionAnalyzer`: Main analyzer
- `AblationImportance`, `PermutationImportance`, `ShapleyImportance`
- `LagImportance`

### Data
- `DataFetcher`: Main fetcher
- `DataCleaner`: Cleaning utilities
- `FinancialDataTokenizer`: Basic tokenizer
- `AdvancedTokenizer`: Enhanced tokenizer

### Configuration
- `DataConfig`: Data settings
- `PreprocessingConfig`: Preprocessing settings
- `ModelConfig`: Model settings
- `EvalConfig`: Evaluation settings
- `AttributionConfig`: Attribution settings

---

## Common Workflows

### Workflow 1: Compare Models on Test Data

```python
from src.models.baselines import NaiveForecaster, ARIMAForecaster, LSTMForecaster
from src.models.chronos_wrapper import ChronosFinancialForecaster
from src.eval.metrics import ForecastEvaluator

# Initialize models
models = {
    'Naive': NaiveForecaster(),
    'ARIMA': ARIMAForecaster(),
    'LSTM': LSTMForecaster(device='cpu'),
    'Chronos': ChronosFinancialForecaster(device='cpu'),
}

# Train all
for name, model in models.items():
    model.fit(train_data, 'Close')

# Forecast
predictions = {}
for name, model in models.items():
    predictions[name] = model.forecast(test_data, 'Close', 20)

# Evaluate
evaluator = ForecastEvaluator()
comparison = evaluator.compare_models(predictions, actual)
print(comparison)
```

### Workflow 2: Full Pipeline from Fetching to Evaluation

```python
from src.data.fetchers import DataFetcher
from src.data.cleaning import DataCleaner, create_features
from src.models.chronos_wrapper import ChronosFinancialForecaster
from src.eval.metrics import ForecastEvaluator

# Fetch
fetcher = DataFetcher()
data = fetcher.fetch_all_data(['AAPL'], [], '2023-01-01', '2024-01-01')['market']

# Clean
cleaner = DataCleaner()
data = cleaner.clean(data)
data = create_features(data)

# Split
split_idx = int(len(data) * 0.8)
train = data.iloc[:split_idx]
test = data.iloc[split_idx:]

# Model
model = ChronosFinancialForecaster(prediction_length=20)
model.fit(train, 'Close')
forecasts = model.forecast_zero_shot(test, 'Close')

# Evaluate
actual = test['Close'].values[-20:]
evaluator = ForecastEvaluator()
metrics = evaluator.evaluate(actual, forecasts['median'])
print(metrics)
```

### Workflow 3: Feature Attribution Analysis

```python
from src.analysis import AttributionAnalyzer
from src.models.baselines import LinearRegressionForecaster

# Train model
model = LinearRegressionForecaster(lags=20)
model.fit(train_data, 'Close')

# Prepare features (lagged values)
features = train_data[['Close']].shift(i) for i in range(1, 21)
targets = train_data['Close']

# Analyze
analyzer = AttributionAnalyzer()
results = analyzer.analyze(
    features,
    targets,
    predict_fn=model.predict,
    methods=['ablation', 'permutation', 'shapley']
)

summary = analyzer.summary()
print(summary)
```

---

## Environment Setup

### Dependencies File Structure

Dependencies specified in `pyproject.toml`:
- Core: pandas, numpy, torch, scikit-learn, statsmodels, transformers
- Data: yfinance, fredapi
- Viz: matplotlib, seaborn
- Dev: pytest, ruff, pyrefly

### Virtual Environment

```bash
# Create
python -m venv .venv

# Activate (Linux/Mac)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

# Install with uv
uv sync
```

---

## Debugging Tips

### Import Errors
- Ensure `uv sync` has been run
- Check that you're in the correct directory
- Verify `PYTHONPATH` includes project root

### Model Training Errors
- Check data shape and types
- Ensure sufficient data (>30 samples for most models)
- LSTM needs GPU for optimal performance, but CPU works for small data

### Forecast Evaluation Errors
- Ensure actual and predicted same length
- Check for NaN values: `np.isnan(predictions).any()`
- Verify metric functions receive correct dtypes (float arrays)

### Memory Issues
- Reduce batch size in LSTM
- Use smaller models
- Process data in chunks

---

## Performance Tips

1. **Data Fetching**: Cache results to avoid repeated API calls
2. **LSTM Training**: Use GPU with CUDA for >1000 samples
3. **Attribution**: Use fewer Monte Carlo samples for quick estimates
4. **Evaluation**: Pre-compute common metrics together

---

## Output Formats

### Phase 3 Results

```
results/phase3/
├── zero_shot_default_metrics.csv
│   # Model | MAE | RMSE | MASE | ... (one row per model)
│
├── zero_shot_default_forecasts.csv
│   # actual | Naive | ARIMA | Chronos | ... (one row per timestep)
│
├── zero_shot_default_results.json
│   # {
│   #   "timestamp": "...",
│   #   "target_col": "Close",
│   #   "model_results": { "Naive": {metrics}, ... },
│   #   "forecasts": { "Naive": [vals], ... },
│   #   "actual": [vals]
│   # }
│
├── zero_shot_default_forecasts.png
│   # Time series plot with all model predictions
│
└── zero_shot_default_metrics.png
    # Bar charts for each metric comparing models
```

---

## Where to Find Things

- **Download financial data**: See `src/data/fetchers.py`
- **Pre-process/tokenize**: See `src/preprocessing/tokenizer.py`
- **Compare models**: See `src/eval/metrics.py`
- **Run experiments**: See `experiments/phase3/zero_shot.py`
- **Test everything**: See `tests/test_smoke.py`
- **Configure settings**: See `src/utils/config.py`

---

**For more details, see SESSION_SUMMARY.md**
