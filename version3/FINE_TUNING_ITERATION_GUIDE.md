# Fine-Tuning Iteration Guide

A comprehensive guide to using the `fine_tuning_iterator` module for systematic model fine-tuning with built-in observability and analysis.

## Overview

The `fine_tuning_iterator` module enables systematic iteration on model fine-tuning with detailed tracking of:

- **Hyperparameter combinations** via grid search
- **Experiment metrics** (MAE, RMSE, MASE, directional accuracy)
- **Data insights** (volatility, regime detection, feature correlations)
- **Training dynamics** (per-epoch loss curves, convergence analysis)
- **Comparative analysis** across experiments to identify what drives performance

This guide covers practical workflows for hypothesis-driven hyperparameter tuning and post-hoc experiment analysis.

## Quick Start

### Installation

The module is part of the `src.models` package:

```python
from src.models.fine_tuning_iterator import (
    FineTuningIterator,
    HyperparameterConfig,
    ExperimentAnalyzer,
)
```

### Basic Workflow

```python
import pandas as pd
from src.models.fine_tuning_iterator import FineTuningIterator

# Load your data (must be pandas DataFrames with DatetimeIndex)
train_data = pd.read_csv('train.csv', index_col='date', parse_dates=['date'])
val_data = pd.read_csv('val.csv', index_col='date', parse_dates=['date'])
test_data = pd.read_csv('test.csv', index_col='date', parse_dates=['date'])

# Initialize iterator
iterator = FineTuningIterator(base_dir='results/fine_tuning')

# Run a single experiment
result = iterator.run_experiment(
    train_data=train_data,
    val_data=val_data,
    test_data=test_data,
    target_col='price',
    experiment_name='baseline'
)

# View results
print(f"MAE: {result.test_metrics['mae']:.4f}")
print(f"Data insights: {result.data_insights}")
```

## Core Components

### 1. HyperparameterConfig

A dataclass that defines all fine-tuning hyperparameters. Use it to specify tuning settings:

```python
from src.models.fine_tuning_iterator import HyperparameterConfig

config = HyperparameterConfig(
    learning_rate=1e-4,
    epochs=5,
    context_length=512,
    prediction_length=24,
    batch_size=32,
    warmup_steps=0,
    weight_decay=0.01,
    adapter_type='lora',
    lora_rank=8,
    lora_alpha=16.0,
    dropout_rate=0.1,
)
```

**Available parameters:**

- `learning_rate` (float): Initial learning rate for fine-tuning
- `epochs` (int): Number of training epochs
- `context_length` (int): Input sequence length
- `prediction_length` (int): Forecast horizon
- `batch_size` (int): Training batch size
- `warmup_steps` (int): Linear warmup steps
- `weight_decay` (float): L2 regularization coefficient
- `adapter_type` (str): 'lora' or other adapter methods
- `lora_rank` (int): LoRA rank parameter
- `lora_alpha` (float): LoRA scaling parameter
- `dropout_rate` (float): Dropout probability

### 2. ExperimentResult

Dataclass storing complete experiment information:

```python
@dataclass
class ExperimentResult:
    experiment_id: str                      # Unique ID: "{name}_{timestamp}"
    timestamp: str                          # ISO format timestamp
    hyperparams: dict[str, Any]            # Hyperparameter values used
    train_metrics: dict[str, float]        # Training set metrics
    val_metrics: dict[str, float]          # Validation set metrics
    test_metrics: dict[str, float]         # Test set metrics
    data_insights: dict[str, Any]          # Data analysis results
    training_history: list[dict]           # Per-epoch loss/metrics
    notes: str                              # Experiment description
    model_path: str                         # Path to saved model
    error: str | None                      # Error message if failed
```

**Metrics included:**

- `mae`: Mean Absolute Error
- `rmse`: Root Mean Squared Error
- `mase`: Mean Absolute Scaled Error
- `directional_accuracy`: Proportion of correct direction predictions

### 3. FineTuningIterator

The main class for running experiments and grid search:

```python
iterator = FineTuningIterator(
    fine_tuner=None,  # Optional: provide ChronosFineTuner instance
    base_dir='results/fine_tuning'
)
```

## Common Workflows

### Workflow 1: Single Baseline Experiment

Establish a baseline with default hyperparameters:

```python
result = iterator.run_experiment(
    train_data=train_data,
    val_data=val_data,
    test_data=test_data,
    target_col='price',
    experiment_name='baseline',
    notes='Default configuration for reference'
)

print(f"Baseline MAE: {result.test_metrics['mae']:.4f}")
print(f"Volatility: {result.data_insights['train_regimes']['volatility_mean']:.4f}")
```

### Workflow 2: Learning Rate Grid Search

Test multiple learning rates while keeping other parameters fixed:

```python
param_grid = {
    'learning_rate': [1e-5, 1e-4, 5e-4, 1e-3],
}

results = iterator.grid_search(
    train_data=train_data,
    val_data=val_data,
    test_data=test_data,
    target_col='price',
    param_grid=param_grid,
    experiment_name='lr_search'
)

# Find best learning rate
comparison = iterator.compare_experiments()
best = iterator.identify_best_experiment(metric='mae')
print(f"Best learning rate: {best.hyperparams['learning_rate']}")
```

### Workflow 3: Multi-Parameter Grid Search

Test combinations of multiple hyperparameters:

```python
param_grid = {
    'learning_rate': [1e-4, 5e-4],
    'epochs': [3, 5, 10],
    'weight_decay': [0.0, 0.01, 0.1],
}

results = iterator.grid_search(
    train_data=train_data,
    val_data=val_data,
    test_data=test_data,
    target_col='price',
    param_grid=param_grid,
    experiment_name='multi_param_search'
)

# Total experiments: 2 * 3 * 3 = 18 combinations
print(f"Completed {len(results)} experiments")
```

### Workflow 4: Progressive Refinement

Use insights from one stage to inform the next:

```python
# Stage 1: Find optimal learning rate (coarse-grained)
param_grid = {
    'learning_rate': [1e-5, 1e-4, 1e-3],
}
results_stage1 = iterator.grid_search(
    train_data=train_data, val_data=val_data, test_data=test_data,
    target_col='price', param_grid=param_grid,
    experiment_name='stage1_lr'
)

# Analyze Stage 1
best_lr = iterator.identify_best_experiment(metric='mae').hyperparams['learning_rate']

# Stage 2: Fine-grain search around best learning rate
param_grid = {
    'learning_rate': [best_lr * 0.5, best_lr, best_lr * 2],
    'epochs': [3, 5, 10],
}
results_stage2 = iterator.grid_search(
    train_data=train_data, val_data=val_data, test_data=test_data,
    target_col='price', param_grid=param_grid,
    experiment_name='stage2_refinement'
)

best = iterator.identify_best_experiment(metric='mae')
print(f"Best configuration found after refinement")
```

### Workflow 5: Custom Hyperparameter Configs

Use dictionaries for flexible experiment definition:

```python
custom_configs = [
    {
        'learning_rate': 1e-4,
        'epochs': 5,
        'weight_decay': 0.01,
        'dropout_rate': 0.1,
    },
    {
        'learning_rate': 5e-4,
        'epochs': 10,
        'weight_decay': 0.05,
        'dropout_rate': 0.2,
    },
]

for i, config in enumerate(custom_configs):
    result = iterator.run_experiment(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        target_col='price',
        hyperparams=config,
        experiment_name=f'custom_config_{i+1}'
    )
```

## Analysis & Interpretation

### Comparing Experiments

View and compare results across multiple experiments:

```python
# Get comparison DataFrame
comparison_df = iterator.compare_experiments()
print(comparison_df)

# Output columns include:
# - experiment_id: Unique ID
# - timestamp: When run
# - hp_*: Hyperparameter values (hp_learning_rate, hp_epochs, etc.)
# - test_mae, test_rmse, test_mase, test_directional_accuracy: Test metrics
```

### Finding the Best Experiment

Identify best experiment by metric:

```python
best = iterator.identify_best_experiment(metric='mae')
print(f"Best experiment: {best.experiment_id}")
print(f"Test MAE: {best.test_metrics['mae']:.4f}")
print(f"Hyperparameters: {best.hyperparams}")
```

### Experiment Analysis

Use `ExperimentAnalyzer` to identify patterns across experiments:

```python
from src.models.fine_tuning_iterator import ExperimentAnalyzer

analyzer = ExperimentAnalyzer(iterator.tracker.experiments)

# Identify important hyperparameters
important_hps = analyzer.identify_important_hyperparameters(metric='mae', top_n=5)
print(important_hps)
# Output:
#   hyperparameter   correlation  abs_correlation
# hp_learning_rate      -0.725          0.725
#    hp_batch_size      -0.416          0.416
#  hp_dropout_rate       0.002          0.002
```

### Overfitting Detection

Check for overfitting patterns:

```python
overfitting_df = analyzer.analyze_overfitting()
print(overfitting_df)
# Output includes:
# - experiment_id: Experiment identifier
# - train_mae, val_mae: Training vs validation error
# - overfitting_flag: True if val_mae >> train_mae
```

### Data Insights

Each experiment includes comprehensive data analysis:

```python
result = iterator.run_experiment(...)

insights = result.data_insights

# Data quality metrics
print(f"Train rows: {insights['train_quality']['n_rows']}")
print(f"Missing values: {insights['train_quality']['missing_pct']:.1%}")

# Target distribution
print(f"Return mean: {insights['target_distribution']['mean']:.4f}")
print(f"Return std: {insights['target_distribution']['std']:.4f}")
print(f"Return skewness: {insights['target_distribution']['skewness']:.4f}")

# Regime detection
print(f"High volatility periods: {insights['train_regimes']['high_vol_periods']}")
print(f"Mean volatility: {insights['train_regimes']['volatility_mean']:.4f}")

# Feature correlations
print(f"Correlations: {insights['correlations']}")
```

## Interpreting Results

### Key Metrics

- **MAE (Mean Absolute Error)**: Average absolute prediction error. Lower is better. Scale-dependent.
- **RMSE (Root Mean Squared Error)**: Penalizes large errors more. Lower is better.
- **MASE (Mean Absolute Scaled Error)**: Scale-independent error metric. MASE < 1 means better than naive forecast.
- **Directional Accuracy**: Proportion of correct direction predictions (up/down). Higher is better.

### Data Insights

- **Volatility**: Higher volatility suggests more difficult regime. Compare train_regimes and test_regimes to identify regime shift.
- **Correlations**: Feature correlations indicate which variables are predictive. Check sign and magnitude.
- **Skewness/Kurtosis**: Non-normal returns (skewness ≠ 0, kurtosis > 3) suggest market stress or fat tails.
- **Missing data**: High missing percentage may indicate data quality issues or gaps.

### Hyperparameter Sensitivity

From `identify_important_hyperparameters`:

- **High correlation magnitude**: Parameter strongly influences performance
- **Positive correlation**: Higher values → better metric
- **Negative correlation**: Lower values → better metric

Example interpretation:
```
hp_learning_rate: -0.73  → Lower learning rates improve MAE
hp_batch_size:    -0.42  → Smaller batches improve MAE
hp_dropout_rate:  +0.01  → Dropout has minimal impact
```

## File Organization

Experiments are saved with the following structure:

```
results/fine_tuning/
├── baseline_20250101_120000/
│   └── results.json              # Complete experiment data
├── lr_search_001_20250101_120100/
│   └── results.json
├── lr_search_002_20250101_120200/
│   └── results.json
└── ...
```

Each `results.json` contains:

```json
{
  "experiment_id": "baseline_20250101_120000",
  "timestamp": "2025-01-01T12:00:00.000000",
  "hyperparams": {
    "learning_rate": 0.0001,
    "epochs": 5,
    ...
  },
  "train_metrics": {...},
  "val_metrics": {...},
  "test_metrics": {...},
  "data_insights": {...},
  "training_history": [
    {"loss": 0.8, "epoch": 1},
    ...
  ]
}
```

## Advanced Usage

### Loading Previously Saved Experiments

```python
iterator.tracker.load_experiments()
all_experiments = iterator.tracker.experiments
print(f"Loaded {len(all_experiments)} experiments")
```

### Exporting Results to CSV

```python
comparison_df = iterator.compare_experiments()
comparison_df.to_csv('experiment_comparison.csv', index=False)
```

### Analyzing Specific Experiment

```python
# Load by ID
target_id = 'baseline_20250101_120000'
result = next(
    (e for e in iterator.tracker.experiments if e.experiment_id == target_id),
    None
)

if result:
    print(f"Experiment: {result.experiment_id}")
    print(f"Notes: {result.notes}")
    print(f"Error: {result.error}")
    if result.training_history:
        print(f"Final epoch loss: {result.training_history[-1]}")
```

### Custom Comparison Logic

```python
# Find experiments by name pattern
import re
lr_experiments = [
    e for e in iterator.tracker.experiments
    if re.match(r'lr_search_\d+', e.experiment_id)
]

# Group by parameter
from itertools import groupby

comparison = iterator.compare_experiments([e.experiment_id for e in lr_experiments])
for lr, group in groupby(comparison, key=lambda x: x['hp_learning_rate']):
    group_list = list(group)
    avg_mae = sum(e['test_mae'] for e in group_list) / len(group_list)
    print(f"LR {lr}: avg MAE = {avg_mae:.4f}")
```

## Performance Tips

1. **Reduce grid search scope**: Start with 2-3 values per parameter
2. **Use shorter data windows**: For quick iteration, use smaller datasets
3. **Set aside evaluation data**: Use val_data for early stopping
4. **Log experiments**: Check `results/fine_tuning/` directory
5. **Analyze patterns**: Use `ExperimentAnalyzer` instead of manual comparison

## Troubleshooting

### Model Loading Issues

If you see "Could not load model" warnings, the iterator uses a mock model for development:

```
Warning: Could not load model: Can't load tokenizer for 'amazon/chronos-t5-small'
Using mock model for development/testing
```

For production, ensure HuggingFace models can be downloaded and cache is properly configured.

### Missing Metrics

If some metrics are zero or missing:

```python
result = iterator.run_experiment(...)
if not result.test_metrics:
    print(f"Error: {result.error}")
    # Check that test_data has enough samples
    # Check that target_col is in the data
```

### Memory Issues with Large Grids

For large parameter grids:

```python
# Break into stages instead of one large search
stage1 = iterator.grid_search(..., experiment_name='stage1')
# Analyze and narrow search space
stage2 = iterator.grid_search(..., experiment_name='stage2')
```

## See Also

- `demonstrate_fine_tuning_iteration.py`: Complete working examples
- `src/models/chronos_wrapper.py`: ChronosFineTuner API
- `src/eval/metrics.py`: ForecastEvaluator implementation
