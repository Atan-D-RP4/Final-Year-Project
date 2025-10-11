# Enhanced Chronos Forecasting Framework

A comprehensive time series forecasting framework built on Amazon's Chronos-Bolt models, enhanced with real macroeconomic data integration, advanced preprocessing, cross-validation, and extensive visualization capabilities.

## 🚀 Features

### Phase 1: Data Integration
- **IMF Data Integration**: Direct access to real macroeconomic data from the International Monetary Fund
- **Multiple Data Sources**: Support for inflation, GDP growth, exchange rates, and other economic indicators
- **Data Validation**: Comprehensive data quality checks and outlier detection
- **Caching System**: Efficient caching to minimize API calls

### Phase 2: Advanced Preprocessing
- **Multivariate Support**: Handle multiple time series simultaneously
- **Feature Engineering**: Automatic creation of lag features, moving averages, volatility measures, and rate of change indicators
- **Economic Features**: Specialized features for economic data including seasonal decomposition and trend analysis
- **Feature Selection**: Intelligent feature selection based on importance scores
- **Multiple Scaling Options**: StandardScaler, RobustScaler, MinMaxScaler support

### Phase 3: Enhanced Forecasting
- **Advanced Tokenization**: Sophisticated tokenization strategies for better model performance
- **Ensemble Methods**: Support for model ensembling and weighted predictions
- **Confidence Intervals**: Uncertainty quantification in forecasts
- **Zero-shot Learning**: Leverage pre-trained Chronos models without fine-tuning

### Phase 4: Comprehensive Evaluation
- **Time Series Cross-Validation**: Expanding, sliding, and blocked window strategies
- **Extensive Metrics**: MSE, MAE, MAPE, MASE, directional accuracy, and more
- **Baseline Comparisons**: Built-in comparison with ARIMA, exponential smoothing, and naive forecasts
- **Backtesting Framework**: Robust backtesting with multiple validation strategies

### Phase 5: Rich Visualization
- **Interactive Plots**: Plotly-based interactive visualizations
- **Comprehensive Reports**: Automated HTML report generation
- **Residual Analysis**: Detailed residual diagnostics
- **Model Comparison**: Visual comparison of multiple models
- **Feature Importance**: Visualization of feature importance scores

### Phase 6: User-Friendly Interface
- **Command Line Interface**: Comprehensive CLI for all framework features
- **Configuration Files**: YAML/JSON configuration support
- **Batch Processing**: Support for processing multiple datasets
- **Export Capabilities**: Multiple output formats (CSV, JSON, HTML, PNG)

## 📦 Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, for faster inference)

### Install Dependencies

```bash
# Clone the repository
git clone <repository-url>
cd chronos

# Install using uv (recommended)
uv sync

# Or install using pip
pip install -r requirements.txt
```

### Required Packages
- chronos-forecasting>=1.5.2
- matplotlib>=3.10.3
- numpy>=2.3.1
- pandas>=2.3.1
- scikit-learn>=1.7.1
- seaborn>=0.13.2
- requests>=2.31.0
- scipy>=1.11.0
- plotly>=5.17.0
- statsmodels>=0.14.0

## 🚀 Quick Start

### 1. Basic Forecasting

```python
from enhanced_data_preparation import EnhancedBehavioralDataLoader
from comprehensive_demo import ComprehensiveForecaster

# Load IMF inflation data
loader = EnhancedBehavioralDataLoader()
data = loader.load_imf_inflation_data(['US'], 2010, 2023, 'US')

# Create forecaster with advanced features
forecaster = ComprehensiveForecaster(
    model_name="amazon/chronos-bolt-small",
    use_advanced_tokenizer=True,
    tokenizer_config={
        'window_size': 10,
        'scaling_method': 'robust',
        'feature_selection': True,
        'economic_features': True
    }
)

# Split data and forecast
train_data = data[:int(0.8 * len(data))]
test_length = len(data) - len(train_data)

tokenized_data, tokenizer = forecaster.prepare_data(train_data)
forecast_result = forecaster.forecast_zero_shot(tokenized_data, test_length)
predictions = forecast_result['mean'][0].numpy()
```

### 2. Using the Command Line Interface

```bash
# Forecast US inflation data
python cli.py forecast --data-type imf_inflation --country US --start-year 2010 --end-year 2023 --use-advanced-tokenizer --create-plots --output-dir results

# Compare multiple models
python cli.py compare --data-type imf_gdp --country US --include-chronos --include-baselines --create-plots --output-dir comparison

# Run cross-validation
python cli.py cross-validate --data-type csv --file-path data.csv --column value --cv-strategy expanding --n-splits 5 --create-plots --output-dir cv_results

# Run comprehensive demo
python cli.py demo --output-dir demo_results
```

### 3. Comprehensive Demo

```python
from comprehensive_demo import main as run_demo

# Run the full demo showcasing all features
results = run_demo()
```

## 📊 Data Sources

### IMF Data Integration

The framework provides seamless integration with IMF databases:

```python
from enhanced_data_preparation import EnhancedBehavioralDataLoader

loader = EnhancedBehavioralDataLoader()

# Load inflation data for multiple countries
inflation_data = loader.load_imf_inflation_data(['US', 'GB', 'DE'], 2000, 2023)

# Load GDP growth data
gdp_data = loader.load_imf_gdp_growth_data(['US'], 2000, 2023, 'US')

# Load exchange rate data
fx_data = loader.load_imf_exchange_rate_data('USD', 'EUR', 2000, 2023)

# Load multivariate economic data
mv_data = loader.load_imf_multivariate_data('US', ['gdp_growth', 'inflation'], 2000, 2023)
```

### Supported Data Types

1. **IMF World Economic Outlook (WEO)**
   - GDP growth rates
   - Inflation rates (CPI)
   - Unemployment rates
   - Government debt and balance

2. **IMF International Financial Statistics (IFS)**
   - Exchange rates
   - Interest rates
   - Monetary aggregates

3. **Custom Data Sources**
   - CSV files
   - JSON data
   - Synthetic data generation

## 🔧 Advanced Configuration

### Tokenizer Configuration

```python
from advanced_tokenizer import AdvancedTokenizer

tokenizer = AdvancedTokenizer(
    window_size=12,                    # Sliding window size
    quantization_levels=1000,          # Number of quantization levels
    scaling_method='robust',           # 'standard', 'robust', 'minmax'
    feature_selection=True,            # Enable feature selection
    max_features=20,                   # Maximum features to select
    economic_features=True             # Enable economic-specific features
)
```

### Cross-Validation Configuration

```python
from cross_validation import BacktestingFramework

backtest = BacktestingFramework(
    forecaster=forecaster,
    cv_strategy='expanding',           # 'expanding', 'sliding', 'blocked'
    n_splits=5,                       # Number of CV splits
    prediction_length=6,              # Forecast horizon
    window_size=10                    # Input window size
)

results = backtest.run_backtest(data, return_predictions=True)
```

### Visualization Configuration

```python
from visualization import ForecastVisualizer, create_forecast_report

visualizer = ForecastVisualizer(figsize=(15, 10))

# Create forecast plot
fig = visualizer.plot_forecast_results(
    actual=test_data,
    predicted=predictions,
    train_data=train_data,
    confidence_intervals={'lower': lower_bounds, 'upper': upper_bounds}
)

# Generate comprehensive report
report_path = create_forecast_report(
    actual=test_data,
    predicted=predictions,
    train_data=train_data,
    model_name="Enhanced Chronos",
    save_dir="forecast_report"
)
```

## 📈 Model Comparison

### Baseline Models

The framework includes comprehensive baseline model comparisons:

```python
from baseline_models import BaselineComparison

comparison = BaselineComparison()
comparison.add_standard_models()  # Adds Naive, ARIMA, Exponential Smoothing, etc.

results = comparison.compare_models(train_data, test_data)
comparison.print_comparison()
comparison.plot_comparison(test_data)
```

### Available Baseline Models

1. **Naive Forecasting**: Repeat last observed value
2. **Seasonal Naive**: Repeat seasonal pattern
3. **Drift Method**: Linear trend extrapolation
4. **Moving Average**: Simple and weighted moving averages
5. **Linear Trend**: Linear regression forecasting
6. **Exponential Smoothing**: Simple, double, and triple exponential smoothing
7. **ARIMA**: Auto-ARIMA with order selection

## 🎯 Evaluation Metrics

### Comprehensive Metrics Suite

- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Error (MAE)**
- **Mean Absolute Percentage Error (MAPE)**
- **Symmetric MAPE (sMAPE)**
- **Mean Absolute Scaled Error (MASE)**
- **Directional Accuracy**
- **Forecast Bias**
- **Theil's U Statistic**
- **Prediction Interval Coverage**

### Cross-Validation Strategies

1. **Expanding Window**: Training set grows with each split
2. **Sliding Window**: Fixed-size training window
3. **Blocked Cross-Validation**: Non-overlapping blocks

## 📁 Project Structure

```
chronos/
├── README.md                          # This file
├── pyproject.toml                     # Project dependencies
├── setup.py                          # Quick setup script
├── main.py                           # Basic example
├── cli.py                            # Command line interface
├── comprehensive_demo.py             # Full framework demo
├── 
├── # Core Framework Files
├── chronos_behavioral_framework.py   # Original framework
├── enhanced_data_preparation.py      # Enhanced data loading
├── imf_data_loader.py                # IMF API integration
├── advanced_tokenizer.py             # Advanced preprocessing
├── cross_validation.py               # CV framework
├── baseline_models.py                # Baseline comparisons
├── visualization.py                  # Visualization tools
├── 
├── # Sample Data
├── sample_data/
│   ├── sentiment_scores.csv
│   ├── engagement_data.csv
│   └── clickstream_data.csv
├── 
└── # Generated Outputs (created during runs)
    ├── imf_cache/                    # Cached IMF data
    ├── forecast_report/              # Generated reports
    └── test_report/                  # Test outputs
```

## 🔍 Examples

### Example 1: Economic Indicator Forecasting

```python
from enhanced_data_preparation import EnhancedBehavioralDataLoader
from comprehensive_demo import ComprehensiveForecaster
from visualization import create_forecast_report

# Load US economic data
loader = EnhancedBehavioralDataLoader()
mv_data = loader.load_imf_multivariate_data(
    'US', 
    ['gdp_growth', 'inflation'], 
    2000, 2023
)

# Create advanced forecaster
forecaster = ComprehensiveForecaster(
    model_name="amazon/chronos-bolt-small",
    use_advanced_tokenizer=True,
    tokenizer_config={
        'window_size': 12,
        'scaling_method': 'robust',
        'feature_selection': True,
        'max_features': 20,
        'economic_features': True
    }
)

# Forecast inflation using GDP growth as additional feature
inflation_data = mv_data['inflation']
train_size = int(0.8 * len(inflation_data))

tokenized_data, tokenizer = forecaster.prepare_data(mv_data, 12)
forecast_result = forecaster.forecast_zero_shot(
    tokenized_data, 
    len(inflation_data) - train_size
)

# Generate comprehensive report
create_forecast_report(
    actual=inflation_data[train_size:],
    predicted=forecast_result['mean'][0].numpy(),
    train_data=inflation_data[:train_size],
    model_name="Enhanced Chronos - US Inflation",
    save_dir="inflation_forecast_report"
)
```

### Example 2: Cross-Validation Study

```python
from cross_validation import BacktestingFramework
from comprehensive_demo import ComprehensiveForecaster

# Load data
data = loader.load_imf_gdp_growth_data(['US'], 2000, 2023, 'US')

# Create forecaster
forecaster = ComprehensiveForecaster(use_advanced_tokenizer=True)

# Run comprehensive cross-validation
backtest = BacktestingFramework(
    forecaster=forecaster,
    cv_strategy='expanding',
    n_splits=8,
    prediction_length=4,
    window_size=12
)

cv_results = backtest.run_backtest(data, return_predictions=True)
backtest.print_results()
backtest.plot_results()

# Compare different CV strategies
strategy_comparison = backtest.compare_strategies(
    data, 
    ['expanding', 'sliding', 'blocked']
)
```

### Example 3: Model Ensemble

```python
from baseline_models import BaselineComparison
from comprehensive_demo import ComprehensiveForecaster

# Create multiple forecasters
forecasters = [
    ComprehensiveForecaster(
        model_name="amazon/chronos-bolt-small",
        use_advanced_tokenizer=False
    ),
    ComprehensiveForecaster(
        model_name="amazon/chronos-bolt-small",
        use_advanced_tokenizer=True,
        tokenizer_config={'economic_features': True}
    )
]

# Get predictions from each
predictions = []
for forecaster in forecasters:
    tokenized_data, _ = forecaster.prepare_data(train_data)
    result = forecaster.forecast_zero_shot(tokenized_data, len(test_data))
    predictions.append(result['mean'][0].numpy())

# Simple ensemble (average)
ensemble_prediction = np.mean(predictions, axis=0)

# Compare with baselines
comparison = BaselineComparison()
comparison.add_standard_models()
baseline_results = comparison.compare_models(train_data, test_data)

# Add ensemble to comparison
from sklearn.metrics import mean_squared_error, mean_absolute_error
ensemble_mse = mean_squared_error(test_data, ensemble_prediction)
ensemble_mae = mean_absolute_error(test_data, ensemble_prediction)

print(f"Ensemble MSE: {ensemble_mse:.6f}")
print(f"Ensemble MAE: {ensemble_mae:.6f}")
```

## 🛠️ Troubleshooting

### Common Issues

1. **IMF API Connection Issues**
   ```python
   # Check internet connection and try with synthetic data
   loader = EnhancedBehavioralDataLoader()
   data = loader.generate_synthetic_inflation(24)  # 24 months of data
   ```

2. **Memory Issues with Large Models**
   ```python
   # Use smaller model or CPU
   forecaster = ComprehensiveForecaster(
       model_name="amazon/chronos-bolt-tiny",  # Smaller model
       device="cpu"  # Use CPU instead of GPU
   )
   ```

3. **Feature Selection Errors**
   ```python
   # Disable feature selection for small datasets
   tokenizer_config = {
       'feature_selection': False,
       'max_features': 10
   }
   ```

### Performance Optimization

1. **Use GPU for Faster Inference**
   ```python
   forecaster = ComprehensiveForecaster(device="cuda")
   ```

2. **Cache IMF Data**
   ```python
   # Data is automatically cached in 'imf_cache/' directory
   # Clear cache if needed: rm -rf imf_cache/
   ```

3. **Reduce Feature Complexity**
   ```python
   tokenizer_config = {
       'window_size': 8,        # Smaller window
       'max_features': 10,      # Fewer features
       'economic_features': False  # Disable complex features
   }
   ```

## 📚 API Reference

### Core Classes

- **`EnhancedBehavioralDataLoader`**: Enhanced data loading with IMF integration
- **`AdvancedTokenizer`**: Advanced preprocessing and feature engineering
- **`ComprehensiveForecaster`**: Main forecasting interface
- **`BacktestingFramework`**: Cross-validation and backtesting
- **`BaselineComparison`**: Baseline model comparisons
- **`ForecastVisualizer`**: Visualization and reporting

### Key Methods

```python
# Data Loading
loader.load_imf_inflation_data(countries, start_year, end_year)
loader.load_imf_gdp_growth_data(countries, start_year, end_year)
loader.load_imf_exchange_rate_data(base_currency, target_currency, start_year, end_year)

# Tokenization
tokenizer.fit_transform(data, target)
tokenizer.get_feature_importance()
tokenizer.get_feature_names()

# Forecasting
forecaster.prepare_data(data, window_size)
forecaster.forecast_zero_shot(context_data, prediction_length)

# Cross-Validation
backtest.run_backtest(data, show_progress, return_predictions)
backtest.compare_strategies(data, strategies)

# Visualization
visualizer.plot_forecast_results(actual, predicted, train_data)
visualizer.plot_residual_analysis(actual, predicted)
create_forecast_report(actual, predicted, train_data, model_name, save_dir)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone and setup development environment
git clone <repository-url>
cd chronos
uv sync --dev

# Run tests
python -m pytest tests/

# Run comprehensive demo
python comprehensive_demo.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Amazon Research**: For the original Chronos models
- **International Monetary Fund**: For providing open access to economic data
- **Hugging Face**: For the transformers library and model hosting
- **The Open Source Community**: For the excellent libraries this framework builds upon

## 📞 Support

- **Issues**: Please report bugs and feature requests via GitHub Issues
- **Discussions**: Join the discussion in GitHub Discussions
- **Documentation**: Full documentation available in the `docs/` directory

## 🔮 Roadmap

### Upcoming Features

- **Real-time Data Streaming**: Live data feeds from financial APIs
- **Model Fine-tuning**: Custom fine-tuning on domain-specific data
- **Web Dashboard**: Interactive web interface for non-technical users
- **API Server**: REST API for integration with other systems
- **More Data Sources**: Integration with FRED, World Bank, and other APIs
- **Advanced Ensembles**: Sophisticated ensemble methods and meta-learning
- **Automated Model Selection**: AutoML-style model and hyperparameter selection

---

**Happy Forecasting! 📈**