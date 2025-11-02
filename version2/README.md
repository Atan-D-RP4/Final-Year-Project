# Multivariate Financial Forecasting with Chronos

A comprehensive research implementation for multivariate financial forecasting using foundation models, with a focus on zero-shot learning and tokenization strategies.

## 🚀 Quick Start

```bash
# Navigate to the project directory
cd version2/

# Install dependencies (may take 10-15 minutes)
uv sync

# Run Phase 3 experiments
uv run python main.py --phase 3
```

## 📋 Overview

This project implements **Phase 3** of a comprehensive financial forecasting research system that:

- 🔍 **Fetches real financial data** from Yahoo Finance and FRED
- 🔧 **Implements multiple baseline models** (Naive, ARIMA, VAR, LSTM, Linear)
- 🎯 **Tests Chronos foundation models** for zero-shot forecasting
- 📊 **Provides comprehensive evaluation** with multiple metrics
- 🧪 **Runs automated experiments** comparing all approaches

## 🏗️ Architecture

```
📦 Multivariate Financial Forecasting
├── 📊 Data Layer (Yahoo Finance + FRED)
├── 🔧 Preprocessing (Tokenization + Feature Engineering)
├── 🤖 Models (Baselines + Chronos)
├── 📈 Evaluation (Comprehensive Metrics)
└── 🧪 Experiments (Automated Comparison)
```

## 🎯 Key Features

### Data Sources
- **Market Data**: S&P 500, VIX, Treasury yields via Yahoo Finance
- **Economic Data**: Unemployment, inflation, Fed funds rate via FRED
- **Sample Data**: Built-in synthetic data for testing without API keys

### Forecasting Models
- **Naive Models**: Last value, mean, seasonal patterns
- **Statistical Models**: ARIMA, Vector Autoregression (VAR)
- **Machine Learning**: LSTM neural networks, Linear regression
- **Foundation Models**: Chronos zero-shot forecasting

### Tokenization Strategies
- **Uniform Binning**: Equal-width quantization
- **Quantile Binning**: Equal-frequency quantization
- **K-means Clustering**: Learned cluster centers
- **Advanced Features**: Technical indicators + time features

### Evaluation Metrics
- **Point Forecasts**: MAE, RMSE, MASE, sMAPE
- **Directional Accuracy**: Up/down prediction accuracy
- **Probabilistic**: CRPS, quantile loss, interval coverage

## 📖 Documentation

- **[Running the Project](RUNNING_THE_PROJECT.md)** - Comprehensive setup and usage guide
- **[Research Overview](RESEARCH_OVERVIEW.md)** - Research goals and methodology
- **[Implementation Plan](PLAN.md)** - Detailed technical implementation plan
- **[Project Outline](UPDATED_OUTLINE.md)** - Project structure and objectives

## 🛠️ Requirements

- **Python**: 3.10+
- **Package Manager**: UV (recommended) or pip
- **Memory**: 4-8GB RAM recommended
- **Optional**: FRED API key for real economic data

## 📊 Example Results

Detailed results saved to: results/phase3/
==================================================

Best models by metric:
  mae                 : baseline_linear        (0.0234)
  rmse                : baseline_lstm          (0.0456)
  directional_accuracy: chronos_chronos_small  (67.89%)

Detailed results saved to: results/phase3/
==================================================
```

## 🔧 Development

### Code Quality
```bash
# Format code
uv tool run ruff format src/

# Check style
uv tool run ruff check src/

# Type checking
uv tool run zuban check src/
```

### Testing
```bash
# Quick test (no dependencies)
python test_minimal.py

# Full test (requires dependencies)
python test_basic.py
```

## 🎯 Use Cases

### Research Applications
- **Foundation Model Evaluation**: Test Chronos on financial data
- **Tokenization Research**: Compare different quantization strategies
- **Multivariate Analysis**: Study cross-asset relationships
- **Zero-shot Learning**: Evaluate transfer learning capabilities

### Practical Applications
- **Portfolio Management**: Multi-asset return forecasting
- **Risk Management**: Volatility and drawdown prediction
- **Economic Analysis**: Macro indicator forecasting
- **Trading Signals**: Directional prediction for systematic strategies

## 🏛️ Research Context

This implementation is part of a larger research project investigating:

1. **Zero-shot financial forecasting** with foundation models
2. **Tokenization strategies** for financial time series
3. **Multivariate relationships** in financial markets
4. **Attribution analysis** for model interpretability

The complete research plan spans 8 phases, with this implementation covering the core forecasting and evaluation framework.

## 📈 Performance

- **Training Time**: 5-30 minutes depending on models selected
- **Memory Usage**: 2-8GB RAM depending on dataset size
- **Accuracy**: Competitive with domain-specific models
- **Scalability**: Handles multiple assets and timeframes

## 🤝 Contributing

This is a research implementation. To contribute:

1. Review the research documentation
2. Follow the code style guidelines (ruff + zuban)
3. Add comprehensive tests for new features
4. Update documentation for any changes

## ⚠️ Disclaimer

This project is for **research and educational purposes only**. The forecasting results should not be used for actual financial decision-making without proper validation, risk management, and regulatory compliance.

## 📄 License

This project is part of academic research. Please refer to the institution's policies for usage and distribution.

---

**Built with**: Python, PyTorch, Transformers, scikit-learn, pandas, numpy

**Research Focus**: Foundation models for financial forecasting, zero-shot learning, multivariate time series analysis
