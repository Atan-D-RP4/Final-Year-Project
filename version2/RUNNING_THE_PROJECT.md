# Running the Multivariate Financial Forecasting Project

This document provides comprehensive instructions for running the multivariate financial forecasting project with Chronos.

## Project Overview

This project implements **Phase 3 and Phase 4** of a comprehensive financial forecasting system that includes:

- **Data Collection & Cleaning**: Fetches financial and economic data from Yahoo Finance and FRED
- **Tokenization**: Converts financial time series into tokens compatible with Chronos models
- **Baseline Models**: Implements multiple forecasting models (Naive, ARIMA, VAR, LSTM, Linear)
- **Chronos Integration**: Zero-shot forecasting using Chronos foundation models (with mock implementation)
- **Fine-Tuning**: Advanced model fine-tuning capabilities for improved performance
- **Causal Attribution**: Analysis of feature importance and model interpretability
- **Evaluation Framework**: Comprehensive metrics for forecast evaluation including probabilistic metrics
- **Experiment Runner**: Automated experiments for both zero-shot comparison and fine-tuning analysis

## Project Structure

```
version2/
├── src/                          # Main source code
│   ├── data/                     # Data fetching and cleaning
│   │   ├── fetchers.py          # Yahoo Finance & FRED data fetchers
│   │   └── cleaning.py          # Data cleaning and preprocessing
│   ├── preprocessing/            # Data preprocessing
│   │   └── tokenizer.py         # Financial data tokenization
│   ├── models/                   # Forecasting models
│   │   ├── baselines.py         # Baseline models (Naive, ARIMA, VAR, LSTM, Linear)
│   │   └── chronos_wrapper.py   # Chronos model wrapper
│   ├── eval/                     # Evaluation framework
│   │   └── metrics.py           # Forecasting metrics
│   ├── utils/                    # Utilities
│   │   ├── config.py            # Configuration management
│   │   └── logger.py            # Logging utilities
│   └── main.py                   # Main entry point
├── experiments/                  # Experiment scripts
│   ├── phase3_zero_shot.py     # Phase 3 experiment runner
│   └── phase4_fine_tuning.py   # Phase 4 experiment runner
├── data/                        # Data storage
│   ├── raw/                     # Raw data files
│   └── cleaned/                 # Cleaned data files
├── pyproject.toml               # Project configuration
├── main.py                      # Entry point
└── README.md                    # Project documentation
```

## Prerequisites

### System Requirements

- Python 3.10 or higher
- UV package manager (for dependency management)
- Git (for version control)

### Required Tools

The project uses the following tools for development:
- **uv**: Package and dependency management
- **ruff**: Code formatting and linting
- **zuban**: Type checking

## Installation & Setup

### 1. Clone and Navigate to Project

```bash
cd /path/to/Final-Year-Project/version2
```

### 2. Install Dependencies

The project uses `uv` for dependency management. Install all dependencies:

```bash
# Install all dependencies (this may take a while due to large packages like PyTorch)
uv sync
```

**Note**: The initial installation may take 10-15 minutes as it downloads large packages including:
- PyTorch (858MB)
- CUDA libraries (several GB)
- Transformers and other ML libraries

### 3. Optional: FRED API Key

For real economic data, you can set up a FRED API key:

1. Get a free API key from [FRED](https://fred.stlouisfed.org/docs/api/api_key.html)
2. Set it as an environment variable:
   ```bash
   export FRED_API_KEY="your_api_key_here"
   ```

**Note**: The project works without a FRED API key by using sample/synthetic data.

## Running the Project

### Basic Usage

Run Phase 3 experiments with default settings:

```bash
# Using uv (recommended)
uv run python main.py --phase 3

# Or directly with Python (if dependencies are installed)
python main.py --phase 3
```

### Running Phase 4: Fine-Tuning and Causal Attribution

Phase 4 focuses on fine-tuning Chronos models and conducting causal attribution analysis:

```bash
# Using uv (recommended)
uv run python experiments/phase4_fine_tuning.py

# Or directly with Python (if dependencies are installed)
python experiments/phase4_fine_tuning.py
```

### Command Line Options

**For Phase 3:**
```bash
python main.py [OPTIONS]

Options:
  --phase {3,phase3,zero-shot}  Which phase to run (default: 3)
  --config PATH                 Path to configuration file
  --fred-api-key KEY           FRED API key for economic data
  --log-level {DEBUG,INFO,WARNING,ERROR}  Logging level (default: INFO)
  --help                       Show help message
```

**For Phase 4:**
```bash
python experiments/phase4_fine_tuning.py [OPTIONS]

Options:
  --config PATH                 Path to configuration file
  --fred-api-key KEY           FRED API key for economic data
  --log-level {DEBUG,INFO,WARNING,ERROR}  Logging level (default: INFO)
  --help                       Show help message
```

### Examples

1. **Basic run with sample data (Phase 3)**:
   ```bash
   uv run python main.py --phase 3
   ```

2. **Run Phase 3 with FRED API key**:
   ```bash
   uv run python main.py --phase 3 --fred-api-key YOUR_API_KEY
   ```

3. **Run Phase 3 with debug logging**:
   ```bash
   uv run python main.py --phase 3 --log-level DEBUG
   ```

4. **Run Phase 4 fine-tuning experiments**:
   ```bash
   uv run python experiments/phase4_fine_tuning.py
   ```

5. **Run Phase 4 with debug logging**:
   ```bash
   uv run python experiments/phase4_fine_tuning.py --log-level DEBUG
   ```

## What Phase 4 Does

Phase 4 focuses on fine-tuning Chronos models and conducting causal attribution analysis:

### 1. Data Preparation
- Fetches financial data (S&P 500, VIX, Treasury yields) from Yahoo Finance
- Fetches economic data (unemployment, inflation, Fed funds rate) from FRED
- Cleans and aligns data across different frequencies
- Creates additional technical indicators and features

### 2. Fine-Tuning Experiments
- Fine-tunes Chronos foundation models on financial datasets
- Adapts pre-trained models to specific market conditions and data patterns
- Evaluates fine-tuned models using comprehensive metrics (MAE, RMSE, MASE, etc.)
- Compares performance against zero-shot baselines

### 3. Causal Attribution Analysis
- Analyzes the contribution of different covariates to model predictions
- Implements multiple attribution methods:
  - **Ablation Importance**: Measures impact by removing features systematically
  - **Permutation Importance**: Assesses feature importance through shuffling
  - **SHAP Analysis**: Uses SHAP values for detailed feature impact analysis
- Identifies key economic and market indicators driving forecasts

### 4. Results Analysis
- Compares fine-tuned models against baseline and zero-shot approaches
- Generates attribution reports showing feature importance rankings
- Provides insights into model interpretability and decision-making processes

## What Phase 3 Does

Phase 3 runs comprehensive experiments comparing different forecasting approaches:

### 1. Data Preparation
- Fetches financial data (S&P 500, VIX, Treasury yields) from Yahoo Finance
- Fetches economic data (unemployment, inflation, Fed funds rate) from FRED
- Cleans and aligns data across different frequencies
- Creates additional technical indicators and features

### 2. Baseline Experiments
Tests multiple baseline forecasting models:
- **Naive Forecasters**: Last value, mean, seasonal patterns
- **ARIMA**: Autoregressive Integrated Moving Average
- **VAR**: Vector Autoregression (multivariate)
- **LSTM**: Long Short-Term Memory neural networks
- **Linear**: Linear regression with lagged features

### 3. Tokenization Experiments
Tests different strategies for converting financial data to tokens:
- **Uniform binning**: Equal-width bins
- **Quantile binning**: Equal-frequency bins
- **K-means clustering**: Learned cluster centers
- **Advanced tokenization**: With technical indicators and time features

### 4. Chronos Zero-Shot Experiments
- Tests Chronos foundation model for zero-shot forecasting
- Uses mock implementation if Chronos is not available
- Compares against baseline models

### 5. Results Analysis
- Compares all models using comprehensive metrics
- Generates performance reports and visualizations
- Identifies best-performing models for each metric


Best models by metric:
  mae                 : baseline_linear        (0.0234)
  rmse                : baseline_lstm          (0.0456)
  directional_accuracy: chronos_chronos_small  (67.8900)

Detailed results saved to: /path/to/results/phase3
==================================================

Best models by metric:
  mae                 : baseline_linear        (0.0234)
  rmse                : baseline_lstm          (0.0456)
  directional_accuracy: chronos_chronos_small  (67.8900)

Detailed results saved to: /path/to/results/phase3
==================================================
```

## Output Files

Results are saved to `results/phase3/` directory:
- `phase3_summary_report.txt`: Text summary of all results
- `metric_comparison.png`: Visualization comparing model performance
- Individual model results and configurations

## Development Tools

### Code Formatting

```bash
# Format code with ruff
uv tool run ruff format src/

# Check code style
uv tool run ruff check src/
```

### Type Checking

```bash
# Run type checking with zuban
uv tool run zuban check src/
```

### Testing

```bash
# Run basic functionality tests (without heavy dependencies)
python test_minimal.py

# Run comprehensive tests (requires dependencies)
python test_basic.py
```

## Troubleshooting

### Common Issues

1. **Dependencies taking too long to install**:
   - This is normal due to large ML packages
   - Consider using a faster internet connection
   - The installation is cached for future runs

2. **CUDA/GPU errors**:
   - The project works on CPU-only systems
   - GPU acceleration is optional and automatically detected

3. **Memory issues with LSTM**:
   - Reduce `batch_size` or `epochs` in the configuration
   - Use smaller datasets for testing

4. **FRED API errors**:
   - The project falls back to sample data if FRED is unavailable
   - Check your API key if using real economic data

### Performance Considerations

- **First run**: May take 15-30 minutes due to dependency installation and model training
- **Subsequent runs**: Much faster (5-10 minutes) as dependencies are cached
- **Memory usage**: Requires ~4-8GB RAM for full experiments
- **CPU vs GPU**: LSTM models benefit from GPU but work on CPU

## Configuration

### Default Settings

The project uses sensible defaults:
- **Prediction horizon**: 24 steps
- **Context length**: 512 tokens
- **Test split**: 20% of data
- **Metrics**: MAE, RMSE, MASE, sMAPE, Directional Accuracy

### Customization

You can customize the configuration by:
1. Modifying `src/utils/config.py`
2. Creating a custom config file and using `--config path/to/config.json`
3. Editing the experiment parameters in `experiments/phase3_zero_shot.py`

## Research Context

This implementation represents **Phase 3 and Phase 4** of a larger research project on:
- **Multivariate Financial Forecasting** using foundation models
- **Zero-shot transfer learning** for financial time series
- **Tokenization strategies** for financial data
- **Fine-tuning adaptation** for domain-specific improvements
- **Attribution analysis** for model interpretability

The complete research plan includes 8 phases, with this implementation covering the core forecasting, fine-tuning, and attribution functionalities.

## Next Steps

After running Phase 3 and Phase 4, you can:
1. Analyze the detailed results in `results/phase3/` and `results/phase4/`
2. Compare zero-shot vs fine-tuned model performance
3. Experiment with different model configurations and attribution methods
4. Implement additional phases (backtesting, robustness testing)
5. Extend to new datasets or forecasting targets
6. Deploy the best-performing models in production

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the code documentation in each module
3. Examine the experiment logs for detailed error messages
4. Refer to the research documentation in `RESEARCH_OVERVIEW.md` and `PLAN.md`

---

**Note**: This project is designed for research and educational purposes. The forecasting results should not be used for actual financial decision-making without proper validation and risk management.
