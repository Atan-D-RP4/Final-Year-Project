# Basic Forecasting Commands

```
# 1. US Inflation Forecasting (original command)
uv run cli.py forecast --data-type imf_inflation --country US --start-year 2010 --end-year 2023 --output-dir results/us_inflation --create-plots

# 2. European Inflation Comparison
uv run cli.py forecast --data-type imf_inflation --country DE --start-year 2010 --end-year 2023 --output-dir results/de_inflation --create-plots

# 3. UK GDP Growth Forecasting
uv run cli.py forecast --data-type imf_gdp --country GB --start-year 2010 --end-year 2023 --output-dir results/uk_gdp --create-plots

# 4. Japanese Economic Indicators
uv run cli.py forecast --data-type imf_gdp --country JP --start-year 2010 --end-year 2023 --output-dir results/jp_gdp --create-plots

# 5. Exchange Rate Forecasting (USD/EUR)
uv run cli.py forecast --data-type imf_exchange --base-currency USD --target-currency EUR --start-year 2010 --end-year 2023 --output-dir
```

results/usd_eur --create-plots

## Advanced Tokenizer Commands

```
# 6. Advanced Tokenizer with Economic Features
uv run cli.py forecast --data-type imf_inflation --country US --start-year 2010 --end-year 2023 --use-advanced-tokenizer --economic-features
```

--feature-selection --scaling-method robust --output-dir results/us_inflation_advanced --create-plots

```
# 7. Different Window Sizes
uv run cli.py forecast --data-type imf_gdp --country US --start-year 2010 --end-year 2023 --window-size 15 --output-dir results/us_gdp_window15
```

--create-plots

```
# 8. High Quantization Levels
uv run cli.py forecast --data-type imf_inflation --country DE --start-year 2010 --end-year 2023 --use-advanced-tokenizer --quantization-levels 2000
```

--output-dir results/de_inflation_hq --create-plots

## Model Comparison Commands

```
# 9. Comprehensive Model Comparison - US Data
uv run cli.py compare --data-type imf_inflation --country US --start-year 2010 --end-year 2023 --include-chronos --include-baselines --output-dir
```

results/us_comparison --create-plots

```
# 10. European GDP Comparison
uv run cli.py compare --data-type imf_gdp --country DE --start-year 2010 --end-year 2023 --include-chronos --include-baselines --output-dir
```

results/de_gdp_comparison --create-plots

```
# 11. Exchange Rate Model Comparison
uv run cli.py compare --data-type imf_exchange --base-currency USD --target-currency EUR --start-year 2010 --end-year 2023 --include-chronos
```

--include-baselines --output-dir results/fx_comparison --create-plots

## Cross-Validation Commands

```
# 12. Expanding Window Cross-Validation
uv run cli.py cross-validate --data-type imf_inflation --country US --start-year 2010 --end-year 2023 --cv-strategy expanding --n-splits 5
```

--prediction-length 3 --output-dir results/us_cv_expanding --create-plots

```
# 13. Sliding Window Cross-Validation
uv run cli.py cross-validate --data-type imf_gdp --country US --start-year 2010 --end-year 2023 --cv-strategy sliding --n-splits 7 --prediction-length
```

5 --output-dir results/us_cv_sliding --create-plots

```
# 14. Blocked Cross-Validation for GDP
uv run cli.py cross-validate --data-type imf_gdp --country GB --start-year 2010 --end-year 2023 --cv-strategy blocked --n-splits 4 --prediction-length
```

4 --output-dir results/uk_cv_blocked --create-plots

## Synthetic Data Experiments

```
# 15. Economic Synthetic Data
uv run cli.py forecast --data-type synthetic --synthetic-type economic --n-samples 60 --output-dir results/synthetic_economic --create-plots

# 16. Seasonal Synthetic Data
uv run cli.py forecast --data-type synthetic --synthetic-type seasonal --n-samples 48 --output-dir results/synthetic_seasonal --create-plots

# 17. Trend Synthetic Data Comparison
uv run cli.py compare --data-type synthetic --synthetic-type trend --n-samples 50 --include-chronos --include-baselines --output-dir
```

results/synthetic_trend_comparison --create-plots

## Multi-Country Analysis

```
# 18. China Inflation Analysis
uv run cli.py forecast --data-type imf_inflation --country CN --start-year 2010 --end-year 2023 --use-advanced-tokenizer --economic-features
```

--output-dir results/cn_inflation --create-plots

```
# 19. Brazil GDP Growth
uv run cli.py forecast --data-type imf_gdp --country BR --start-year 2010 --end-year 2023 --output-dir results/br_gdp --create-plots

# 20. Canada Economic Comparison
uv run cli.py compare --data-type imf_inflation --country CA --start-year 2010 --end-year 2023 --include-chronos --include-baselines --output-dir
```

results/ca_comparison --create-plots

## Different Time Periods

```
# 21. Pre-2008 Crisis Period
uv run cli.py forecast --data-type imf_inflation --country US --start-year 2000 --end-year 2007 --output-dir results/us_pre_crisis --create-plots

# 22. Post-2008 Recovery Period
uv run cli.py forecast --data-type imf_gdp --country US --start-year 2009 --end-year 2019 --output-dir results/us_post_crisis --create-plots

# 23. Recent Period (2015-2023)
uv run cli.py forecast --data-type imf_inflation --country US --start-year 2015 --end-year 2023 --use-advanced-tokenizer --output-dir
```

results/us_recent --create-plots

## Demo and Comprehensive Analysis

```
# 24. Full Framework Demo
uv run cli.py demo --output-dir results/full_demo

# 25. CSV Data Analysis (if you have custom data)
uv run cli.py forecast --data-type csv --file-path sample_data/engagement_data.csv --column likes --output-dir results/csv_analysis --create-plots
```

## Performance Testing Commands

```
# 26. Large Window Size Test
uv run cli.py forecast --data-type imf_inflation --country US --start-year 2010 --end-year 2023 --window-size 20 --output-dir results/large_window
```

--create-plots

```
# 27. Small Test Split
uv run cli.py forecast --data-type imf_gdp --country US --start-year 2010 --end-year 2023 --test-split 0.1 --output-dir results/small_test
```

--create-plots

```
# 28. Different Scaling Methods
uv run cli.py forecast --data-type imf_inflation --country US --start-year 2010 --end-year 2023 --use-advanced-tokenizer --scaling-method minmax
```

--output-dir results/minmax_scaling --create-plots

## Recommended Execution Order for Report

For a comprehensive report, I recommend running these commands in this order:

```
1. **Start with basic forecasting** (commands 1-5) to establish baseline results
2. **Run model comparisons** (commands 9-11) to show performance differences
3. **Execute cross-validation** (commands 12-14) to demonstrate robustness
4. **Test advanced features** (commands 6-8) to show framework capabilities
5. **Run the demo** (command 24) for a comprehensive overview
6. **Add multi-country analysis** (commands 18-20) for international perspective
```

Each command will generate:

```
* **JSON results** with metrics and predictions
* **Visualization plots** (when `--create-plots` is used)
* **Comprehensive reports** with detailed analysis
```

The outputs will be organized in separate directories under results/ making it easy to compile into your final report.
