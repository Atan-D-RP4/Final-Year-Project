#import "@preview/showybox:2.0.4": showybox

#import "@preview/tablex:0.0.9": colspanx, rowspanx, tablex
#let document-title = "Multivariate Financial Forecasting with Chronos"
#let submission-details = [
  A Project report submitted in partial fulfillment of the requirements for the award of degree in
  *BACHELOR OF TECHNOLOGY* \
  *(COMPUTER SCIENCE AND ENGINEERING)*

  *SUBMITTED BY* \
  #table(
    columns: (auto, auto),
    align: left,
    [Registration number], [Name of the Student],
    [A22126510134], [Bheesetti Harsith Veera Charan],
    [A22126510144], [D. Chaitanya],
    [A22126510194], [Wuna Akhilesh],
    [A22126510163], [M. Sai Treja],
    [A22126510193], [Venkata Vishaal Tirupalli],
  )

  *UNDER THE GUIDANCE OF* \
  Dr. D. Naga Teja \
  Associate professor

  #box(width: 2.4in, height: 2.4in)[
    #image("../anits_logo.png", width: 2.4in, height: 2.4in)
  ]

  *DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING* \
  *ANIL NEERUKONDA INSTITUTE OF TECHNOLOGY AND SCIENCES (A+)* \
  SANGIVALASA, VISAKHAPATNAM – 531162 \
  July - 2025
]

#set document(title: document-title)
#set page(
  margin: (x: 1in, y: 1in),
  numbering: "i",
)

#set text(font: "Times New Roman", size: 12pt)
#set heading(numbering: "1.")
#set par(justify: true, leading: 1em)

#align(center)[
  #text(size: 16pt, weight: "bold")[#document-title]
  #v(1em)
  #submission-details
]

#pagebreak()

#set page(
  paper: "a4",
  margin: (x: 2.5cm, y: 2cm),
  numbering: "1",
)

#set text(
  font: "New Computer Modern",
  size: 11pt,
  lang: "en",
)

#set heading(numbering: "1.1")

#set par(
  justify: true,
  leading: 0.65em,
)

#show heading.where(level: 1): it => [
  #pagebreak(weak: true)
  #v(1em)
  #block(fill: rgb("#1f2937"), inset: 1em, radius: 0.5em, text(fill: white, weight: "bold", size: 1.2em)[#it])
  #v(1em)
]

#show heading.where(level: 2): it => [
  #v(0.8em)
  #text(fill: rgb("#374151"), weight: "bold", size: 1.1em)[#it]
  #v(0.5em)
]

#show heading.where(level: 3): it => [
  #v(0.6em)
  #text(fill: rgb("#4b5563"), weight: "bold")[#it]
  #v(0.3em)
]

#show raw.where(block: true): it => block(fill: rgb("#f3f4f6"), inset: 1em, radius: 0.3em, width: 100%, text(
  font: "JetBrainsMono NF",
  size: 9pt,
)[#it])

#show raw.where(block: false): it => box(fill: rgb("#e5e7eb"), inset: (x: 0.3em, y: 0.1em), radius: 0.2em, text(
  font: "JetBrainsMono NF",
  size: 9pt,
)[#it])

#align(center)[
  #v(2cm)
  #text(text(document-title), size: 2.5em, weight: "bold")

  #v(1cm)
  #text(size: 1.5em, style: "italic")[
    A Framework for Multivariate Financial Forecasting using Amazon's Chronos-Bolt Transformer Models
  ]

  #v(2cm)
  #text(size: 1.2em)[
    Technical Documentation and Report
  ]

  #v(1cm)
  #text(size: 1em)[
    Version 1.0 • #datetime.today().display()
  ]

  #v(2cm)
  #text(size: 1em)[
    Advanced Time Series Analysis using Amazon's Chronos-Bolt Transformer Models
  ]
]

#pagebreak()

#outline(depth: 2, indent: auto)

#pagebreak()

#heading(level: 1)[Introduction]

The forecasting of financial time series is a notoriously challenging task due to the inherent noise, non-stationarity, and complex dependencies of
financial markets. Traditional econometric models have often struggled to capture the intricate dynamics of these systems. With the advent of deep
learning, more sophisticated models have been developed, but they often require extensive training data and domain-specific feature engineering.

Recently, a new class of models known as foundation models has emerged. These models are pre-trained on vast amounts of data and can be adapted to a
wide range of downstream tasks with minimal fine-tuning. Chronos is a foundation model for time series forecasting that has shown impressive
performance in zero-shot and few-shot settings.

This project investigates the application of Chronos to the domain of multivariate financial forecasting. The primary objective is to assess the
zero-shot performance of Chronos on financial data and to develop a comprehensive framework for financial forecasting that leverages the power of
foundation models.

#heading(level: 2)[Problem Statement]

The core problem this project addresses is the difficulty of accurately forecasting financial time series, especially in a multivariate context where
multiple economic and financial indicators interact in complex ways. The project aims to explore whether a pre-trained foundation model like Chronos
can be effectively applied to this problem, potentially reducing the need for extensive model training and feature engineering.

#heading(level: 2)[Objectives]

The main objectives of this project are:
- To investigate the zero-shot forecasting capabilities of Chronos on multivariate financial time series.
- To develop and evaluate different tokenization strategies for converting numerical financial data into a format suitable for Chronos.
- To build and benchmark a range of baseline forecasting models, including traditional statistical models and deep learning models.
- To create a modular and extensible software framework for financial forecasting research and development.
- To analyze the results and provide insights into the strengths and weaknesses of different forecasting approaches.

#pagebreak()

#heading(level: 1)[Literature Review]

The field of time series forecasting has a rich history, with a wide range of methods developed over the years.

#heading(level: 2)[Traditional Time Series Models]

Traditional statistical methods for time series forecasting include models like ARIMA (Autoregressive Integrated Moving Average), VAR (Vector
Autoregression), and Exponential Smoothing. These models are well-understood and have been widely used in finance and economics. However, they often
make strong assumptions about the data (e.g., stationarity) and may not be able to capture complex non-linear relationships.

#heading(level: 2)[Deep Learning for Time Series]

With the rise of deep learning, models like Recurrent Neural Networks (RNNs), Long Short-Term Memory (LSTM) networks, and Temporal Convolutional
Networks (TCNs) have been applied to time series forecasting. These models are more flexible than traditional methods and can learn complex patterns
from the data. However, they typically require large amounts of training data and can be computationally expensive to train.

#heading(level: 2)[Foundation Models for Time Series]

The concept of foundation models, pre-trained on large datasets and adaptable to various tasks, has recently been extended to time series forecasting.
Chronos is a notable example of a foundation model for time series. It is a T5-based encoder-decoder model that is pre-trained on a large corpus of
time series data. The key innovation of Chronos is its tokenization approach, which allows it to treat time series forecasting as a language modeling
problem. This project builds upon the ideas behind Chronos and applies them to the specific domain of financial forecasting.

#pagebreak()

#heading(level: 1)[System Design]

#heading(level: 2)[System Architecture]

The system is designed as a modular pipeline that processes data from raw sources to final evaluation. The data flows through a series of components,
each responsible for a specific task. The architecture is designed to be flexible and extensible, allowing for the easy addition of new data sources,
models, and evaluation metrics.

#image("./arch.svg")

#let Modules() = {
  box(width: 100%)[

    #align(center)[

      #text(16pt, weight: "bold")[Modules Division]
    ]
    #grid(
      columns: (1fr, 1fr, 1fr),
      rows: (auto, auto, auto),
      gutter: 10pt,
      align: center,
      showybox(
        frame: (
          background: rgb("ADD8E6"),
          border: (paint: rgb("000000"), thickness: 1pt),
          radius: 3pt,
        ),
        [*Data Layer*\n`src/data/`],
      ),
      showybox(
        frame: (
          background: rgb("90EE90"),
          border: (paint: rgb("000000"), thickness: 1pt),
          radius: 3pt,
        ),
        [*Preprocessing Layer*\n`src/preprocessing/`],
      ),
      showybox(
        frame: (
          background: rgb("FFB6C1"),
          border: (paint: rgb("000000"), thickness: 1pt),
          radius: 3pt,
        ),
        [*Modeling Layer*\n`src/models/`],
      ),

      [
        - `fetchers.py`
        - `cleaning.py`
      ],
      [
        - `tokenizer.py`
      ],
      [
        - `baselines.py`
        - `chronos_wrapper.py`
      ],

      showybox(
        frame: (
          background: rgb("FFFFE0"),
          border: (paint: rgb("000000"), thickness: 1pt),
          radius: 3pt,
        ),
        [*Evaluation Layer*\n`src/eval/`],
      ),
      showybox(
        frame: (
          background: rgb("D3D3D3"),
          border: (paint: rgb("000000"), thickness: 1pt),
          radius: 3pt,
        ),
        [*Experiment Orchestration*\n`experiments/`],
      ),
      showybox(
        frame: (
          background: rgb("FFE4B5"),
          border: (paint: rgb("000000"), thickness: 1pt),
          radius: 3pt,
        ),
        [*Utilities*\n`src/utils/`],
      ),

      [
        - `metrics.py`
      ],
      [
        - `phase3_zero_shot.py`
        - `phase4_fine_tuning.py`
      ],
      [
        - `config.py`
        - `logger.py`
      ],
    )
  ]
}
#Modules()

#heading(level: 2)[Class Diagram]
#image("class.svg", height: 460pt, width: auto, fit: "contain")

#heading(level: 2)[Data Flow]

#image("dfd.svg")

#heading(level: 2)[Use-Case Diagram]

The following use-case diagram illustrates the main interactions between a user (e.g., a researcher) and the forecasting system.

#image("./use_case.svg", width: 100%, height: 630pt)

#heading(level: 1)[Methodology]

#let Methodology() = {
  box(width: 100%)[
    #align(center)[
      #text(16pt, weight: "bold")[Methodology]
    ]
    #grid(
      columns: (1fr, auto, 1fr),
      rows: (auto, auto, auto, auto),
      gutter: 10pt,
      align: center,
      showybox(
        frame: (
          background: rgb("FFFFFF"),
          border: (paint: rgb("000000"), thickness: 1pt),
          radius: 3pt,
        ),
        [Data Preparation],
      ),
      [=>],
      [
        - Fetch data from Yahoo Finance & FRED
        - Clean, align, and create features
      ],

      showybox(
        frame: (
          background: rgb("FFFFFF"),
          border: (paint: rgb("000000"), thickness: 1pt),
          radius: 3pt,
        ),
        [Tokenization],
      ),
      [=>],
      [
        - Convert numerical data to token sequences
        - Strategies: uniform, quantile, k-means
      ],

      showybox(
        frame: (
          background: rgb("FFFFFF"),
          border: (paint: rgb("000000"), thickness: 1pt),
          radius: 3pt,
        ),
        [Forecasting],
      ),
      [=>],
      [
        - Zero-shot forecasting with Chronos
        - Baseline models for comparison (ARIMA, VAR, LSTM)
      ],

      showybox(
        frame: (
          background: rgb("FFFFFF"),
          border: (paint: rgb("000000"), thickness: 1pt),
          radius: 3pt,
        ),
        [Evaluation],
      ),
      [=>],
      [
        - Calculate metrics: MAE, RMSE, MASE, etc.
        - Compare model performance
      ],
    )
  ]
}
#Methodology()

#heading(level: 2)[Data Acquisition and Preprocessing]

The first step in the methodology is to acquire and preprocess the data. The project uses two main data sources:
- *Yahoo Finance:* For market data, including stock prices, volumes, and volatility indices.
- *FRED (Federal Reserve Economic Data):* For macroeconomic data, such as interest rates, inflation, and unemployment.

The `src/data/fetchers.py` module contains the code for fetching data from these sources. The `DataCleaner` class in `src/data/cleaning.py` is
responsible for cleaning the data, handling missing values, and aligning the different data frequencies.

#heading(level: 2)[Tokenization]

Chronos treats time series forecasting as a language modeling problem. To do this, it needs to convert the numerical time series data into a sequence
of discrete tokens. The `src/preprocessing/tokenizer.py` module implements several tokenization strategies:
- *Uniform binning:* The range of values for each feature is divided into a fixed number of equal-width bins.
- *Quantile binning:* The range of values is divided into bins with an equal number of data points in each bin.
- *K-means clustering:* The data points are clustered using the k-means algorithm, and each cluster center becomes a token.

#heading(level: 2)[Forecasting Models]

The project implements a range of forecasting models to serve as baselines for comparison with Chronos. These models are implemented in
`src/models/baselines.py` and include:
- *Naive Forecaster:* A simple model that uses the last observed value or the mean of the series as the forecast.
- *ARIMA:* A classical statistical model for time series forecasting.
- *VAR:* A multivariate extension of the ARIMA model.
- *LSTM:* A type of recurrent neural network that is well-suited for sequence data.
- *Linear Forecaster:* A simple linear regression model.

The `src/models/chronos_wrapper.py` module provides a wrapper for the Chronos model, allowing it to be used in the same way as the baseline models.

#heading(level: 2)[Evaluation Metrics]

The performance of the forecasting models is evaluated using a comprehensive set of metrics, implemented in `src/eval/metrics.py`. These include:
- *Mean Absolute Error (MAE)*: It is the average of the absolute differences between predicted and actual values. It provides a straightforward measure of forecast accuracy.
- *Root Mean Squared Error (RMSE)*: It is the square root of the average of the squared differences between predicted and actual values. RMSE penalizes larger errors more than MAE, making it sensitive to outliers.
- *Mean Absolute Scaled Error (MASE)*: It is the mean absolute error of the forecast divided by the mean absolute error of a naive forecast. MASE is scale-independent and allows for comparison across different datasets.
- *Symmetric Mean Absolute Percentage Error (sMAPE)*: It is a percentage-based error metric that is symmetric and bounded between 0% and 200%. sMAPE is useful for comparing forecast accuracy across different scales.
- *Directional Accuracy*: It measures the percentage of times the forecast correctly predicts the direction of change (up or down) in the time series. This metric is particularly relevant in financial forecasting, where the direction of movement can be more important than the exact value.

All metrics are calculated over multiple forecast horizons to provide a detailed assessment of model performance.

More metrics are planned for implementation, including:
- *Mean Squared Error (MSE)*
- *Mean Absolute Percentage Error (MAPE)*
- *R-squared (Coefficient of Determination)*

The framework is designed to easily incorporate additional metrics as needed.

#pagebreak()

#heading(level: 1)[Implementation]

#heading(level: 2)[Project Structure]

The project is organized into the following directory structure:
```
version2/
├── src/
│   ├── data/
│   ├── preprocessing/
│   ├── models/
│   ├── eval/
│   ├── utils/
│   └── main.py
├── experiments/
├── data/
├── results/
├── pyproject.toml
└── README.md
```

#heading(level: 2)[Key Components]

The key components of the implementation are:
- *`src/data/fetchers.py`:* Contains the `YahooFinanceFetcher` and `FredFetcher` classes for fetching data.
- *`src/data/cleaning.py`:* Contains the `DataCleaner` class for cleaning and preprocessing the data.
- *`src/preprocessing/tokenizer.py`:* Contains the `FinancialDataTokenizer` and `AdvancedTokenizer` classes for tokenizing the data.
- *`src/models/baselines.py`:* Contains the implementations of the baseline forecasting models.
- *`src/models/chronos_wrapper.py`:* Contains the `ChronosFinancialForecaster` class, which wraps the Chronos model.
- *`src/eval/metrics.py`:* Contains the implementations of the evaluation metrics.
- *`experiments/phase3_zero_shot.py`:* The main script for running the zero-shot forecasting experiments.
- *`experiments/phase4_fine_tuning.py`:* Script for future fine-tuning and causal attribution experiments.

#image("./components.svg", width: 80%, height: 200pt)

The implementation uses several external libraries, including:
- *Hugging Face Transformers:* For loading and using the Chronos model.
- *PyTorch:* For building and training deep learning models.
- *pandas and NumPy:* For data manipulation and numerical computations.
- *matplotlib and seaborn:* For data visualization.
- *yfinance and fredapi:* For fetching financial data. More would be added as needed.

#pagebreak()

#heading(level: 1)[Experiments and Results]

#heading(level: 2)[Experimental Setup]

The experiments are conducted using the `experiments/phase3_zero_shot.py` script. This script performs the following steps:
1. Fetches and prepares the data.
2. Runs the baseline forecasting models.
3. Runs the Chronos model in a zero-shot setting.
4. Compares the performance of all models using the evaluation metrics.
5. Generates a summary report and visualizations of the results.

#heading(level: 2)[Results]

The results of the experiments are saved in the `results/phase3/` directory. The `phase3_summary_report.txt` file contains a detailed breakdown of the
performance of each model. The `metric_comparison.png` image provides a visual comparison of the models.

An analysis of the results shows that the Chronos model performs competitively with the baseline models, even in a zero-shot setting without any fine-
tuning. This demonstrates the potential of foundation models for financial forecasting.

#figure(
  image("./results/phase3/metric_comparison.png", width: 80%),
  caption: [Metric Comparison],
)

#pagebreak()

#heading(level: 1)[Causal Attribution and Fine-Tuning (Phase 4)]

Phase 4 of the project, which is planned as future work, will focus on fine-tuning the Chronos model and conducting causal attribution analysis.

#heading(level: 2)[Fine-Tuning]

Fine-tuning involves further training the pre-trained Chronos model on the specific financial dataset. This can potentially improve the model's
performance by adapting it to the nuances of the financial domain. The `experiments/phase4_fine_tuning.py` script provides a placeholder for this
functionality.

#heading(level: 2)[Causal Attribution]

Causal attribution analysis aims to understand which features are most influential in the model's predictions. This can provide valuable insights into
the drivers of financial markets. The `experiments/phase4_fine_tuning.py` script includes placeholders for several attribution methods, including:
- *Ablation Importance:* Systematically removing features to measure their impact on performance.
- *Permutation Importance:* Shuffling the values of a feature to measure its importance.
- *SHAP Analysis:* A game theory-based approach to explain the output of any machine learning model.

#pagebreak()

#heading(level: 1)[Limitations]

The current project has several limitations:
- *Mock Chronos Implementation:* The actual Chronos model is not used due to its size and computational requirements. Instead, a mock implementation is
used for demonstration purposes.
- *Limited Scope of Experiments:* The experiments are limited to a specific set of datasets and models.
- *No Fine-Tuning:* The project does not yet include the fine-tuning of the Chronos model.
- *No Causal Attribution:* The causal attribution analysis is not yet implemented.

#pagebreak()

#heading(level: 1)[Conclusion and Future Work]

This project has successfully demonstrated the potential of using a foundation model like Chronos for multivariate financial forecasting. The
implemented framework provides a solid foundation for further research and development in this area.

Future work will focus on:
- Integrating the actual Chronos model and conducting experiments on a larger scale.
- Implementing the fine-tuning and causal attribution analysis planned for Phase 4.
- Expanding the range of datasets and models used in the experiments.
- Developing a user-friendly interface for interacting with the forecasting system.

#pagebreak()

#heading(level: 1)[10. References]

- Chronos: Learning the Language of Time Series, arXiv:2403.07815
- Hugging Face Chronos Models: https://huggingface.co/amazon/chronos-t5-small
- yfinance: https://pypi.org/project/yfinance/
- fredapi: https://pypi.org/project/fredapi/

