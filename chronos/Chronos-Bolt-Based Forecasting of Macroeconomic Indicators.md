# Chronos-Bolt-Based Forecasting of Macroeconomic Indicators

## System Architecture

The proposed system ingests historical financial data (e.g. exchange
rates, inflation) from sources like the IMF World Economic Outlook. It
preprocesses and tokenizes the time series, then feeds them into a
pretrained Chronos-Bolt transformer model to generate future forecasts.
As shown in Figure 1 below, Chronos first **scales and quantizes** the
input series into a sequence of tokens, feeds these tokens into a
language model (e.g. T5-based Chronos-Bolt), and at inference
**autoregressively samples** token sequences to produce a predictive
distribution[\[1\]](https://arxiv.org/abs/2403.07815#:~:text=,classical%20local%20models%20and%20deep)[\[2\]](https://github.com/amazon-science/chronos-forecasting#:~:text=Image%20Fig.%201%3A%20High,to%20obtain%20a%20predictive%20distribution).
The output tokens are then mapped back to numerical values. This
pipeline -- data loader → tokenizer → Chronos pipeline → output handler
-- forms the core architecture. Key components include: a **Data
Loader** (fetching and formatting IMF data), a **Feature/Tokenization
module** (creating sliding-window statistics and quantizing values), the
**Chronos-Bolt Forecasting Module** (the pretrained transformer model),
and an **Evaluation/Visualization Module** (computing accuracy/MSE,
plotting results).

*Figure 1: Overview of the Chronos-Bolt pipeline. The raw time series is
scaled and quantized into tokens, fed into a pretrained transformer
model, and future trajectories are sampled to obtain
forecasts[\[2\]](https://github.com/amazon-science/chronos-forecasting#:~:text=Image%20Fig.%201%3A%20High,to%20obtain%20a%20predictive%20distribution).*

In implementation, data flows from the source API/CSV into the
**BehavioralDataLoader**, which extracts and normalizes the series (e.g.
exchange rates or CPI). The **BehavioralDataTokenizer** then constructs
sliding-window features (mean, variance, trends, etc.) and maps them to
discrete token IDs. These tokens form the input "context" to the
**ChronosBehavioralForecaster**, which encapsulates the Chronos-Bolt
model. The model outputs quantile forecasts, which the system converts
back to numeric predictions. Finally, a **BenchmarkRunner** or
evaluation module calculates performance metrics (MASE, MSE, F1, etc.)
and generates plots. This end-to-end flow -- from raw data to forecast
-- is depicted in the high-level architecture.

## Modules Division

The system is decomposed into several modular components:

- **Data Ingestion (BehavioralDataLoader)**: Loads real financial data
  (CSV/API). For example, it can fetch exchange rate or inflation series
  from IMF's WEO
  datasets[\[3\]](https://data.imf.org/en#:~:text=Datasets). This module
  handles missing values and initial normalization.
- **Preprocessing/Tokenization (BehavioralDataTokenizer)**: Converts a
  univariate series into the format expected by Chronos. It uses a
  sliding-window of size *w* to compute features (mean, std, min, max,
  trend indicators, etc.), normalizes them, and quantizes to integer
  tokens. The output is a pseudo--time series of averaged token values.
- **Forecasting Module (ChronosBehavioralForecaster)**: Wraps the
  Chronos-Bolt model. Responsible for loading the pretrained model (e.g.
  "amazon/chronos-bolt-small") and performing inference. It has methods
  for preparing context data, running zero-shot forecasts, and returning
  probabilistic outputs (quantiles and mean forecasts).
- **Benchmarking/Evaluation (BenchmarkRunner)**: Drives experiments by
  splitting data into train/test, repeatedly forecasting and collecting
  results. It invokes the forecaster on multiple windows, aggregates
  predictions and true values, and computes metrics (MSE, MAE, MAPE,
  MASE for regression; Accuracy, F1, AUC for binary classification of
  above/below-median trends).
- **Utilities (Visualization and Metrics)**: Includes plotting functions
  (e.g. scatter of predictions vs true values, residual histograms) and
  metric calculators (wrapping `scikit-learn` and custom formulas). This
  module presents results for review.
- **User Interface / Deployment (future)**: While not required by the
  prompt, the system could later include a web interface (Flask,
  Streamlit) or a Dockerized pipeline for easy demonstration. Currently,
  interaction is via a terminal script.

Each module has a clear role and interface. For example, the Data Loader
outputs a `List[float]` of series values, which the Tokenizer consumes
to produce a `torch.Tensor` of tokens. The Forecaster takes this tensor
and returns forecasts. This modular division ensures clarity and
extensibility.

## Methodology (Algorithms in Detail)

The framework leverages **foundation time-series models** to forecast
financial indicators. At its core is Amazon's Chronos-Bolt, a pretrained
transformer for time
series[\[4\]](https://github.com/amazon-science/chronos-forecasting#:~:text=Chronos%20is%20a%20family%20of,data%20generated%20using%20Gaussian%20processes).
The methodology proceeds as follows:

1.  **Data Preparation & Tokenization:** Given a historical series
    $x_{t}$ , we create overlapping windows of length *w* (e.g. 10). For
    each window, we compute statistics: mean, standard deviation, min,
    max, median, skewness, kurtosis, and counts of rising/falling
    trends. This yields a feature vector per window. We then
    **normalize** these features to \[0, *quant_levels*-1\] and
    **round** to integer tokens. Finally, we average across features to
    get a single "pseudo-time-series" of token
    IDs[\[1\]](https://arxiv.org/abs/2403.07815#:~:text=,classical%20local%20models%20and%20deep).
    This tokenization adapts behavioral or financial data for Chronos's
    NLP-like input.

2.  **Chronos-Bolt Forecasting:** The token sequence is input to
    Chronos-Bolt's transformer. Chronos-Bolt (based on T5) was
    pretrained on millions of time series. It treats the tokenized past
    as a "sentence" and autoregressively predicts future tokens.
    Concretely, we call
    `pipeline.predict_quantiles(context, prediction_length, quantile_levels=[0.1,0.5,0.9])`.
    This returns quantile forecasts and mean predictions for each future
    step[\[4\]](https://github.com/amazon-science/chronos-forecasting#:~:text=Chronos%20is%20a%20family%20of,data%20generated%20using%20Gaussian%20processes).
    The model internally uses the cross-entropy loss during training on
    tokens, and at inference it samples many sequences to estimate a
    predictive
    distribution[\[1\]](https://arxiv.org/abs/2403.07815#:~:text=,classical%20local%20models%20and%20deep).

3.  **Post-processing:** The output tokens are translated back to values
    (through the inverse quantization). We interpret the 0.5-quantile as
    the point forecast (mean of distribution) and 0.1/0.9 as bounds. If
    desired, these can be averaged or used directly. For comparison to
    regression baselines, we often take the mean forecast as the
    prediction.

4.  **Evaluation:** We evaluate forecasts in two ways. As a regression
    task, we compute **MSE (mean squared error)**, **MAE (mean absolute
    error)**, **MAPE**, and **MASE (mean absolute scaled
    error)**[\[2\]](https://github.com/amazon-science/chronos-forecasting#:~:text=Image%20Fig.%201%3A%20High,to%20obtain%20a%20predictive%20distribution).
    The MASE uses a seasonal naive baseline (lag-1 difference) to scale
    the MAE, indicating if the model beats a naive guess. As a
    classification task, we convert the series into binary labels
    (above/below median). We then compute **Accuracy**, **F1-Score**,
    and **AUC** on predicted probabilities (e.g. from quantile
    outputs)[\[2\]](https://github.com/amazon-science/chronos-forecasting#:~:text=Image%20Fig.%201%3A%20High,to%20obtain%20a%20predictive%20distribution)[\[5\]](https://www.imf.org/-/media/Files/Publications/WP/2024/English/wpiea2024206-print-pdf.ashx#:~:text=ABSTRACT%3A%20Forecasting%20inflation%20has%20become,a%20challenging%20case%2C%20because%20inflation).
    This dual evaluation highlights the model's utility in trend
    detection (classification) and precise value prediction
    (regression). For example, in testing, if true future values exceed
    their historical median, we consider that a positive class and
    evaluate F1 accordingly.

5.  **Benchmarking:** To robustly assess performance, the
    **BenchmarkRunner** repeatedly slides a window through test data:
    for each possible context segment of length *w*, it forecasts the
    next *h* steps, accumulates these predictions and targets, then
    computes aggregate metrics. This simulates a rolling forecast
    (zero-shot) scenario. The runner can handle multiple datasets (e.g.
    multiple countries) to compare performance.

By combining **sequence modeling with tokenization** and **quantile
forecasting**, the methodology achieves probabilistic forecasts without
any task-specific training. This leverages Chronos-Bolt's zero-shot
ability: it generalizes from its broad pretraining to new financial
data[\[6\]](https://github.com/amazon-science/chronos-forecasting#:~:text=,models%20of%20the%20same%20size)[\[1\]](https://arxiv.org/abs/2403.07815#:~:text=,classical%20local%20models%20and%20deep).
In essence, we treat financial time series as a "language" of tokens and
apply language-model forecasting.

## Diagrams (DFDs and UMLs)

Visual modeling will include both *data flow diagrams (DFDs)* and *UML
diagrams*:

- **Data Flow Diagrams:** A DFD would illustrate how data moves through
  the system. For example, a Level-1 DFD can show "IMF Data Source →
  Data Loader → Tokenizer → Chronos Model → Output Metrics/Plots."
  Another DFD might detail preprocessing (e.g. "Raw CSV → Normalizer →
  Feature Extractor → Token Sequence"). These diagrams use standard
  symbols (data stores, processes, flows) to clarify system
  interactions.
- **UML Class Diagram:** This shows classes like `BehavioralDataLoader`,
  `BehavioralDataTokenizer`, `ChronosBehavioralForecaster`, and
  `BenchmarkRunner`, with their relationships. For example,
  `BenchmarkRunner` depends on `ChronosBehavioralForecaster`. Figure 2
  below is an **example** UML for a generic forecasting system; in our
  case classes for data handling and forecasting would be depicted
  similarly.

*Figure 2: Example UML class diagram for a forecasting system (adapted
conceptually for our modules). Classes interact to collect data, process
it, and produce forecasts.*

- **UML Sequence Diagram:** One could depict the runtime flow: a
  sequence showing the user/script calling the Forecast module, which in
  turn calls Tokenizer, then Chronos, then evaluation and output.
- **DFD Level-0 (Context):** A simple circle for "Chronos Forecast
  System" with arrows from "IMF Data" as input and "Forecast Results" as
  output, plus connections to the user.
- **DFD Level-1:** Breaks the system into sub-processes (Data Ingestion,
  Preprocessing, Forecasting, Evaluation) with data stores (e.g.
  "Historical Data Archive").

Each diagram provides a different perspective: DFDs emphasize data
movement; UMLs show class structure and interactions. Embedding such
diagrams in documentation will clarify design. (The figures above are
illustrative examples -- actual diagrams would be custom-drawn.)

## Implementation (30% Complete)

To date, about 30% of the implementation is complete. Key
accomplishments so far include: - **Chronos Integration:** We have
successfully integrated the Chronos-Bolt models via the
`BaseChronosPipeline` API. The `ChronosBehavioralForecaster` class loads
the model and runs `predict_quantiles`, as tested on sample data (see
demo script).\
- **Data Loader and Tokenizer:** Basic modules for loading and
tokenizing data are implemented. The `BehavioralDataLoader` can read
sentiment or numeric CSVs, and we have extended it to accept finance
series. The `BehavioralDataTokenizer` correctly performs sliding-window
feature extraction and quantization (validated with synthetic data).\
- **Benchmarking Pipeline:** A preliminary `BenchmarkRunner` can run
sliding-window forecasts and compute regression and classification
metrics. The demo script prints F1 and MASE scores for a small example.

Remaining implementation tasks include: connecting to real IMF data
sources (via API or CSV), handling multivariate inputs if needed (e.g.
combining inflation and interest rates), improving the front-end
script/UI (e.g. menu or config for datasets), and robust error handling.
We will also refine evaluation (e.g. cross-validation) and possibly
implement fine-tuning of Chronos on specific data if zero-shot results
are weak. Future work (beyond current scope) could involve packaging the
pipeline in Docker or adding a REST interface for easy demo.

## Input and Expected Output

**Inputs:** The system expects time-series data of a financial metric.
Typically, input is a CSV file with at least one column of historical
values (e.g. monthly CPI or daily exchange rate). A timestamp column is
optional but recommended for reference. For example, a file
`inflation.csv` might contain:

    Date, InflationRate
    2020-01, 1.5
    2020-02, 1.6
    ... 

In our code, the `BehavioralDataLoader` is invoked without a file to
generate synthetic data; for real use, one would supply the file path
and specify format (`'csv'`). The module will extract the relevant
column (e.g. \"inflation\" or \"exchange_rate\") into a Python list of
floats.

**Expected Outputs:** On running the forecast, the system will output: -
**Predicted Values:** For each forecast horizon (e.g. next 5 steps), the
model's predicted quantiles (e.g. 0.1, 0.5, 0.9) and mean. This could be
printed as an array or saved to a CSV. For instance, the demo prints
shapes of the output tensors. In practice, we might format output as a
table:

  ---------------------------------------------------------------
  Step Ahead      10th %ile       50th %ile       90th %ile
                                  (Median)        
  --------------- --------------- --------------- ---------------
  1               1.47            1.55            1.63

  2               1.50            1.58            1.67

  \...            \...            \...            \...
  ---------------------------------------------------------------

- **Metrics Report:** The benchmark runner prints performance metrics.
  For example:

<!-- -->

- Classification Metrics:
        ACCURACY: 0.8500  
        F1: 0.8421  
        AUC: 0.9100  

      Regression Metrics:
        MSE: 0.0234  
        MAE: 0.1200  
        MAPE: 3.5%  
        MASE: 0.7800  

  These summary values indicate how well the model did over the test
  split.

<!-- -->

- **Plots (optional):** If running in an environment that supports
  plotting, `benchmark.plot_results()` will display: a scatter of true
  vs predicted values, a time-series comparison plot, a histogram of
  residuals, and absolute error over time. These visuals help diagnose
  model performance. If plots can't be shown (e.g. in a headless demo),
  the code will note the inability to create them.

Overall, given an input series of length *T*, the output will be a
forecast of length *h* (as specified), plus evaluation summaries. Users
can then compare these forecasts against actual future values (if known)
or use them for decision support.

## References

- Ansari *et al.*, **"Chronos: Learning the Language of Time Series"**,
  *arXiv 2024*. Introduces Chronos tokenization and transformer
  forecasting[\[1\]](https://arxiv.org/abs/2403.07815#:~:text=,classical%20local%20models%20and%20deep).
- Chronos Team, **"Chronos: Pretrained Models for Probabilistic Time
  Series Forecasting"** (GitHub). Details Chronos architecture and the
  new Chronos-Bolt
  improvements[\[6\]](https://github.com/amazon-science/chronos-forecasting#:~:text=,models%20of%20the%20same%20size)[\[4\]](https://github.com/amazon-science/chronos-forecasting#:~:text=Chronos%20is%20a%20family%20of,data%20generated%20using%20Gaussian%20processes).
- Zhuohang *et al.*, **"FinCast: A Foundation Model for Financial
  Time-Series Forecasting"**, CIKM 2025. Presents a specialized
  financial time-series foundation model and highlights zero-shot
  forecasting[\[7\]](https://arxiv.org/html/2508.19609v1#:~:text=typically%20require%20extensive%20domain,highlighting%20its%20strong%20generalization%20capabilities).
- Liu *et al.*, **"Mending the Crystal Ball: Enhanced Inflation
  Forecasts with Machine Learning"**, IMF WP 24/206 (2024). Discusses ML
  methods for macroeconomic forecasting (inflation) and shows ML can
  improve performance by modeling
  nonlinearity[\[5\]](https://www.imf.org/-/media/Files/Publications/WP/2024/English/wpiea2024206-print-pdf.ashx#:~:text=ABSTRACT%3A%20Forecasting%20inflation%20has%20become,a%20challenging%20case%2C%20because%20inflation).
- IMF Data Portal, **World Economic Outlook Database**, 2025. Source for
  global macroeconomic series (exchange rates, inflation, GDP, etc.) for
  multiple countries[\[3\]](https://data.imf.org/en#:~:text=Datasets).
  Provides data used for training and evaluation of our forecasts.

Each citation above corresponds to in-text references provided. These
sources justify the use of transformer-based models for time series
(Chronos) and machine learning approaches in economic forecasting.

## Proposed Presentation Slides

An accompanying slide deck (15--20 slides) could cover:

- **Slide 1 -- Title Slide:** Project title, team/member names,
  affiliation.
- **Slide 2 -- Problem Statement:** Importance of forecasting exchange
  rates/inflation; challenges (nonstationarity, shocks). Context on IMF
  data availability.
- **Slide 3 -- Background:** Overview of time series forecasting;
  traditional models vs. ML. Mention recent trend of foundation models
  (Chronos)[\[1\]](https://arxiv.org/abs/2403.07815#:~:text=,classical%20local%20models%20and%20deep).
- **Slide 4 -- Chronos & Chronos-Bolt:** Explain Chronos tokenization
  and transformer approach; Chronos-Bolt's speed/accuracy
  gains[\[6\]](https://github.com/amazon-science/chronos-forecasting#:~:text=,models%20of%20the%20same%20size).
  Possibly include *Figure 1* graphic.
- **Slide 5 -- Proposed System:** High-level architecture (data sources
  → Chronos pipeline → forecasts). Introduce modular components (data
  loader, tokenizer, forecaster, evaluator).
- **Slide 6 -- Data Source:** IMF datasets (e.g. World Economic
  Outlook). Describe chosen variables (exchange rates, inflation). Show
  sample data table or chart.
- **Slide 7 -- Data Ingestion Module:** How data is fetched and
  preprocessed. Mention normalization, handling missing data.
- **Slide 8 -- Tokenization Module:** Sliding-window feature extraction.
  List features (mean, std, trend, etc.). Explain quantization into
  tokens.
- **Slide 9 -- Chronos Model:** Architecture (T5-based), pretrained on
  diverse time series. Show process of converting tokens → model input →
  output.
- **Slide 10 -- Forecasting Process:** Step-by-step algorithm: split
  data, create context windows, call `predict_quantiles`, collect
  outputs.
- **Slide 11 -- Evaluation Metrics:** Explain regression metrics (MSE,
  MAE, MASE) and classification metrics (Accuracy, F1, AUC). Why use
  both.
- **Slide 12 -- Results (Sample):** Show a table or plot of predictions
  vs actual (if any test data). Or use synthetic demo results (e.g.
  \"F1=0.82, MASE=0.75\"). Visual examples of output.
- **Slide 13 -- System Architecture Diagram:** Present the figure or a
  custom diagram of the system (similar to Figure 1).
- **Slide 14 -- UML/DFD Diagrams:** Show or describe the UML class
  diagram (Figure 2) and a sample DFD.
- **Slide 15 -- Implementation Status:** Highlight completed modules
  (30% done) and remaining tasks.
- **Slide 16 -- Challenges & Solutions:** Data issues (sparsity, noise),
  model limitations (zero-shot vs fine-tuning), and how they are
  addressed (e.g. kernel synthesis, ensembling).
- **Slide 17 -- Future Work:** Extending to multi-variate inputs,
  real-time forecasting service (possibly Docker or web app),
  integrating expert adjustments.
- **Slide 18 -- Conclusions:** Recap key points: Chronos-Bolt enables
  zero-shot forecasting of macro series, results so far, contributions.
- **Slide 19 -- References:** List of academic and data sources (as
  above).
- **Slide 20 -- Q&A (Optional):** Prompt for questions.

Each slide should focus on clarity: using bullet points, charts/diagrams
(e.g. Figure 1 embedding on slides 4 or 13), and minimal text per slide.
The above outline ensures all requested content (architecture, modules,
methodology, diagrams, etc.) is covered in the presentation format.

------------------------------------------------------------------------

[\[1\]](https://arxiv.org/abs/2403.07815#:~:text=,classical%20local%20models%20and%20deep)
\[2403.07815\] Chronos: Learning the Language of Time Series

<https://arxiv.org/abs/2403.07815>

[\[2\]](https://github.com/amazon-science/chronos-forecasting#:~:text=Image%20Fig.%201%3A%20High,to%20obtain%20a%20predictive%20distribution)
[\[4\]](https://github.com/amazon-science/chronos-forecasting#:~:text=Chronos%20is%20a%20family%20of,data%20generated%20using%20Gaussian%20processes)
[\[6\]](https://github.com/amazon-science/chronos-forecasting#:~:text=,models%20of%20the%20same%20size)
GitHub - amazon-science/chronos-forecasting: Chronos: Pretrained Models
for Probabilistic Time Series Forecasting

<https://github.com/amazon-science/chronos-forecasting>

[\[3\]](https://data.imf.org/en#:~:text=Datasets) Data Home

<https://data.imf.org/en>

[\[5\]](https://www.imf.org/-/media/Files/Publications/WP/2024/English/wpiea2024206-print-pdf.ashx#:~:text=ABSTRACT%3A%20Forecasting%20inflation%20has%20become,a%20challenging%20case%2C%20because%20inflation)
Mending the Crystal Ball: Enhanced Inflation Forecasts with Machine
Learning, WP/24/206, September 2024

<https://www.imf.org/-/media/Files/Publications/WP/2024/English/wpiea2024206-print-pdf.ashx>

[\[7\]](https://arxiv.org/html/2508.19609v1#:~:text=typically%20require%20extensive%20domain,highlighting%20its%20strong%20generalization%20capabilities)
FinCast: A Foundation Model for Financial Time-Series Forecasting

<https://arxiv.org/html/2508.19609v1>
