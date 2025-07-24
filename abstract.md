## 🚀 Project Abstract

**Title (tentative):** **“Zero‑Shot Token‑Based Forecasting of
Sequential Behavioral Data using Chronos‑Bolt”**

**Summary:** We propose to explore the application of
**Chronos‑Bolt**, a pretrained foundation model for
probabilistic time‑series forecasting, to a novel sequential
domain such as **user clickstream behavior** or **dialogue
sentiment streams**. By discretizing real‑valued
sliding‑window features (e.g., event counts, sentiment scores)
into token sequences, we treat these behavioral signals as a
pseudo‑time‑series “language,” in line with Chronos’s
quantization-based tokenization approach ([Amazon Web
Services, Inc.][1], [Amazon Science][2]).

Our primary goal is to evaluate Chronos‑Bolt’s **zero‑shot
forecasting performance** on the chosen domain—i.e. applying
the pretrained model without fine‑tuning. If time and
computational resources permit, we will optionally fine‑tune
Chronos‑Bolt on tokenized sequences to assess gains in
accuracy using AutoGluon‑TimeSeries infrastructure ([HKU SPACE
AI Hub][3]).

**Research questions include:**

* Can zero‑shot Chronos‑Bolt predict future behavioral tokens
  (e.g. next event type, sentiment shift) with acceptable
  accuracy?
* Does fine‑tuning yield measurable improvements in
  forecasting performance for domain‑specific sequential data?

**Methodology:**

1. **Domain & dataset selection**: choose one domain (e.g.
Kaggle clickstream, conversation logs with sentiment scores).
2. **Data preprocessing**: compute sliding‑window metrics,
normalize scales, discretize into fixed token vocabulary.
3. **Zero‑shot evaluation**: use Chronos‑Bolt via AutoGluon to
forecast target token sequences; compute metrics like
classification accuracy (F1, AUC) or quantile forecast
accuracy (MASE, CRPS) depending on output type.
4. **Optional fine‑tuning**: if feasible, fine‑tune
Chronos‑Bolt on a training split using AutoGluon’s
`fine_tune=True` setting and compare to zero‑shot baseline
([arXiv][4]).

**Expected contributions:**

* Demonstrate whether Chronos’s pre‑trained capabilities
  transfer to behavioral or textual-like sequential domains
  without retraining.
* Show feasibility of token-based forecasting in
  non-traditional time series.
* Provide comparisons of zero‑shot vs fine‑tuned performance
  in a real sequential domain.

**Scope & constraints:**

* Focus on a single domain to manage complexity within a
  5‑month undergraduate team project.
* Zero‑shot evaluation is mandatory; fine‑tuning is optional
  and contingent on resource availability and schedule.
* Use pre‑trained Chronos‑Bolt models (e.g., mini, small, or
  base) to avoid heavy training overhead and simplify pipeline
  integration via AutoGluon ([Reddit][5], [HKU SPACE AI
  Hub][3], [huggingface.co][6]).

---


[1]:
https://aws.amazon.com/blogs/machine-learning/fast-and-accurate-zero-shot-forecasting-with-chronos-bolt-and-autogluon/?utm_source=chatgpt.com
"Fast and accurate zero-shot forecasting with Chronos-Bolt and
AutoGluon | Artificial Intelligence" [2]:
https://www.amazon.science/blog/adapting-language-model-architectures-for-time-series-forecasting/?utm_source=chatgpt.com
"Chronos: Adapting language model architectures for time
series forecasting - Amazon Science" [3]:
https://aihub.hkuspace.hku.hk/2024/12/03/fast-and-accurate-zero-shot-forecasting-with-chronos-bolt-and-autogluon/?utm_source=chatgpt.com
"Fast and accurate zero-shot forecasting with Chronos-Bolt and
AutoGluon - HKU SPACE AI Hub" [4]:
https://arxiv.org/abs/2403.07815?utm_source=chatgpt.com
"Chronos: Learning the Language of Time Series" [5]:
https://www.reddit.com/r/MachineLearning/comments/1behp7t?utm_source=chatgpt.com
"[R] Chronos: Learning the Language of Time Series" [6]:
https://huggingface.co/amazon/chronos-bolt-small?utm_source=chatgpt.com
"amazon/chronos-bolt-small · Hugging Face"