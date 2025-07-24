#let document-title = "Chronos-Bolt for Sequential Behavioral Data Forecasting"
#let submission-details = [
  A Project report submitted in partial fulfillment of the requirements for the award of degree in        
  *BACHELOR OF TECHNOLOGY* \
  *(COMPUTER SCIENCE AND ENGINEERING)*

  *SUBMITTED BY* \
  #table(
    columns: (auto, auto),
    align: left,
    [Registration number], [Name of the Student],
    [A22126510134], [M. Bheesetti Harsith Veera Charan],
    [A22126510144], [D. Chaitanya],
    [A22126510194], [Wuna Akhilesh],
    [A22126510163], [M. Sai Teja],
    [A22126510193], [Venkata Vishaal Tirupalli],
  )

  *UNDER THE GUIDANCE OF* \
  Dr. D. Naga Teja \                                                                                              Associate professor

  #box(width: 2.4in, height: 2.4in)[
   // #image("anits_logo.png", width: 2.4in, height: 2.4in)
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

#set page(numbering: "1")


= About Domain

This project operates within the domain of *Time-Series Forecasting* and *Sequential Data Analysis*, specifically focusing on the application of foundation models to behavioral and conversational data streams. The domain encompasses probabilistic forecasting methods, tokenization-based approaches for non-traditional time series, and zero-shot learning capabilities in pretrained models. By leveraging Chronos-Bolt, a state-of-the-art foundation model originally designed for numerical time-series forecasting, we explore its adaptability to sequential behavioral patterns such as user clickstream behavior and dialogue sentiment streams.

= How it is Feasible to Present Society Needs

Modern digital interactions generate vast amounts of sequential behavioral data that require accurate forecasting for improved user experience, content personalization, and system optimization. Traditional time-series forecasting methods often struggle with the discrete, heterogeneous nature of behavioral data. This project addresses the growing societal need for:

- *Personalized Digital Experiences*: Predicting user behavior patterns to enhance recommendation systems and content delivery
- *Real-time Decision Making*: Enabling systems to anticipate user actions and sentiment shifts for proactive responses  
- *Resource Optimization*: Forecasting user engagement patterns to optimize computational resources and service delivery
- *Enhanced Human-Computer Interaction*: Understanding sequential patterns in dialogue and interaction for better conversational AI systems

= What the Problem Identified

Current approaches to behavioral sequence forecasting face several critical limitations:

1. *Domain-Specific Model Requirements*: Most forecasting models require extensive domain-specific training and cannot generalize across different types of behavioral data
2. *Limited Transfer Learning*: Existing time-series models struggle to leverage knowledge from traditional numerical forecasting when applied to tokenized behavioral sequences
3. *Lack of Zero-Shot Capabilities*: Current methods require substantial training data for each new behavioral domain, limiting their applicability in data-scarce scenarios
4. *Inefficient Tokenization Approaches*: Traditional methods for converting behavioral data into forecast-ready formats often lose critical sequential information

= What the Solution Suggested

This project proposes a novel approach that adapts Chronos-Bolt's pretrained foundation model capabilities to sequential behavioral data through:

1. *Innovative Tokenization Strategy*: Converting real-valued sliding-window features (event counts, sentiment scores) into token sequences that align with Chronos's quantization-based approach, treating behavioral signals as a pseudo-time-series "language"

2. *Zero-Shot Forecasting Evaluation*: Applying the pretrained Chronos-Bolt model without fine-tuning to assess its transfer learning capabilities across behavioral domains, measuring performance through classification accuracy (F1, AUC) and quantile forecast accuracy (MASE, CRPS)

3. *Optional Fine-Tuning Framework*: Implementing domain-specific fine-tuning using AutoGluon-TimeSeries infrastructure to quantify performance improvements over zero-shot baselines

4. *Comprehensive Performance Assessment*: Establishing evaluation metrics that account for both discrete classification tasks (next event type prediction) and continuous forecasting tasks (sentiment trend prediction)

= How the Proposed Solution is Apt for Present Needs of Users

The solution directly addresses current user and system requirements by:

- *Reducing Development Time*: Zero-shot capabilities eliminate the need for extensive model training for new behavioral domains
- *Improving Scalability*: A single pretrained model can potentially handle multiple types of sequential behavioral data
- *Enhancing Accuracy*: Leveraging foundation model capabilities trained on diverse time-series data for better generalization
- *Enabling Rapid Deployment*: Quick adaptation to new behavioral forecasting tasks without significant computational overhead

= Technologies Used

- *Chronos-Bolt*: Pretrained foundation model for probabilistic time-series forecasting
- *AutoGluon-TimeSeries*: Framework for model deployment, evaluation, and optional fine-tuning
- *Python Ecosystem*: Data preprocessing, tokenization, and evaluation pipeline
- *Statistical Evaluation Metrics*: MASE, CRPS for quantile forecasting; F1, AUC for classification tasks

= Feasibility Study

== Operational Feasibility
The project leverages existing pretrained models and established frameworks (AutoGluon), reducing operational complexity. The focus on a single domain ensures manageable scope within a 5-month undergraduate timeline.

== Technical Feasibility  
Chronos-Bolt's tokenization-based architecture is well-suited for adaptation to behavioral sequences. The use of established evaluation metrics and frameworks ensures technical viability.

== Economical Feasibility
Utilizing pretrained models minimizes computational costs. The optional nature of fine-tuning allows for budget-conscious execution while maintaining research value.

= Architecture Model

The proposed system follows a pipeline architecture:
1. *Data Preprocessing Layer*: Sliding-window feature extraction and normalization
2. *Tokenization Layer*: Conversion of behavioral signals to token sequences
3. *Forecasting Layer*: Chronos-Bolt model application (zero-shot and optionally fine-tuned)
4. *Evaluation Layer*: Performance assessment using domain-appropriate metrics

= Expected Contributions

- Demonstrate the transferability of Chronos-Bolt's pretrained capabilities to non-traditional sequential domains
- Establish feasibility of token-based forecasting for behavioral data streams  
- Provide empirical comparison between zero-shot and fine-tuned performance in real sequential domains
- Create a reusable framework for applying foundation models to diverse behavioral forecasting tasks

= References

#set enum(numbering: "[1]")
+ Amazon Web Services, Inc., "Fast and accurate zero-shot forecasting with Chronos-Bolt and AutoGluon"
+ Amazon Science, "Chronos: Adapting language model architectures for time series forecasting"
+ "Chronos: Learning the Language of Time Series," arXiv:2403.07815
+ Hugging Face, "amazon/chronos-bolt-small" model repository

= Conclusion

This project represents a novel exploration of foundation model capabilities in behavioral forecasting, potentially opening new avenues for efficient, scalable prediction systems in digital interaction domains. The combination of zero-shot evaluation and optional fine-tuning provides a comprehensive assessment framework that could inform future applications of pretrained models to sequential behavioral data.