import pandas as pd
import torch
from chronos import BaseChronosPipeline

# Load the model
pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-bolt-tiny",  # or "amazon/chronos-bolt-small" for faster inference
    device_map="cpu",  # use "cuda" if you have GPU
    torch_dtype=torch.bfloat16,
)

# Load sample data
df = pd.read_csv(
    "https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv"
)

# Make predictions
quantiles, mean = pipeline.predict_quantiles(
    context=torch.tensor(df["#Passengers"]),
    prediction_length=12,
    quantile_levels=[0.1, 0.5, 0.9],
)

print("Forecast completed!")
print(f"Quantiles shape: {quantiles.shape}")
print(f"Mean shape: {mean.shape}")
