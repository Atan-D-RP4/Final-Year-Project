import numpy as np
import matplotlib.pyplot as plt

def generate_pit_histogram(predicted_cdf, true_values, bins=10):
    """Generate a Probability Integral Transform (PIT) histogram."""
    # Evaluate the cumulative distribution function (CDF) at true values
    pit_values = np.array([np.interp(tv, pred_cdf[0], pred_cdf[1]) for tv, pred_cdf in zip(true_values, predicted_cdf)])

    # Create histogram
    plt.hist(pit_values, bins=bins, range=(0, 1), edgecolor="k", alpha=0.7)
    plt.title("PIT Histogram")
    plt.xlabel("PIT Values")
    plt.ylabel("Frequency")
    plt.show()

# Test data setup
np.random.seed(42)
true_values = np.random.uniform(0, 1, 100)
predicted_cdf = [(np.linspace(0, 1, 100), np.linspace(0, 1, 100)**np.random.uniform(0.5, 1.5)) for _ in range(100)]

# Validate PIT histogram generation
generate_pit_histogram(predicted_cdf, true_values, bins=10)

