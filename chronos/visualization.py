"""
Advanced Visualization Module for Chronos Forecasting
Provides comprehensive plotting capabilities for time series analysis and forecasting results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple, Union
import warnings

# Try to import plotly for interactive plots
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not available. Interactive plots will not be available.")

warnings.filterwarnings("ignore")

# Set style
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class ForecastVisualizer:
    """
    Comprehensive visualization for forecasting results
    """

    def __init__(self, figsize: Tuple[int, int] = (12, 8), style: str = "seaborn-v0_8"):
        """
        Initialize visualizer

        Args:
            figsize: Default figure size
            style: Matplotlib style
        """
        self.figsize = figsize
        self.style = style
        plt.style.use(style)

    def plot_time_series(
        self,
        data: Union[List[float], Dict[str, List[float]]],
        title: str = "Time Series Data",
        labels: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        show_trend: bool = False,
        show_seasonal: bool = False,
    ) -> plt.Figure:
        """
        Plot time series data

        Args:
            data: Time series data (single series or multiple series)
            title: Plot title
            labels: Series labels
            save_path: Path to save figure
            show_trend: Whether to show trend line
            show_seasonal: Whether to show seasonal decomposition

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        if isinstance(data, dict):
            # Multiple series
            for i, (name, series) in enumerate(data.items()):
                ax.plot(series, label=name, alpha=0.8, linewidth=2)
        else:
            # Single series
            ax.plot(data, label=labels[0] if labels else "Data", alpha=0.8, linewidth=2)

            if show_trend:
                # Add trend line
                x = np.arange(len(data))
                z = np.polyfit(x, data, 1)
                p = np.poly1d(z)
                ax.plot(x, p(x), "--", alpha=0.7, label="Trend")

        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_forecast_results(
        self,
        actual: List[float],
        predicted: List[float],
        train_data: Optional[List[float]] = None,
        confidence_intervals: Optional[Dict[str, List[float]]] = None,
        title: str = "Forecast Results",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot forecasting results with actual vs predicted

        Args:
            actual: Actual values
            predicted: Predicted values
            train_data: Training data (optional)
            confidence_intervals: Dict with 'lower' and 'upper' bounds
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot training data if provided
        if train_data:
            train_x = range(len(train_data))
            ax.plot(train_x, train_data, "b-", alpha=0.6, label="Training Data")
            forecast_start = len(train_data)
        else:
            forecast_start = 0

        # Plot actual and predicted
        forecast_x = range(forecast_start, forecast_start + len(actual))
        ax.plot(
            forecast_x,
            actual,
            "g-",
            linewidth=2,
            label="Actual",
            marker="o",
            markersize=4,
        )
        ax.plot(
            forecast_x,
            predicted,
            "r--",
            linewidth=2,
            label="Predicted",
            marker="s",
            markersize=4,
        )

        # Plot confidence intervals if provided
        if confidence_intervals:
            ax.fill_between(
                forecast_x,
                confidence_intervals["lower"],
                confidence_intervals["upper"],
                alpha=0.3,
                color="red",
                label="Confidence Interval",
            )

        # Add vertical line to separate train/test
        if train_data:
            ax.axvline(
                x=forecast_start,
                color="black",
                linestyle=":",
                alpha=0.7,
                label="Forecast Start",
            )

        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_residual_analysis(
        self,
        actual: List[float],
        predicted: List[float],
        title: str = "Residual Analysis",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot comprehensive residual analysis

        Args:
            actual: Actual values
            predicted: Predicted values
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        residuals = np.array(actual) - np.array(predicted)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight="bold")

        # 1. Residuals over time
        axes[0, 0].plot(residuals, "b-", alpha=0.7)
        axes[0, 0].axhline(y=0, color="r", linestyle="--")
        axes[0, 0].set_title("Residuals Over Time")
        axes[0, 0].set_xlabel("Time")
        axes[0, 0].set_ylabel("Residual")
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Residual histogram
        axes[0, 1].hist(residuals, bins=20, alpha=0.7, edgecolor="black")
        axes[0, 1].axvline(x=0, color="r", linestyle="--")
        axes[0, 1].set_title("Residual Distribution")
        axes[0, 1].set_xlabel("Residual")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Q-Q plot
        from scipy import stats

        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title("Q-Q Plot (Normal Distribution)")
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Residuals vs fitted values
        axes[1, 1].scatter(predicted, residuals, alpha=0.6)
        axes[1, 1].axhline(y=0, color="r", linestyle="--")
        axes[1, 1].set_title("Residuals vs Fitted Values")
        axes[1, 1].set_xlabel("Fitted Values")
        axes[1, 1].set_ylabel("Residuals")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_model_comparison(
        self,
        results: Dict[str, Dict[str, Any]],
        metrics: List[str] = ["mse", "mae", "mape", "mase"],
        title: str = "Model Comparison",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot comparison of multiple models

        Args:
            results: Dictionary with model results
            metrics: List of metrics to compare
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]

        fig.suptitle(title, fontsize=16, fontweight="bold")

        model_names = list(results.keys())

        for i, metric in enumerate(metrics):
            values = []
            labels = []

            for model_name in model_names:
                if metric in results[model_name]:
                    value = results[model_name][metric]
                    if not np.isnan(value) and not np.isinf(value):
                        values.append(value)
                        labels.append(model_name)

            if values:
                bars = axes[i].bar(range(len(values)), values)
                axes[i].set_title(f"{metric.upper()}")
                axes[i].set_xticks(range(len(labels)))
                axes[i].set_xticklabels(labels, rotation=45, ha="right")
                axes[i].grid(True, alpha=0.3)

                # Color bars based on performance (lower is better for most metrics)
                if metric.lower() in ["mse", "mae", "mape", "mase", "rmse"]:
                    # Lower is better - color best performance in green
                    best_idx = np.argmin(values)
                    for j, bar in enumerate(bars):
                        if j == best_idx:
                            bar.set_color("green")
                        else:
                            bar.set_color("lightblue")
                else:
                    # Higher is better - color best performance in green
                    best_idx = np.argmax(values)
                    for j, bar in enumerate(bars):
                        if j == best_idx:
                            bar.set_color("green")
                        else:
                            bar.set_color("lightblue")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_feature_importance(
        self,
        importance_dict: Dict[str, float],
        title: str = "Feature Importance",
        top_n: int = 15,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot feature importance

        Args:
            importance_dict: Dictionary mapping feature names to importance scores
            title: Plot title
            top_n: Number of top features to show
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        if not importance_dict:
            print("No feature importance data to plot")
            return None

        # Sort features by importance
        sorted_features = sorted(
            importance_dict.items(), key=lambda x: x[1], reverse=True
        )[:top_n]

        features, scores = zip(*sorted_features)

        fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.4)))

        y_pos = np.arange(len(features))
        bars = ax.barh(y_pos, scores)

        # Color bars in gradient
        colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()  # Top feature at top
        ax.set_xlabel("Importance Score")
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_cross_validation_results(
        self,
        cv_results: Dict[str, Any],
        title: str = "Cross-Validation Results",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot cross-validation results

        Args:
            cv_results: Cross-validation results dictionary
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        if "split_results" not in cv_results:
            print("No split results found in CV results")
            return None

        split_results = cv_results["split_results"]

        # Extract metrics across splits
        metrics_data = {}
        for split in split_results:
            for metric, value in split["metrics"].items():
                if metric not in metrics_data:
                    metrics_data[metric] = []
                metrics_data[metric].append(value)

        # Create subplots
        n_metrics = len(metrics_data)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle(title, fontsize=16, fontweight="bold")

        for i, (metric, values) in enumerate(metrics_data.items()):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]

            # Filter out invalid values
            valid_values = [v for v in values if not np.isnan(v) and not np.isinf(v)]

            if valid_values:
                splits = range(1, len(valid_values) + 1)
                ax.plot(splits, valid_values, "o-", linewidth=2, markersize=6)
                ax.set_title(f"{metric.upper()}")
                ax.set_xlabel("Split")
                ax.set_ylabel(metric.upper())
                ax.grid(True, alpha=0.3)

                # Add mean line
                mean_value = np.mean(valid_values)
                ax.axhline(
                    y=mean_value,
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                    label=f"Mean: {mean_value:.4f}",
                )
                ax.legend()

        # Hide empty subplots
        for i in range(n_metrics, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if n_rows > 1:
                axes[row, col].set_visible(False)
            elif n_cols > 1:
                axes[col].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig


class InteractivePlotter:
    """
    Interactive plotting using Plotly (if available)
    """

    def __init__(self):
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for interactive plotting")

    def plot_interactive_forecast(
        self,
        actual: List[float],
        predicted: List[float],
        train_data: Optional[List[float]] = None,
        confidence_intervals: Optional[Dict[str, List[float]]] = None,
        title: str = "Interactive Forecast",
    ) -> go.Figure:
        """
        Create interactive forecast plot

        Args:
            actual: Actual values
            predicted: Predicted values
            train_data: Training data (optional)
            confidence_intervals: Confidence intervals (optional)
            title: Plot title

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        # Add training data if provided
        if train_data:
            train_x = list(range(len(train_data)))
            fig.add_trace(
                go.Scatter(
                    x=train_x,
                    y=train_data,
                    mode="lines",
                    name="Training Data",
                    line=dict(color="blue", width=2),
                    opacity=0.7,
                )
            )
            forecast_start = len(train_data)
        else:
            forecast_start = 0

        # Add forecast data
        forecast_x = list(range(forecast_start, forecast_start + len(actual)))

        # Add confidence intervals if provided
        if confidence_intervals:
            fig.add_trace(
                go.Scatter(
                    x=forecast_x + forecast_x[::-1],
                    y=confidence_intervals["upper"]
                    + confidence_intervals["lower"][::-1],
                    fill="toself",
                    fillcolor="rgba(255,0,0,0.2)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="Confidence Interval",
                    showlegend=True,
                )
            )

        # Add actual values
        fig.add_trace(
            go.Scatter(
                x=forecast_x,
                y=actual,
                mode="lines+markers",
                name="Actual",
                line=dict(color="green", width=3),
                marker=dict(size=6),
            )
        )

        # Add predicted values
        fig.add_trace(
            go.Scatter(
                x=forecast_x,
                y=predicted,
                mode="lines+markers",
                name="Predicted",
                line=dict(color="red", width=3, dash="dash"),
                marker=dict(size=6, symbol="square"),
            )
        )

        # Add vertical line for forecast start
        if train_data:
            fig.add_vline(
                x=forecast_start,
                line_dash="dot",
                line_color="black",
                annotation_text="Forecast Start",
            )

        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Value",
            hovermode="x unified",
            template="plotly_white",
        )

        return fig

    def plot_interactive_comparison(
        self,
        results: Dict[str, Dict[str, Any]],
        metric: str = "mase",
        title: str = "Interactive Model Comparison",
    ) -> go.Figure:
        """
        Create interactive model comparison plot

        Args:
            results: Model results dictionary
            metric: Metric to compare
            title: Plot title

        Returns:
            Plotly figure
        """
        model_names = []
        values = []

        for model_name, model_results in results.items():
            if metric in model_results:
                value = model_results[metric]
                if not np.isnan(value) and not np.isinf(value):
                    model_names.append(model_name)
                    values.append(value)

        # Sort by performance
        sorted_data = sorted(zip(model_names, values), key=lambda x: x[1])
        model_names, values = zip(*sorted_data)

        # Create color scale (green for best, red for worst)
        colors = px.colors.sample_colorscale("RdYlGn_r", len(values))

        fig = go.Figure(
            data=[
                go.Bar(
                    x=list(model_names),
                    y=list(values),
                    marker_color=colors,
                    text=[f"{v:.4f}" for v in values],
                    textposition="auto",
                )
            ]
        )

        fig.update_layout(
            title=title,
            xaxis_title="Model",
            yaxis_title=metric.upper(),
            template="plotly_white",
        )

        return fig


def create_forecast_report(
    actual: List[float],
    predicted: List[float],
    train_data: Optional[List[float]] = None,
    model_name: str = "Chronos",
    save_dir: str = "forecast_report",
) -> str:
    """
    Create a comprehensive forecast report with multiple visualizations

    Args:
        actual: Actual values
        predicted: Predicted values
        train_data: Training data (optional)
        model_name: Name of the forecasting model
        save_dir: Directory to save report files

    Returns:
        Path to the main report file
    """
    import os

    # Create directory
    os.makedirs(save_dir, exist_ok=True)

    visualizer = ForecastVisualizer()

    # 1. Main forecast plot
    fig1 = visualizer.plot_forecast_results(
        actual,
        predicted,
        train_data,
        title=f"{model_name} Forecast Results",
        save_path=os.path.join(save_dir, "forecast_results.png"),
    )
    plt.close(fig1)

    # 2. Residual analysis
    fig2 = visualizer.plot_residual_analysis(
        actual,
        predicted,
        title=f"{model_name} Residual Analysis",
        save_path=os.path.join(save_dir, "residual_analysis.png"),
    )
    plt.close(fig2)

    # 3. Calculate metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = (
        np.mean(
            np.abs((np.array(actual) - np.array(predicted)) / (np.array(actual) + 1e-8))
        )
        * 100
    )

    # 4. Create HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{model_name} Forecast Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #333; }}
            h2 {{ color: #666; }}
            .metric {{ background-color: #f0f0f0; padding: 10px; margin: 10px 0; border-radius: 5px; }}
            .metric-value {{ font-weight: bold; color: #2c3e50; }}
            img {{ max-width: 100%; height: auto; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>{model_name} Forecast Report</h1>

        <h2>Performance Metrics</h2>
        <div class="metric">Mean Squared Error (MSE): <span class="metric-value">{mse:.6f}</span></div>
        <div class="metric">Root Mean Squared Error (RMSE): <span class="metric-value">{rmse:.6f}</span></div>
        <div class="metric">Mean Absolute Error (MAE): <span class="metric-value">{mae:.6f}</span></div>
        <div class="metric">Mean Absolute Percentage Error (MAPE): <span class="metric-value">{mape:.2f}%</span></div>

        <h2>Forecast Results</h2>
        <img src="forecast_results.png" alt="Forecast Results">

        <h2>Residual Analysis</h2>
        <img src="residual_analysis.png" alt="Residual Analysis">

        <h2>Summary</h2>
        <p>This report shows the forecasting performance of the {model_name} model.</p>
        <p>Total predictions: {len(predicted)}</p>
        <p>Training data points: {len(train_data) if train_data else "N/A"}</p>

    </body>
    </html>
    """

    report_path = os.path.join(save_dir, "forecast_report.html")
    with open(report_path, "w") as f:
        f.write(html_content)

    print(f"Forecast report saved to: {report_path}")
    return report_path


def test_visualization():
    """
    Test visualization functions
    """
    print("Testing Visualization Module...")

    # Generate synthetic data
    np.random.seed(42)
    n_train = 50
    n_test = 10

    # Training data
    train_data = np.cumsum(np.random.randn(n_train)) + 100

    # Test data
    actual = np.cumsum(np.random.randn(n_test)) + train_data[-1]
    predicted = actual + np.random.normal(0, 0.5, n_test)

    print(f"Generated data: {n_train} train, {n_test} test points")

    # Test visualizer
    visualizer = ForecastVisualizer()

    # 1. Test time series plot
    print("1. Testing time series plot...")
    fig1 = visualizer.plot_time_series(
        train_data.tolist(), title="Test Time Series", show_trend=True
    )
    plt.show()

    # 2. Test forecast results plot
    print("2. Testing forecast results plot...")
    fig2 = visualizer.plot_forecast_results(
        actual.tolist(),
        predicted.tolist(),
        train_data.tolist(),
        title="Test Forecast Results",
    )
    plt.show()

    # 3. Test residual analysis
    print("3. Testing residual analysis...")
    fig3 = visualizer.plot_residual_analysis(
        actual.tolist(), predicted.tolist(), title="Test Residual Analysis"
    )
    plt.show()

    # 4. Test model comparison
    print("4. Testing model comparison...")
    mock_results = {
        "Model A": {"mse": 0.5, "mae": 0.3, "mape": 5.2},
        "Model B": {"mse": 0.7, "mae": 0.4, "mape": 6.1},
        "Model C": {"mse": 0.3, "mae": 0.2, "mape": 4.8},
    }

    fig4 = visualizer.plot_model_comparison(mock_results, title="Test Model Comparison")
    plt.show()

    # 5. Test feature importance
    print("5. Testing feature importance...")
    mock_importance = {
        "feature_1": 0.25,
        "feature_2": 0.18,
        "feature_3": 0.15,
        "feature_4": 0.12,
        "feature_5": 0.10,
        "feature_6": 0.08,
        "feature_7": 0.07,
        "feature_8": 0.05,
    }

    fig5 = visualizer.plot_feature_importance(
        mock_importance, title="Test Feature Importance"
    )
    plt.show()

    # 6. Test interactive plotting (if available)
    if PLOTLY_AVAILABLE:
        print("6. Testing interactive plotting...")
        interactive_plotter = InteractivePlotter()

        fig_interactive = interactive_plotter.plot_interactive_forecast(
            actual.tolist(),
            predicted.tolist(),
            train_data.tolist(),
            title="Test Interactive Forecast",
        )

        # Save as HTML
        fig_interactive.write_html("test_interactive_forecast.html")
        print("   Interactive plot saved as 'test_interactive_forecast.html'")

    # 7. Test report generation
    print("7. Testing report generation...")
    report_path = create_forecast_report(
        actual.tolist(),
        predicted.tolist(),
        train_data.tolist(),
        model_name="Test Model",
        save_dir="test_report",
    )

    print("Visualization module test completed!")
    return visualizer


if __name__ == "__main__":
    # Run tests
    visualizer = test_visualization()

