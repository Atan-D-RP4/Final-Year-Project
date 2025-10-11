"""
Cross-Validation Framework for Time Series Forecasting
Implements time-series specific validation methods and comprehensive evaluation metrics
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Callable
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")


class TimeSeriesCrossValidator:
    """
    Time series cross-validation with multiple strategies
    """
    
    def __init__(self, 
                 strategy: str = 'expanding',
                 n_splits: int = 5,
                 test_size: int = 5,
                 gap: int = 0,
                 min_train_size: int = 20):
        """
        Initialize time series cross-validator
        
        Args:
            strategy: 'expanding', 'sliding', or 'blocked'
            n_splits: Number of cross-validation splits
            test_size: Size of test set in each split
            gap: Gap between train and test sets
            min_train_size: Minimum training set size
        """
        self.strategy = strategy
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.min_train_size = min_train_size
    
    def split(self, data: List[float]) -> List[Tuple[List[int], List[int]]]:
        """
        Generate train/test splits for time series data
        
        Args:
            data: Time series data
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        n_samples = len(data)
        splits = []
        
        if self.strategy == 'expanding':
            # Expanding window: training set grows with each split
            for i in range(self.n_splits):
                test_start = self.min_train_size + i * self.test_size + self.gap
                test_end = test_start + self.test_size
                
                if test_end > n_samples:
                    break
                
                train_indices = list(range(test_start - self.gap))
                test_indices = list(range(test_start, test_end))
                
                if len(train_indices) >= self.min_train_size:
                    splits.append((train_indices, test_indices))
        
        elif self.strategy == 'sliding':
            # Sliding window: training set size remains constant
            train_size = max(self.min_train_size, 
                           (n_samples - self.n_splits * self.test_size - self.gap) // self.n_splits)
            
            for i in range(self.n_splits):
                test_start = self.min_train_size + i * self.test_size + self.gap
                test_end = test_start + self.test_size
                
                if test_end > n_samples:
                    break
                
                train_start = max(0, test_start - self.gap - train_size)
                train_end = test_start - self.gap
                
                train_indices = list(range(train_start, train_end))
                test_indices = list(range(test_start, test_end))
                
                if len(train_indices) >= self.min_train_size:
                    splits.append((train_indices, test_indices))
        
        elif self.strategy == 'blocked':
            # Blocked cross-validation: non-overlapping blocks
            block_size = n_samples // (self.n_splits * 2)  # Alternate train/test blocks
            
            for i in range(self.n_splits):
                train_start = i * 2 * block_size
                train_end = train_start + block_size
                test_start = train_end + self.gap
                test_end = test_start + min(block_size, self.test_size)
                
                if test_end > n_samples:
                    break
                
                train_indices = list(range(train_start, train_end))
                test_indices = list(range(test_start, test_end))
                
                if len(train_indices) >= self.min_train_size:
                    splits.append((train_indices, test_indices))
        
        return splits
    
    def visualize_splits(self, data: List[float], splits: List[Tuple[List[int], List[int]]]):
        """
        Visualize the cross-validation splits
        
        Args:
            data: Time series data
            splits: List of (train_indices, test_indices) tuples
        """
        fig, axes = plt.subplots(len(splits), 1, figsize=(12, 2 * len(splits)))
        if len(splits) == 1:
            axes = [axes]
        
        for i, (train_idx, test_idx) in enumerate(splits):
            ax = axes[i]
            
            # Plot full data in light gray
            ax.plot(range(len(data)), data, color='lightgray', alpha=0.5)
            
            # Plot training data in blue
            train_data = [data[j] for j in train_idx]
            ax.plot(train_idx, train_data, color='blue', label='Train')
            
            # Plot test data in red
            test_data = [data[j] for j in test_idx]
            ax.plot(test_idx, test_data, color='red', label='Test')
            
            ax.set_title(f'Split {i+1}: Train={len(train_idx)}, Test={len(test_idx)}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


class ForecastingMetrics:
    """
    Comprehensive forecasting evaluation metrics
    """
    
    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Squared Error"""
        return mean_squared_error(y_true, y_pred)
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error"""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error"""
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error"""
        return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    @staticmethod
    def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Symmetric Mean Absolute Percentage Error"""
        return np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100
    
    @staticmethod
    def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, seasonality: int = 1) -> float:
        """Mean Absolute Scaled Error"""
        # Calculate naive forecast error on training set
        if len(y_train) <= seasonality:
            naive_error = np.mean(np.abs(np.diff(y_train)))
        else:
            naive_forecast = y_train[:-seasonality]
            naive_actual = y_train[seasonality:]
            naive_error = np.mean(np.abs(naive_actual - naive_forecast))
        
        if naive_error == 0:
            return np.inf if np.any(y_true != y_pred) else 0
        
        return np.mean(np.abs(y_true - y_pred)) / naive_error
    
    @staticmethod
    def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Directional Accuracy (percentage of correct direction predictions)"""
        if len(y_true) < 2:
            return 0.0
        
        true_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))
        
        return np.mean(true_direction == pred_direction) * 100
    
    @staticmethod
    def forecast_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Forecast Bias (mean of prediction errors)"""
        return np.mean(y_pred - y_true)
    
    @staticmethod
    def prediction_interval_coverage(y_true: np.ndarray, 
                                   lower_bound: np.ndarray, 
                                   upper_bound: np.ndarray) -> float:
        """Prediction Interval Coverage Probability"""
        coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
        return coverage * 100
    
    @staticmethod
    def theil_u_statistic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Theil's U Statistic"""
        mse_pred = np.mean((y_true - y_pred) ** 2)
        mse_naive = np.mean((y_true[1:] - y_true[:-1]) ** 2)
        
        if mse_naive == 0:
            return np.inf if mse_pred > 0 else 0
        
        return np.sqrt(mse_pred / mse_naive)


class BacktestingFramework:
    """
    Comprehensive backtesting framework for time series forecasting
    """
    
    def __init__(self, 
                 forecaster,
                 cv_strategy: str = 'expanding',
                 n_splits: int = 5,
                 prediction_length: int = 5,
                 window_size: int = 10):
        """
        Initialize backtesting framework
        
        Args:
            forecaster: Forecasting model with fit_transform and forecast methods
            cv_strategy: Cross-validation strategy
            n_splits: Number of CV splits
            prediction_length: Forecast horizon
            window_size: Input window size
        """
        self.forecaster = forecaster
        self.cv_strategy = cv_strategy
        self.n_splits = n_splits
        self.prediction_length = prediction_length
        self.window_size = window_size
        
        self.cv = TimeSeriesCrossValidator(
            strategy=cv_strategy,
            n_splits=n_splits,
            test_size=prediction_length,
            min_train_size=window_size * 2
        )
        
        self.metrics = ForecastingMetrics()
        self.results = {}\n    \n    def run_backtest(self, data: List[float], \n                    show_progress: bool = True,\n                    return_predictions: bool = False) -> Dict[str, Any]:\n        \"\"\"\n        Run comprehensive backtest\n        \n        Args:\n            data: Time series data\n            show_progress: Whether to show progress\n            return_predictions: Whether to return individual predictions\n            \n        Returns:\n            Dictionary with backtest results\n        \"\"\"\n        if show_progress:\n            print(f\"Running backtest with {self.cv_strategy} CV, {self.n_splits} splits...\")\n        \n        splits = self.cv.split(data)\n        \n        if not splits:\n            raise ValueError(\"No valid splits generated. Check data size and parameters.\")\n        \n        all_predictions = []\n        all_actuals = []\n        all_train_data = []\n        split_results = []\n        \n        for i, (train_idx, test_idx) in enumerate(splits):\n            if show_progress:\n                print(f\"  Processing split {i+1}/{len(splits)}...\")\n            \n            # Extract train and test data\n            train_data = [data[j] for j in train_idx]\n            test_data = [data[j] for j in test_idx]\n            \n            try:\n                # Prepare training data\n                tokenized_data, tokenizer = self.forecaster.prepare_data(\n                    train_data, self.window_size\n                )\n                \n                # Make forecast\n                forecast_result = self.forecaster.forecast_zero_shot(\n                    tokenized_data, len(test_data)\n                )\n                \n                predictions = forecast_result['mean'][0].numpy()\n                \n                # Store results\n                all_predictions.extend(predictions)\n                all_actuals.extend(test_data)\n                all_train_data.append(train_data)\n                \n                # Calculate metrics for this split\n                split_metrics = self._calculate_metrics(\n                    np.array(test_data), \n                    np.array(predictions), \n                    np.array(train_data)\n                )\n                \n                split_results.append({\n                    'split': i + 1,\n                    'train_size': len(train_data),\n                    'test_size': len(test_data),\n                    'metrics': split_metrics,\n                    'predictions': predictions.tolist() if return_predictions else None,\n                    'actuals': test_data if return_predictions else None\n                })\n                \n            except Exception as e:\n                if show_progress:\n                    print(f\"    Error in split {i+1}: {e}\")\n                continue\n        \n        if not all_predictions:\n            raise ValueError(\"No successful predictions made\")\n        \n        # Calculate overall metrics\n        overall_metrics = self._calculate_metrics(\n            np.array(all_actuals),\n            np.array(all_predictions),\n            np.concatenate(all_train_data) if all_train_data else np.array([])\n        )\n        \n        # Calculate metric statistics across splits\n        metric_stats = self._calculate_metric_statistics(split_results)\n        \n        self.results = {\n            'overall_metrics': overall_metrics,\n            'metric_statistics': metric_stats,\n            'split_results': split_results,\n            'cv_strategy': self.cv_strategy,\n            'n_splits': len(splits),\n            'total_predictions': len(all_predictions),\n            'all_predictions': all_predictions if return_predictions else None,\n            'all_actuals': all_actuals if return_predictions else None\n        }\n        \n        if show_progress:\n            print(\"  Backtest completed!\")\n        \n        return self.results\n    \n    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, \n                          y_train: np.ndarray) -> Dict[str, float]:\n        \"\"\"Calculate all metrics for given predictions\"\"\"\n        metrics = {\n            'mse': self.metrics.mse(y_true, y_pred),\n            'rmse': self.metrics.rmse(y_true, y_pred),\n            'mae': self.metrics.mae(y_true, y_pred),\n            'mape': self.metrics.mape(y_true, y_pred),\n            'smape': self.metrics.smape(y_true, y_pred),\n            'directional_accuracy': self.metrics.directional_accuracy(y_true, y_pred),\n            'forecast_bias': self.metrics.forecast_bias(y_true, y_pred),\n            'theil_u': self.metrics.theil_u_statistic(y_true, y_pred)\n        }\n        \n        # Add MASE if training data is available\n        if len(y_train) > 1:\n            metrics['mase'] = self.metrics.mase(y_true, y_pred, y_train)\n        \n        return metrics\n    \n    def _calculate_metric_statistics(self, split_results: List[Dict]) -> Dict[str, Dict[str, float]]:\n        \"\"\"Calculate statistics across splits for each metric\"\"\"\n        metric_names = list(split_results[0]['metrics'].keys())\n        stats = {}\n        \n        for metric in metric_names:\n            values = [split['metrics'][metric] for split in split_results \n                     if not np.isnan(split['metrics'][metric]) and not np.isinf(split['metrics'][metric])]\n            \n            if values:\n                stats[metric] = {\n                    'mean': np.mean(values),\n                    'std': np.std(values),\n                    'min': np.min(values),\n                    'max': np.max(values),\n                    'median': np.median(values)\n                }\n            else:\n                stats[metric] = {\n                    'mean': np.nan, 'std': np.nan, 'min': np.nan, \n                    'max': np.nan, 'median': np.nan\n                }\n        \n        return stats\n    \n    def print_results(self):\n        \"\"\"Print formatted backtest results\"\"\"\n        if not self.results:\n            print(\"No results to display. Run backtest first.\")\n            return\n        \n        print(\"\\n\" + \"=\" * 70)\n        print(\"TIME SERIES FORECASTING BACKTEST RESULTS\")\n        print(\"=\" * 70)\n        \n        print(f\"\\nBacktest Configuration:\")\n        print(f\"  CV Strategy: {self.results['cv_strategy']}\")\n        print(f\"  Number of Splits: {self.results['n_splits']}\")\n        print(f\"  Total Predictions: {self.results['total_predictions']}\")\n        print(f\"  Prediction Length: {self.prediction_length}\")\n        \n        print(f\"\\nOverall Performance:\")\n        overall = self.results['overall_metrics']\n        for metric, value in overall.items():\n            print(f\"  {metric.upper()}: {value:.4f}\")\n        \n        print(f\"\\nCross-Validation Statistics:\")\n        stats = self.results['metric_statistics']\n        print(f\"{'Metric':<20} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}\")\n        print(\"-\" * 70)\n        \n        for metric, stat_dict in stats.items():\n            print(f\"{metric.upper():<20} {stat_dict['mean']:<10.4f} {stat_dict['std']:<10.4f} \"\n                  f\"{stat_dict['min']:<10.4f} {stat_dict['max']:<10.4f}\")\n    \n    def plot_results(self):\n        \"\"\"Plot backtest results\"\"\"\n        if not self.results or not self.results.get('all_predictions'):\n            print(\"No prediction data to plot. Run backtest with return_predictions=True.\")\n            return\n        \n        predictions = self.results['all_predictions']\n        actuals = self.results['all_actuals']\n        \n        fig, axes = plt.subplots(2, 2, figsize=(15, 10))\n        \n        # Time series plot\n        axes[0, 0].plot(actuals, label='Actual', alpha=0.7)\n        axes[0, 0].plot(predictions, label='Predicted', alpha=0.7)\n        axes[0, 0].set_title('Predictions vs Actuals')\n        axes[0, 0].legend()\n        axes[0, 0].grid(True, alpha=0.3)\n        \n        # Scatter plot\n        axes[0, 1].scatter(actuals, predictions, alpha=0.6)\n        min_val, max_val = min(min(actuals), min(predictions)), max(max(actuals), max(predictions))\n        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)\n        axes[0, 1].set_xlabel('Actual')\n        axes[0, 1].set_ylabel('Predicted')\n        axes[0, 1].set_title('Prediction Scatter Plot')\n        axes[0, 1].grid(True, alpha=0.3)\n        \n        # Residuals\n        residuals = np.array(actuals) - np.array(predictions)\n        axes[1, 0].plot(residuals)\n        axes[1, 0].axhline(y=0, color='r', linestyle='--')\n        axes[1, 0].set_title('Residuals Over Time')\n        axes[1, 0].set_ylabel('Residual')\n        axes[1, 0].grid(True, alpha=0.3)\n        \n        # Residual histogram\n        axes[1, 1].hist(residuals, bins=20, alpha=0.7, edgecolor='black')\n        axes[1, 1].axvline(x=0, color='r', linestyle='--')\n        axes[1, 1].set_title('Residual Distribution')\n        axes[1, 1].set_xlabel('Residual')\n        axes[1, 1].set_ylabel('Frequency')\n        axes[1, 1].grid(True, alpha=0.3)\n        \n        plt.tight_layout()\n        plt.show()\n    \n    def compare_strategies(self, data: List[float], strategies: List[str]) -> Dict[str, Dict]:\n        \"\"\"Compare different CV strategies\"\"\"\n        comparison_results = {}\n        \n        original_strategy = self.cv_strategy\n        \n        for strategy in strategies:\n            print(f\"\\nTesting {strategy} strategy...\")\n            self.cv_strategy = strategy\n            self.cv.strategy = strategy\n            \n            try:\n                results = self.run_backtest(data, show_progress=False)\n                comparison_results[strategy] = results['overall_metrics']\n            except Exception as e:\n                print(f\"Error with {strategy}: {e}\")\n                comparison_results[strategy] = None\n        \n        # Restore original strategy\n        self.cv_strategy = original_strategy\n        self.cv.strategy = original_strategy\n        \n        # Print comparison\n        print(\"\\n\" + \"=\" * 80)\n        print(\"CROSS-VALIDATION STRATEGY COMPARISON\")\n        print(\"=\" * 80)\n        \n        if any(result is not None for result in comparison_results.values()):\n            # Get metric names from first successful result\n            metric_names = None\n            for result in comparison_results.values():\n                if result is not None:\n                    metric_names = list(result.keys())\n                    break\n            \n            if metric_names:\n                print(f\"{'Strategy':<15} \", end=\"\")\n                for metric in metric_names:\n                    print(f\"{metric.upper():<10} \", end=\"\")\n                print()\n                print(\"-\" * (15 + len(metric_names) * 11))\n                \n                for strategy, results in comparison_results.items():\n                    print(f\"{strategy:<15} \", end=\"\")\n                    if results:\n                        for metric in metric_names:\n                            print(f\"{results[metric]:<10.4f} \", end=\"\")\n                    else:\n                        print(\"FAILED\" + \" \" * (len(metric_names) * 11 - 6), end=\"\")\n                    print()\n        \n        return comparison_results\n\n\ndef test_cross_validation():\n    \"\"\"\n    Test the cross-validation framework\n    \"\"\"\n    print(\"Testing Cross-Validation Framework...\")\n    \n    # Generate synthetic time series data\n    np.random.seed(42)\n    n_samples = 100\n    trend = np.linspace(0, 10, n_samples)\n    seasonal = 3 * np.sin(2 * np.pi * np.arange(n_samples) / 12)\n    noise = np.random.normal(0, 1, n_samples)\n    data = (trend + seasonal + noise + 50).tolist()\n    \n    print(f\"Generated synthetic data with {len(data)} points\")\n    \n    # Test different CV strategies\n    strategies = ['expanding', 'sliding', 'blocked']\n    \n    for strategy in strategies:\n        print(f\"\\nTesting {strategy} strategy:\")\n        cv = TimeSeriesCrossValidator(\n            strategy=strategy,\n            n_splits=5,\n            test_size=5,\n            min_train_size=20\n        )\n        \n        splits = cv.split(data)\n        print(f\"  Generated {len(splits)} splits\")\n        \n        for i, (train_idx, test_idx) in enumerate(splits[:2]):  # Show first 2 splits\n            print(f\"    Split {i+1}: Train={len(train_idx)}, Test={len(test_idx)}\")\n    \n    # Test metrics\n    print(\"\\nTesting metrics:\")\n    y_true = np.array([1, 2, 3, 4, 5])\n    y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])\n    y_train = np.array([0.5, 0.8, 1.2, 1.5])\n    \n    metrics = ForecastingMetrics()\n    print(f\"  MSE: {metrics.mse(y_true, y_pred):.4f}\")\n    print(f\"  MAE: {metrics.mae(y_true, y_pred):.4f}\")\n    print(f\"  MAPE: {metrics.mape(y_true, y_pred):.4f}\")\n    print(f\"  MASE: {metrics.mase(y_true, y_pred, y_train):.4f}\")\n    print(f\"  Directional Accuracy: {metrics.directional_accuracy(y_true, y_pred):.2f}%\")\n    \n    print(\"\\nCross-validation framework test completed!\")\n    return data\n\n\nif __name__ == \"__main__\":\n    # Run tests\n    test_data = test_cross_validation()