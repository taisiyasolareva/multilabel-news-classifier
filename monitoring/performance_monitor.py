"""Model performance monitoring utilities."""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Monitor model performance over time.
    
    Tracks metrics, detects performance degradation, and generates alerts.
    """

    def __init__(
        self,
        metrics_file: str = "monitoring/performance_metrics.json",
        window_size: int = 100,
        threshold_drop: float = 0.05,
    ):
        """
        Initialize performance monitor.
        
        Args:
            metrics_file: Path to store metrics
            window_size: Number of predictions to track
            threshold_drop: Threshold for performance drop alert (5%)
        """
        self.metrics_file = Path(metrics_file)
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        self.window_size = window_size
        self.threshold_drop = threshold_drop
        
        # Load existing metrics
        self.metrics_history = self._load_metrics()
        
        logger.info(f"PerformanceMonitor initialized: window_size={window_size}")

    def _load_metrics(self) -> List[Dict]:
        """Load metrics from file."""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metrics: {e}")
                return []
        return []

    def _save_metrics(self) -> None:
        """Save metrics to file."""
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")

    def record_prediction(
        self,
        prediction: Dict,
        ground_truth: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Record a prediction for monitoring.
        
        Args:
            prediction: Prediction results (tags, scores)
            ground_truth: Optional ground truth labels
            metadata: Optional metadata (timestamp, model_version, etc.)
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "prediction": prediction,
            "ground_truth": ground_truth,
            "metadata": metadata or {},
        }
        
        # Calculate metrics if ground truth available
        if ground_truth:
            metrics = self._calculate_metrics(prediction, ground_truth)
            record["metrics"] = metrics
        
        self.metrics_history.append(record)
        
        # Keep only recent records
        if len(self.metrics_history) > self.window_size * 2:
            self.metrics_history = self.metrics_history[-self.window_size:]
        
        self._save_metrics()

    def _calculate_metrics(
        self,
        prediction: Dict,
        ground_truth: Dict,
    ) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Args:
            prediction: Prediction results
            ground_truth: Ground truth labels
            
        Returns:
            Dictionary of metrics
        """
        pred_tags = set(prediction.get("tags", []))
        true_tags = set(ground_truth.get("tags", []))
        
        if not true_tags:
            return {}
        
        # Calculate precision, recall, F1
        tp = len(pred_tags & true_tags)
        fp = len(pred_tags - true_tags)
        fn = len(true_tags - pred_tags)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Exact match
        exact_match = 1.0 if pred_tags == true_tags else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "exact_match": exact_match,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    def get_recent_metrics(
        self,
        window: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Get recent performance metrics.
        
        Args:
            window: Number of recent predictions to analyze
            
        Returns:
            Dictionary of average metrics
        """
        window = window or self.window_size
        recent = self.metrics_history[-window:]
        
        if not recent:
            return {}
        
        # Filter records with metrics
        records_with_metrics = [r for r in recent if "metrics" in r]
        
        if not records_with_metrics:
            return {}
        
        # Calculate averages
        metrics_list = [r["metrics"] for r in records_with_metrics]
        
        return {
            "precision": np.mean([m["precision"] for m in metrics_list]),
            "recall": np.mean([m["recall"] for m in metrics_list]),
            "f1": np.mean([m["f1"] for m in metrics_list]),
            "exact_match": np.mean([m["exact_match"] for m in metrics_list]),
            "count": len(records_with_metrics),
        }

    def check_performance_degradation(
        self,
        baseline_metrics: Optional[Dict[str, float]] = None,
    ) -> Tuple[bool, Dict]:
        """
        Check for performance degradation.
        
        Args:
            baseline_metrics: Baseline metrics to compare against
            
        Returns:
            Tuple of (is_degraded, degradation_info)
        """
        recent_metrics = self.get_recent_metrics()
        
        if not recent_metrics or not baseline_metrics:
            return False, {}
        
        degradation_info = {}
        is_degraded = False
        
        for metric in ["precision", "recall", "f1"]:
            if metric in baseline_metrics and metric in recent_metrics:
                drop = baseline_metrics[metric] - recent_metrics[metric]
                drop_pct = drop / baseline_metrics[metric] if baseline_metrics[metric] > 0 else 0
                
                degradation_info[metric] = {
                    "baseline": baseline_metrics[metric],
                    "current": recent_metrics[metric],
                    "drop": drop,
                    "drop_pct": drop_pct,
                }
                
                if drop_pct > self.threshold_drop:
                    is_degraded = True
                    logger.warning(
                        f"Performance degradation detected: {metric} dropped by "
                        f"{drop_pct:.2%} ({baseline_metrics[metric]:.3f} -> {recent_metrics[metric]:.3f})"
                    )
        
        return is_degraded, degradation_info

    def get_performance_trends(
        self,
        metric: str = "f1",
        days: int = 7,
    ) -> pd.DataFrame:
        """
        Get performance trends over time.
        
        Args:
            metric: Metric to track
            days: Number of days to analyze
            
        Returns:
            DataFrame with trends
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        filtered = [
            r for r in self.metrics_history
            if datetime.fromisoformat(r["timestamp"]) >= cutoff_date
            and "metrics" in r
            and metric in r["metrics"]
        ]
        
        if not filtered:
            return pd.DataFrame()
        
        data = {
            "timestamp": [r["timestamp"] for r in filtered],
            metric: [r["metrics"][metric] for r in filtered],
        }
        
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        return df

    def generate_report(self) -> Dict:
        """
        Generate performance monitoring report.
        
        Returns:
            Dictionary with report data
        """
        recent_metrics = self.get_recent_metrics()
        trends = self.get_performance_trends()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "recent_metrics": recent_metrics,
            "total_predictions": len(self.metrics_history),
            "predictions_with_metrics": len([r for r in self.metrics_history if "metrics" in r]),
        }
        
        if not trends.empty:
            report["trends"] = {
                "f1_mean": trends["f1"].mean() if "f1" in trends.columns else None,
                "f1_std": trends["f1"].std() if "f1" in trends.columns else None,
                "f1_min": trends["f1"].min() if "f1" in trends.columns else None,
                "f1_max": trends["f1"].max() if "f1" in trends.columns else None,
            }
        
        return report




