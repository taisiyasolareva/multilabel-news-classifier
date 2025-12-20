"""Error analysis utilities for multi-label classification."""

import logging
from typing import List, Dict, Tuple, Optional
import torch
import pandas as pd
import numpy as np
from collections import defaultdict, Counter

from evaluation.metrics import per_class_metrics, confusion_matrix_per_class

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrorAnalyzer:
    """
    Analyze classification errors for multi-label classification.
    
    Identifies common misclassification patterns, false positives/negatives,
    and provides insights for model improvement.
    """
    
    def __init__(self):
        """Initialize error analyzer."""
        pass
    
    def analyze_false_positives(
        self,
        target: torch.Tensor,
        y_pred: torch.Tensor,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, List[int]]:
        """
        Identify false positive predictions per class.
        
        Args:
            target: Ground truth binary matrix [batch_size, num_classes]
            y_pred: Predicted binary matrix [batch_size, num_classes]
            class_names: Optional list of class names
            
        Returns:
            Dictionary mapping class name to list of sample indices with false positives
            
        Example:
            >>> analyzer = ErrorAnalyzer()
            >>> target = torch.tensor([[0, 1], [1, 0]])
            >>> pred = torch.tensor([[1, 1], [1, 0]])
            >>> fps = analyzer.analyze_false_positives(target, pred)
            >>> fps["class_0"]
            [0]
        """
        num_classes = target.shape[1]
        if class_names is None:
            class_names = [f"class_{i}" for i in range(num_classes)]
        
        false_positives = {name: [] for name in class_names}
        
        for i in range(num_classes):
            class_target = target[:, i]
            class_pred = y_pred[:, i]
            
            # False positives: predicted but not in target
            fp_mask = (class_pred == 1) & (class_target == 0)
            fp_indices = torch.where(fp_mask)[0].tolist()
            
            false_positives[class_names[i]] = fp_indices
        
        return false_positives
    
    def analyze_false_negatives(
        self,
        target: torch.Tensor,
        y_pred: torch.Tensor,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, List[int]]:
        """
        Identify false negative predictions per class.
        
        Args:
            target: Ground truth binary matrix [batch_size, num_classes]
            y_pred: Predicted binary matrix [batch_size, num_classes]
            class_names: Optional list of class names
            
        Returns:
            Dictionary mapping class name to list of sample indices with false negatives
        """
        num_classes = target.shape[1]
        if class_names is None:
            class_names = [f"class_{i}" for i in range(num_classes)]
        
        false_negatives = {name: [] for name in class_names}
        
        for i in range(num_classes):
            class_target = target[:, i]
            class_pred = y_pred[:, i]
            
            # False negatives: in target but not predicted
            fn_mask = (class_pred == 0) & (class_target == 1)
            fn_indices = torch.where(fn_mask)[0].tolist()
            
            false_negatives[class_names[i]] = fn_indices
        
        return false_negatives
    
    def find_common_misclassification_patterns(
        self,
        target: torch.Tensor,
        y_pred: torch.Tensor,
        class_names: Optional[List[str]] = None,
        top_k: int = 10
    ) -> List[Tuple[Tuple[str, ...], Tuple[str, ...], int]]:
        """
        Find common patterns of misclassification.
        
        Identifies frequently co-occurring classes that are misclassified together.
        
        Args:
            target: Ground truth binary matrix [batch_size, num_classes]
            y_pred: Predicted binary matrix [batch_size, num_classes]
            class_names: Optional list of class names
            top_k: Number of top patterns to return
            
        Returns:
            List of tuples: (predicted_classes, actual_classes, count)
            Sorted by frequency (descending)
        """
        num_classes = target.shape[1]
        if class_names is None:
            class_names = [f"class_{i}" for i in range(num_classes)]
        
        patterns = Counter()
        
        for sample_idx in range(target.shape[0]):
            # Get predicted and actual classes
            pred_classes = tuple(sorted([
                class_names[i] for i in range(num_classes) if y_pred[sample_idx, i] == 1
            ]))
            actual_classes = tuple(sorted([
                class_names[i] for i in range(num_classes) if target[sample_idx, i] == 1
            ]))
            
            # Only count if there's a mismatch
            if pred_classes != actual_classes:
                patterns[(pred_classes, actual_classes)] += 1
        
        # Return top K patterns
        return patterns.most_common(top_k)
    
    def analyze_class_confusion(
        self,
        target: torch.Tensor,
        y_pred: torch.Tensor,
        class_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Analyze confusion between classes.
        
        Creates a confusion matrix showing which classes are frequently
        confused with each other.
        
        Args:
            target: Ground truth binary matrix [batch_size, num_classes]
            y_pred: Predicted binary matrix [batch_size, num_classes]
            class_names: Optional list of class names
            
        Returns:
            DataFrame with confusion analysis
        """
        num_classes = target.shape[1]
        if class_names is None:
            class_names = [f"class_{i}" for i in range(num_classes)]
        
        # Count confusions: when class A is predicted but class B is actual
        confusion_counts = defaultdict(int)
        
        for sample_idx in range(target.shape[0]):
            pred_indices = set(i for i in range(num_classes) if y_pred[sample_idx, i] == 1)
            actual_indices = set(i for i in range(num_classes) if target[sample_idx, i] == 1)
            
            # False positives: predicted but not actual
            for pred_idx in pred_indices - actual_indices:
                for actual_idx in actual_indices:
                    confusion_counts[(class_names[pred_idx], class_names[actual_idx])] += 1
        
        # Create DataFrame
        if confusion_counts:
            data = [
                {"predicted": pred, "actual": actual, "count": count}
                for (pred, actual), count in confusion_counts.items()
            ]
            df = pd.DataFrame(data)
            df = df.sort_values("count", ascending=False)
        else:
            df = pd.DataFrame(columns=["predicted", "actual", "count"])
        
        return df
    
    def get_error_summary(
        self,
        target: torch.Tensor,
        y_pred: torch.Tensor,
        class_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Get comprehensive error summary.
        
        Args:
            target: Ground truth binary matrix [batch_size, num_classes]
            y_pred: Predicted binary matrix [batch_size, num_classes]
            class_names: Optional list of class names
            
        Returns:
            Dictionary with error statistics
        """
        num_classes = target.shape[1]
        if class_names is None:
            class_names = [f"class_{i}" for i in range(num_classes)]
        
        # Get per-class metrics
        per_class = per_class_metrics(target, y_pred, class_names)
        
        # Calculate totals
        total_fp = sum(metrics["fp"] for metrics in per_class.values())
        total_fn = sum(metrics["fn"] for metrics in per_class.values())
        total_tp = sum(metrics["tp"] for metrics in per_class.values())
        total_tn = sum(metrics["tn"] for metrics in per_class.values())
        
        # Find classes with most errors
        classes_by_fp = sorted(
            per_class.items(),
            key=lambda x: x[1]["fp"],
            reverse=True
        )[:10]
        
        classes_by_fn = sorted(
            per_class.items(),
            key=lambda x: x[1]["fn"],
            reverse=True
        )[:10]
        
        return {
            "total_samples": target.shape[0],
            "total_classes": num_classes,
            "total_false_positives": total_fp,
            "total_false_negatives": total_fn,
            "total_true_positives": total_tp,
            "total_true_negatives": total_tn,
            "fp_rate": total_fp / (total_fp + total_tn + 1e-5),
            "fn_rate": total_fn / (total_fn + total_tp + 1e-5),
            "top_fp_classes": [
                {"class": name, "count": metrics["fp"]}
                for name, metrics in classes_by_fp
            ],
            "top_fn_classes": [
                {"class": name, "count": metrics["fn"]}
                for name, metrics in classes_by_fn
            ],
            "per_class_metrics": per_class
        }
    
    def visualize_errors(
        self,
        target: torch.Tensor,
        y_pred: torch.Tensor,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Create visualizations-ready DataFrames for error analysis.
        
        Args:
            target: Ground truth binary matrix [batch_size, num_classes]
            y_pred: Predicted binary matrix [batch_size, num_classes]
            class_names: Optional list of class names
            
        Returns:
            Dictionary with DataFrames for visualization
        """
        num_classes = target.shape[1]
        if class_names is None:
            class_names = [f"class_{i}" for i in range(num_classes)]
        
        # Per-class metrics DataFrame
        per_class = per_class_metrics(target, y_pred, class_names)
        metrics_df = pd.DataFrame(per_class).T
        
        # Confusion analysis DataFrame
        confusion_df = self.analyze_class_confusion(target, y_pred, class_names)
        
        # Error counts per class
        error_counts = []
        for name, metrics in per_class.items():
            error_counts.append({
                "class": name,
                "false_positives": metrics["fp"],
                "false_negatives": metrics["fn"],
                "true_positives": metrics["tp"],
                "true_negatives": metrics["tn"]
            })
        error_df = pd.DataFrame(error_counts)
        
        return {
            "per_class_metrics": metrics_df,
            "confusion_analysis": confusion_df,
            "error_counts": error_df
        }



