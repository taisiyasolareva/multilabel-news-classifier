"""Evaluation metrics for multi-label classification."""

from typing import List, Callable, Optional, Dict
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import logging

logger = logging.getLogger(__name__)


def precision(target: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calculate precision for multi-label classification.
    
    Precision = (1/n) * sum(|y_i ∩ y_pred_i| / |y_pred_i|)
    
    Args:
        target: Ground truth binary matrix [batch_size, num_classes]
        y_pred: Predicted binary matrix [batch_size, num_classes]
        
    Returns:
        Average precision score
        
    Example:
        >>> target = torch.tensor([[1, 0, 1], [0, 1, 0]])
        >>> pred = torch.tensor([[1, 1, 1], [0, 1, 0]])
        >>> precision(target, pred)
        0.75
    """
    num = ((y_pred == 1) & (target == 1)).sum(dim=1).float()
    denum = (y_pred == 1).sum(dim=1).float()
    return (num / (denum + 1e-5)).mean().item()


def recall(target: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calculate recall for multi-label classification.
    
    Recall = (1/n) * sum(|y_i ∩ y_pred_i| / |y_i|)
    
    Args:
        target: Ground truth binary matrix [batch_size, num_classes]
        y_pred: Predicted binary matrix [batch_size, num_classes]
        
    Returns:
        Average recall score
        
    Example:
        >>> target = torch.tensor([[1, 0, 1], [0, 1, 0]])
        >>> pred = torch.tensor([[1, 1, 1], [0, 1, 0]])
        >>> recall(target, pred)
        1.0
    """
    num = ((y_pred == 1) & (target == 1)).sum(dim=1).float()
    denum = (target == 1).sum(dim=1).float()
    return (num / (denum + 1e-5)).mean().item()


def f1_score(target: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calculate F1 score for multi-label classification.
    
    F1 = 2 * (precision * recall) / (precision + recall)
    
    Args:
        target: Ground truth binary matrix [batch_size, num_classes]
        y_pred: Predicted binary matrix [batch_size, num_classes]
        
    Returns:
        F1 score
        
    Example:
        >>> target = torch.tensor([[1, 0, 1], [0, 1, 0]])
        >>> pred = torch.tensor([[1, 1, 1], [0, 1, 0]])
        >>> f1_score(target, pred)
        0.8
    """
    prec = precision(target, y_pred)
    rec = recall(target, y_pred)
    return 2 * (prec * rec) / (prec + rec + 1e-5)


def exact_match(target: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calculate exact match accuracy.
    
    Exact Match = (1/n) * (1/k) * sum(sum([y_ij == y_pred_ij]))
    
    Args:
        target: Ground truth binary matrix [batch_size, num_classes]
        y_pred: Predicted binary matrix [batch_size, num_classes]
        
    Returns:
        Exact match score
        
    Example:
        >>> target = torch.tensor([[1, 0, 1], [0, 1, 0]])
        >>> pred = torch.tensor([[1, 0, 1], [0, 1, 0]])
        >>> exact_match(target, pred)
        1.0
    """
    return (1.0 * (y_pred == target)).mean().item()


def get_predict(
    model: torch.nn.Module,
    dataset: Dataset,
    use_snippet: bool = False,
    device: Optional[torch.device] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get predictions from model on dataset.
    
    Args:
        model: Trained model
        dataset: Dataset to predict on
        use_snippet: Whether model uses snippets
        device: Device to run inference on. If None, uses model's device
        
    Returns:
        Tuple of (predicted_probabilities, targets)
        Shapes: [num_samples, num_classes]
        
    Example:
        >>> probs, targets = get_predict(model, test_dataset, use_snippet=False)
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    dataloader = DataLoader(dataset, shuffle=False, batch_size=len(dataset))
    
    with torch.no_grad():
        for batch in dataloader:
            if use_snippet:
                title, snippet, target = batch
                title = title.to(device)
                snippet = snippet.to(device)
                logits = model(title, snippet)
            else:
                title, target = batch
                title = title.to(device)
                logits = model(title)
            
            # Convert logits to probabilities (sigmoid for multi-label)
            pred_prob = torch.sigmoid(logits)
            
            return pred_prob.cpu(), target
    
    raise RuntimeError("No data in dataset")


def get_optimal_threshold(
    model: torch.nn.Module,
    dataset: Dataset,
    threshold_list: List[float],
    metric: str = 'precision',
    use_snippet: bool = False
) -> float:
    """
    Find optimal threshold for binary classification.
    
    Args:
        model: Trained model
        dataset: Validation dataset
        threshold_list: List of thresholds to try
        metric: Metric to optimize ('precision', 'recall', or 'f1')
        use_snippet: Whether model uses snippets
        
    Returns:
        Optimal threshold value
        
    Example:
        >>> thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
        >>> best = get_optimal_threshold(model, val_dataset, thresholds, 'f1')
    """
    pred_prob, target = get_predict(model, dataset, use_snippet)
    
    best_threshold = threshold_list[0]
    best_score = 0.0
    
    metric_funcs = {
        'precision': precision,
        'recall': recall,
        'f1': f1_score,
    }
    
    if metric not in metric_funcs:
        raise ValueError(f"Unknown metric: {metric}. Choose from {list(metric_funcs.keys())}")
    
    metric_func = metric_funcs[metric]
    
    for threshold in threshold_list:
        y_pred = (pred_prob > threshold).float()
        score = metric_func(target, y_pred)
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    logger.info(f"Optimal threshold: {best_threshold} (score: {best_score:.4f})")
    return best_threshold


def per_class_metrics(
    target: torch.Tensor,
    y_pred: torch.Tensor,
    class_names: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Calculate per-class metrics (precision, recall, F1).
    
    Args:
        target: Ground truth binary matrix [batch_size, num_classes]
        y_pred: Predicted binary matrix [batch_size, num_classes]
        class_names: Optional list of class names. If None, uses indices.
        
    Returns:
        Dictionary mapping class name/index to metrics dict
        
    Example:
        >>> target = torch.tensor([[1, 0, 1], [0, 1, 0]])
        >>> pred = torch.tensor([[1, 1, 1], [0, 1, 0]])
        >>> metrics = per_class_metrics(target, pred, ["class1", "class2", "class3"])
        >>> metrics["class1"]["precision"]
        0.5
    """
    num_classes = target.shape[1]
    
    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]
    
    per_class = {}
    
    for i in range(num_classes):
        class_target = target[:, i]
        class_pred = y_pred[:, i]
        
        # True positives, false positives, false negatives
        tp = ((class_pred == 1) & (class_target == 1)).sum().float().item()
        fp = ((class_pred == 1) & (class_target == 0)).sum().float().item()
        fn = ((class_pred == 0) & (class_target == 1)).sum().float().item()
        tn = ((class_pred == 0) & (class_target == 0)).sum().float().item()
        
        # Calculate metrics
        prec = tp / (tp + fp + 1e-5)
        rec = tp / (tp + fn + 1e-5)
        f1 = 2 * (prec * rec) / (prec + rec + 1e-5)
        
        # Support (number of true positives)
        support = class_target.sum().float().item()
        
        per_class[class_names[i]] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn
        }
    
    return per_class


def confusion_matrix_per_class(
    target: torch.Tensor,
    y_pred: torch.Tensor,
    class_names: Optional[List[str]] = None
) -> Dict[str, torch.Tensor]:
    """
    Calculate confusion matrix for each class.
    
    Args:
        target: Ground truth binary matrix [batch_size, num_classes]
        y_pred: Predicted binary matrix [batch_size, num_classes]
        class_names: Optional list of class names
        
    Returns:
        Dictionary mapping class name/index to 2x2 confusion matrix
        Each matrix is [[TN, FP], [FN, TP]]
        
    Example:
        >>> target = torch.tensor([[1, 0], [0, 1]])
        >>> pred = torch.tensor([[1, 1], [0, 1]])
        >>> matrices = confusion_matrix_per_class(target, pred)
        >>> matrices["class_0"]
        tensor([[1., 0.],
                [0., 1.]])
    """
    num_classes = target.shape[1]
    
    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]
    
    confusion_matrices = {}
    
    for i in range(num_classes):
        class_target = target[:, i]
        class_pred = y_pred[:, i]
        
        # Calculate confusion matrix components
        tn = ((class_pred == 0) & (class_target == 0)).sum().float()
        fp = ((class_pred == 1) & (class_target == 0)).sum().float()
        fn = ((class_pred == 0) & (class_target == 1)).sum().float()
        tp = ((class_pred == 1) & (class_target == 1)).sum().float()
        
        # Create 2x2 confusion matrix
        # [[TN, FP],
        #  [FN, TP]]
        matrix = torch.tensor([[tn, fp], [fn, tp]])
        
        confusion_matrices[class_names[i]] = matrix
    
    return confusion_matrices

