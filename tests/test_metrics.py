"""Tests for evaluation metrics."""

import pytest
import torch
from evaluation.metrics import precision, recall, exact_match


def test_precision():
    """Test precision calculation."""
    target = torch.tensor([[1, 0, 1], [0, 1, 0], [1, 1, 1]])
    pred = torch.tensor([[1, 1, 1], [0, 1, 0], [1, 1, 0]])
    
    # Sample 0: 2 correct / 3 predicted = 2/3
    # Sample 1: 1 correct / 1 predicted = 1/1
    # Sample 2: 2 correct / 2 predicted = 2/2
    # Average: (2/3 + 1/1 + 2/2) / 3 = (0.667 + 1.0 + 1.0) / 3 ≈ 0.889
    
    prec = precision(target, pred)
    assert 0.8 < prec < 0.95  # Approximate check


def test_recall():
    """Test recall calculation."""
    target = torch.tensor([[1, 0, 1], [0, 1, 0], [1, 1, 1]])
    pred = torch.tensor([[1, 1, 1], [0, 1, 0], [1, 1, 0]])
    
    # Sample 0: 2 correct / 2 actual = 1.0
    # Sample 1: 1 correct / 1 actual = 1.0
    # Sample 2: 2 correct / 3 actual = 2/3
    # Average: (1.0 + 1.0 + 2/3) / 3 ≈ 0.889
    
    rec = recall(target, pred)
    assert 0.8 < rec < 0.95  # Approximate check


def test_exact_match():
    """Test exact match calculation."""
    target = torch.tensor([[1, 0, 1], [0, 1, 0]])
    pred = torch.tensor([[1, 0, 1], [0, 1, 0]])
    
    # Perfect match
    em = exact_match(target, pred)
    assert em == 1.0
    
    # No match
    pred2 = torch.tensor([[0, 1, 0], [1, 0, 1]])
    em2 = exact_match(target, pred2)
    assert em2 == 0.0

