"""Tests for model architectures."""

import pytest
import torch
from models.simple_classifier import SimpleClassifier
from models.cnn_classifier import CNNClassifier


def test_simple_classifier_title_only():
    """Test SimpleClassifier with title only."""
    model = SimpleClassifier(
        vocab_size=1000,
        embedding_dim=100,
        output_dim=50,
        use_snippet=False
    )
    
    batch_size = 4
    seq_len = 20
    title = torch.randint(0, 1000, (batch_size, seq_len))
    
    output = model(title)
    
    assert output.shape == (batch_size, 50)
    assert not torch.isnan(output).any()


def test_simple_classifier_with_snippet():
    """Test SimpleClassifier with title and snippet."""
    model = SimpleClassifier(
        vocab_size=1000,
        embedding_dim=100,
        output_dim=50,
        use_snippet=True
    )
    
    batch_size = 4
    title_len = 20
    snippet_len = 50
    title = torch.randint(0, 1000, (batch_size, title_len))
    snippet = torch.randint(0, 1000, (batch_size, snippet_len))
    
    output = model(title, snippet)
    
    assert output.shape == (batch_size, 50)
    assert not torch.isnan(output).any()


def test_cnn_classifier():
    """Test CNNClassifier."""
    model = CNNClassifier(
        vocab_size=1000,
        embedding_dim=100,
        output_dim=50,
        max_title_len=20,
        max_snippet_len=50,
        conv_channels=[64, 128],
        kernel_sizes=[3, 3],
    )
    
    batch_size = 4
    title = torch.randint(0, 1000, (batch_size, 20))
    snippet = torch.randint(0, 1000, (batch_size, 50))
    
    output = model(title, snippet)
    
    assert output.shape == (batch_size, 50)
    assert not torch.isnan(output).any()


def test_cnn_classifier_shape_consistency():
    """Test that CNN classifier handles expected input sizes correctly."""
    model = CNNClassifier(
        vocab_size=1000,
        embedding_dim=100,
        output_dim=50,
        max_title_len=20,
        max_snippet_len=50,
    )
    
    # Test with expected max sequence lengths (model is designed for fixed sizes)
    title = torch.randint(0, 1000, (2, 20))
    snippet = torch.randint(0, 1000, (2, 50))
    
    output = model(title, snippet)
    assert output.shape == (2, 50)

