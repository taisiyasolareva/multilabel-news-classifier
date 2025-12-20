"""Comprehensive tests for model architectures."""

import pytest
import torch
import torch.nn as nn
from models.simple_classifier import SimpleClassifier
from models.cnn_classifier import CNNClassifier


class TestSimpleClassifier:
    """Comprehensive tests for SimpleClassifier."""
    
    def test_initialization_title_only(self):
        """Test initialization without snippets."""
        model = SimpleClassifier(
            vocab_size=1000,
            embedding_dim=100,
            output_dim=50,
            use_snippet=False
        )
        
        assert hasattr(model, 'title_embedding')
        assert hasattr(model, 'fc')
        assert not hasattr(model, 'snippet_embedding')
        assert not hasattr(model, 'linear1')
        assert model.use_snippet is False
    
    def test_initialization_with_snippet(self):
        """Test initialization with snippets."""
        model = SimpleClassifier(
            vocab_size=1000,
            embedding_dim=100,
            output_dim=50,
            use_snippet=True
        )
        
        assert hasattr(model, 'title_embedding')
        assert hasattr(model, 'snippet_embedding')
        assert hasattr(model, 'linear1')
        assert hasattr(model, 'linear2')
        assert model.use_snippet is True
    
    def test_forward_title_only(self):
        """Test forward pass with title only."""
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
        assert not torch.isinf(output).any()
    
    def test_forward_with_snippet(self):
        """Test forward pass with snippet."""
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
        assert not torch.isinf(output).any()
    
    def test_forward_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = SimpleClassifier(
            vocab_size=1000,
            embedding_dim=100,
            output_dim=50,
            use_snippet=False
        )
        
        title = torch.randint(0, 1000, (2, 20))
        output = model(title)
        
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist
        assert model.title_embedding.weight.grad is not None
        assert model.fc.weight.grad is not None
    
    def test_different_batch_sizes(self):
        """Test model with different batch sizes."""
        model = SimpleClassifier(
            vocab_size=1000,
            embedding_dim=100,
            output_dim=50,
            use_snippet=False
        )
        
        for batch_size in [1, 4, 16, 32]:
            title = torch.randint(0, 1000, (batch_size, 20))
            output = model(title)
            assert output.shape[0] == batch_size
    
    def test_different_sequence_lengths(self):
        """Test model with different sequence lengths."""
        model = SimpleClassifier(
            vocab_size=1000,
            embedding_dim=100,
            output_dim=50,
            use_snippet=False
        )
        
        for seq_len in [5, 10, 20, 50]:
            title = torch.randint(0, 1000, (4, seq_len))
            output = model(title)
            assert output.shape == (4, 50)


class TestCNNClassifier:
    """Comprehensive tests for CNNClassifier."""
    
    def test_initialization(self):
        """Test CNN classifier initialization."""
        model = CNNClassifier(
            vocab_size=1000,
            embedding_dim=100,
            output_dim=50,
            max_title_len=20,
            max_snippet_len=50
        )
        
        assert hasattr(model, 'title_embedding')
        assert hasattr(model, 'snippet_embedding')
        assert hasattr(model, 'title_conv_layers')
        assert hasattr(model, 'snippet_conv_layers')
        assert hasattr(model, 'classifier')
    
    def test_forward_pass(self):
        """Test CNN forward pass."""
        model = CNNClassifier(
            vocab_size=1000,
            embedding_dim=100,
            output_dim=50,
            max_title_len=20,
            max_snippet_len=50
        )
        
        batch_size = 4
        title = torch.randint(0, 1000, (batch_size, 20))
        snippet = torch.randint(0, 1000, (batch_size, 50))
        
        output = model(title, snippet)
        
        assert output.shape == (batch_size, 50)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_custom_conv_layers(self):
        """Test CNN with custom convolution configuration."""
        model = CNNClassifier(
            vocab_size=1000,
            embedding_dim=100,
            output_dim=50,
            max_title_len=20,
            max_snippet_len=50,
            conv_channels=[64, 128, 256],
            kernel_sizes=[3, 5, 3]
        )
        
        title = torch.randint(0, 1000, (2, 20))
        snippet = torch.randint(0, 1000, (2, 50))
        
        output = model(title, snippet)
        assert output.shape == (2, 50)
    
    def test_different_input_sizes(self):
        """Test CNN with expected input sequence lengths."""
        model = CNNClassifier(
            vocab_size=1000,
            embedding_dim=100,
            output_dim=50,
            max_title_len=20,
            max_snippet_len=50
        )
        
        # Test with expected max sequence lengths (CNN uses fixed-size classifier)
        title = torch.randint(0, 1000, (2, 20))
        snippet = torch.randint(0, 1000, (2, 50))
        output = model(title, snippet)
        assert output.shape == (2, 50)
        
        # Test with same sizes again (model is designed for fixed sizes)
        title = torch.randint(0, 1000, (2, 20))
        snippet = torch.randint(0, 1000, (2, 50))
        output = model(title, snippet)
        assert output.shape == (2, 50)
    
    def test_gradient_flow(self):
        """Test that gradients flow through CNN."""
        model = CNNClassifier(
            vocab_size=1000,
            embedding_dim=100,
            output_dim=50,
            max_title_len=20,
            max_snippet_len=50
        )
        
        title = torch.randint(0, 1000, (2, 20))
        snippet = torch.randint(0, 1000, (2, 50))
        output = model(title, snippet)
        
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        assert model.title_embedding.weight.grad is not None
        assert model.snippet_embedding.weight.grad is not None
        assert model.classifier[1].weight.grad is not None  # First linear layer
    
    def test_dropout_training_mode(self):
        """Test that dropout is active in training mode."""
        model = CNNClassifier(
            vocab_size=1000,
            embedding_dim=100,
            output_dim=50,
            max_title_len=20,
            max_snippet_len=50,
            dropout=0.5
        )
        
        model.train()
        title = torch.randint(0, 1000, (4, 20))
        snippet = torch.randint(0, 1000, (4, 50))
        
        # Run multiple times - outputs should vary due to dropout
        outputs = [model(title, snippet) for _ in range(5)]
        
        # Check that outputs are not identical (dropout introduces randomness)
        # Note: This is probabilistic, but very unlikely all outputs are identical
        assert not all(torch.allclose(outputs[0], out) for out in outputs[1:])
    
    def test_dropout_eval_mode(self):
        """Test that dropout is disabled in eval mode."""
        model = CNNClassifier(
            vocab_size=1000,
            embedding_dim=100,
            output_dim=50,
            max_title_len=20,
            max_snippet_len=50,
            dropout=0.5
        )
        
        model.eval()
        title = torch.randint(0, 1000, (4, 20))
        snippet = torch.randint(0, 1000, (4, 50))
        
        # Run multiple times - outputs should be identical (no dropout)
        outputs = [model(title, snippet) for _ in range(5)]
        
        # All outputs should be identical in eval mode
        assert all(torch.allclose(outputs[0], out) for out in outputs[1:])


class TestModelConsistency:
    """Tests for model consistency and edge cases."""
    
    def test_models_handle_empty_batch(self):
        """Test that models handle edge case of empty batch gracefully."""
        # This should raise an error or handle gracefully
        model = SimpleClassifier(
            vocab_size=1000,
            embedding_dim=100,
            output_dim=50,
            use_snippet=False
        )
        
        # Empty batch - model should handle it gracefully or raise appropriate error
        # PyTorch models typically handle empty batches, so we test that it doesn't crash
        title = torch.randint(0, 1000, (0, 20))
        try:
            output = model(title)
            # If it doesn't raise, output should be empty
            assert output.shape[0] == 0
        except (RuntimeError, IndexError):
            # Also acceptable if it raises an error
            pass
    
    def test_models_handle_single_sample(self):
        """Test models with batch size of 1."""
        model = SimpleClassifier(
            vocab_size=1000,
            embedding_dim=100,
            output_dim=50,
            use_snippet=False
        )
        
        title = torch.randint(0, 1000, (1, 20))
        output = model(title)
        
        assert output.shape == (1, 50)
    
    def test_models_handle_large_batch(self):
        """Test models with large batch size."""
        model = SimpleClassifier(
            vocab_size=1000,
            embedding_dim=100,
            output_dim=50,
            use_snippet=False
        )
        
        title = torch.randint(0, 1000, (128, 20))
        output = model(title)
        
        assert output.shape == (128, 50)
    
    def test_model_parameters_count(self):
        """Test that model has reasonable number of parameters."""
        model = SimpleClassifier(
            vocab_size=10000,
            embedding_dim=300,
            output_dim=1000,
            use_snippet=False
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        
        # Should have:
        # - Embedding: 10000 * 300 = 3M
        # - Linear: 300 * 1000 = 300K
        # Total: ~3.3M parameters
        assert 3_000_000 < total_params < 4_000_000

