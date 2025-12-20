"""Tests for advanced transformer architectures."""

import pytest
import torch
from models.advanced_transformers import (
    RoBERTaNewsClassifier,
    DistilBERTNewsClassifier,
    MultiHeadAttentionClassifier,
    EnsembleClassifier,
)
from models.ensemble import WeightedEnsemble, VotingEnsemble


class TestRoBERTaClassifier:
    """Tests for RoBERTa classifier."""
    
    def test_initialization(self):
        """Test RoBERTa initialization."""
        model = RoBERTaNewsClassifier(
            model_name="xlm-roberta-base",
            num_labels=10,
            use_snippet=False
        )
        
        assert hasattr(model, 'roberta')
        assert hasattr(model, 'classifier')
    
    def test_forward_pass(self):
        """Test RoBERTa forward pass."""
        model = RoBERTaNewsClassifier(num_labels=10, use_snippet=False)
        
        batch_size = 2
        title_input_ids = torch.randint(0, 1000, (batch_size, 20))
        title_attention_mask = torch.ones(batch_size, 20)
        
        output = model(
            title_input_ids=title_input_ids,
            title_attention_mask=title_attention_mask
        )
        
        assert output.shape == (batch_size, 10)


class TestDistilBERTClassifier:
    """Tests for DistilBERT classifier."""
    
    def test_initialization(self):
        """Test DistilBERT initialization."""
        model = DistilBERTNewsClassifier(num_labels=10)
        
        assert hasattr(model, 'distilbert')
        assert hasattr(model, 'classifier')
    
    def test_forward_pass(self):
        """Test DistilBERT forward pass."""
        model = DistilBERTNewsClassifier(num_labels=10, use_snippet=False)
        
        batch_size = 2
        title_input_ids = torch.randint(0, 1000, (batch_size, 20))
        title_attention_mask = torch.ones(batch_size, 20)
        
        output = model(
            title_input_ids=title_input_ids,
            title_attention_mask=title_attention_mask
        )
        
        assert output.shape == (batch_size, 10)


class TestMultiHeadAttentionClassifier:
    """Tests for multi-head attention classifier."""
    
    def test_initialization(self):
        """Test multi-head attention initialization."""
        model = MultiHeadAttentionClassifier(num_labels=10)
        
        assert hasattr(model, 'bert')
        assert hasattr(model, 'attention_pooling')
        assert hasattr(model, 'classifier')
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = MultiHeadAttentionClassifier(num_labels=10, use_snippet=False)
        
        batch_size = 2
        title_input_ids = torch.randint(0, 1000, (batch_size, 20))
        title_attention_mask = torch.ones(batch_size, 20)
        
        output = model(
            title_input_ids=title_input_ids,
            title_attention_mask=title_attention_mask
        )
        
        assert output.shape == (batch_size, 10)
    
    def test_forward_with_attention(self):
        """Test forward pass with attention weights."""
        model = MultiHeadAttentionClassifier(num_labels=10, use_snippet=False)
        
        batch_size = 2
        title_input_ids = torch.randint(0, 1000, (batch_size, 20))
        title_attention_mask = torch.ones(batch_size, 20)
        
        output, attention = model(
            title_input_ids=title_input_ids,
            title_attention_mask=title_attention_mask,
            return_attention=True
        )
        
        assert output.shape == (batch_size, 10)
        assert 'title' in attention


class TestEnsembleModels:
    """Tests for ensemble models."""
    
    def test_weighted_ensemble(self):
        """Test weighted ensemble."""
        # Create dummy models
        class DummyModel(torch.nn.Module):
            def forward(self, title_input_ids, title_attention_mask):
                return torch.randn(title_input_ids.size(0), 10)
        
        models = [DummyModel() for _ in range(3)]
        ensemble = WeightedEnsemble(models)
        
        batch_size = 2
        title_input_ids = torch.randint(0, 1000, (batch_size, 20))
        title_attention_mask = torch.ones(batch_size, 20)
        
        output = ensemble(
            title_input_ids=title_input_ids,
            title_attention_mask=title_attention_mask
        )
        
        assert output.shape == (batch_size, 10)
    
    def test_voting_ensemble(self):
        """Test voting ensemble."""
        class DummyModel(torch.nn.Module):
            def forward(self, title_input_ids, title_attention_mask):
                return torch.randn(title_input_ids.size(0), 10)
        
        models = [DummyModel() for _ in range(3)]
        ensemble = VotingEnsemble(models, voting_type="soft")
        
        batch_size = 2
        title_input_ids = torch.randint(0, 1000, (batch_size, 20))
        title_attention_mask = torch.ones(batch_size, 20)
        
        output = ensemble(
            title_input_ids=title_input_ids,
            title_attention_mask=title_attention_mask
        )
        
        assert output.shape == (batch_size, 10)

