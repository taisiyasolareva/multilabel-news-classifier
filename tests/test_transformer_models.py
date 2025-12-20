"""Tests for transformer-based models."""

import pytest
import torch
from utils.tokenization import create_tokenizer
from models.transformer_model import RussianNewsClassifier, MultilingualBERTClassifier
from data.transformer_dataset import TransformerNewsDataset
import pandas as pd


class TestRussianNewsClassifier:
    """Tests for Russian BERT classifier."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = RussianNewsClassifier(
            model_name="DeepPavlov/rubert-base-cased",
            num_labels=10,
            use_snippet=False
        )
        
        assert hasattr(model, 'bert')
        assert hasattr(model, 'classifier')
        assert model.num_labels == 10
        assert model.use_snippet is False
    
    def test_forward_title_only(self):
        """Test forward pass with title only."""
        model = RussianNewsClassifier(
            model_name="DeepPavlov/rubert-base-cased",
            num_labels=10,
            use_snippet=False
        )
        
        batch_size = 2
        seq_len = 20
        title_input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        title_attention_mask = torch.ones(batch_size, seq_len)
        
        output = model(
            title_input_ids=title_input_ids,
            title_attention_mask=title_attention_mask
        )
        
        assert output.shape == (batch_size, 10)
        assert not torch.isnan(output).any()
    
    def test_forward_with_snippet(self):
        """Test forward pass with snippet."""
        model = RussianNewsClassifier(
            model_name="DeepPavlov/rubert-base-cased",
            num_labels=10,
            use_snippet=True
        )
        
        batch_size = 2
        title_len = 20
        snippet_len = 50
        
        title_input_ids = torch.randint(0, 1000, (batch_size, title_len))
        title_attention_mask = torch.ones(batch_size, title_len)
        snippet_input_ids = torch.randint(0, 1000, (batch_size, snippet_len))
        snippet_attention_mask = torch.ones(batch_size, snippet_len)
        
        output = model(
            title_input_ids=title_input_ids,
            title_attention_mask=title_attention_mask,
            snippet_input_ids=snippet_input_ids,
            snippet_attention_mask=snippet_attention_mask,
        )
        
        assert output.shape == (batch_size, 10)
        assert not torch.isnan(output).any()
    
    def test_freeze_bert(self):
        """Test that BERT weights can be frozen."""
        model = RussianNewsClassifier(
            model_name="DeepPavlov/rubert-base-cased",
            num_labels=10,
            freeze_bert=True
        )
        
        # Check that BERT parameters are frozen
        for param in model.bert.parameters():
            assert not param.requires_grad
        
        # Check that classifier parameters are trainable
        for param in model.classifier.parameters():
            assert param.requires_grad


class TestMultilingualBERTClassifier:
    """Tests for Multilingual BERT classifier."""
    
    def test_initialization(self):
        """Test multilingual BERT initialization."""
        model = MultilingualBERTClassifier(
            model_name="bert-base-multilingual-cased",
            num_labels=10
        )
        
        assert hasattr(model, 'base_model')
        assert model.base_model.num_labels == 10
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = MultilingualBERTClassifier(num_labels=10, use_snippet=False)
        
        batch_size = 2
        title_input_ids = torch.randint(0, 1000, (batch_size, 20))
        title_attention_mask = torch.ones(batch_size, 20)
        
        output = model(
            title_input_ids=title_input_ids,
            title_attention_mask=title_attention_mask
        )
        
        assert output.shape == (batch_size, 10)


class TestTransformerDataset:
    """Tests for transformer dataset."""
    
    def test_dataset_creation(self):
        """Test dataset creation with proper tokenization."""
        # Create sample data
        df = pd.DataFrame({
            'title_clean': ['тест заголовок', 'другой заголовок'],
            'target_tags': [[0, 1], [2]]
        })
        
        tokenizer = create_tokenizer("DeepPavlov/rubert-base-cased")
        tag_to_idx = {"tag1": 0, "tag2": 1, "tag3": 2}
        
        dataset = TransformerNewsDataset(
            df=df,
            tokenizer=tokenizer,
            max_title_len=20,
            label_to_idx=tag_to_idx
        )
        
        assert len(dataset) == 2
        assert not dataset.use_snippet
    
    def test_dataset_with_snippet(self):
        """Test dataset with snippets and subword tokenization."""
        df = pd.DataFrame({
            'title_clean': ['тест заголовок'],
            'snippet_clean': ['тест описание'],
            'target_tags': [[0]]
        })
        
        tokenizer = create_tokenizer("DeepPavlov/rubert-base-cased")
        tag_to_idx = {"tag1": 0}
        
        dataset = TransformerNewsDataset(
            df=df,
            tokenizer=tokenizer,
            max_title_len=20,
            max_snippet_len=50,
            label_to_idx=tag_to_idx
        )
        
        assert dataset.use_snippet
        
        # Test __getitem__
        sample = dataset[0]
        assert 'title_input_ids' in sample
        assert 'snippet_input_ids' in sample
        assert 'labels' in sample
        
        # Verify subword tokenization worked
        assert sample['title_input_ids'].shape[0] == 20
        assert sample['snippet_input_ids'].shape[0] == 50

