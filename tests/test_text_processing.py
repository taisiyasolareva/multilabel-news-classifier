"""Tests for text processing utilities."""

import pytest
from utils.text_processing import normalise_text, create_vocab


def test_normalise_text():
    """Test text normalization."""
    # Test basic normalization
    assert normalise_text("Привет, мир!") == "привет мир"
    assert normalise_text("Hello, World!") == "hello world"
    assert normalise_text("Test123") == "test123"
    
    # Test with special characters
    assert normalise_text("Text@#$%^&*()") == "text"
    
    # Test empty string
    assert normalise_text("") == ""
    
    # Test whitespace handling
    assert normalise_text("  test  ") == "test"


def test_create_vocab():
    """Test vocabulary creation."""
    text = "word1 word2 word1 word3"
    vocab = create_vocab(text, vocab_size=10)
    
    # Check special tokens
    assert "#PAD#" in vocab
    assert "#UNKN#" in vocab
    assert vocab["#PAD#"] == 0
    assert vocab["#UNKN#"] == 1
    
    # Check words are included
    assert "word1" in vocab
    assert "word2" in vocab
    assert "word3" in vocab
    
    # Check vocab size limit
    vocab_limited = create_vocab(text, vocab_size=2)
    # Should have 2 special tokens + 2 words = 4 total
    assert len(vocab_limited) <= 4

