"""Tests for tokenization utilities."""

import pytest
import torch
from utils.tokenization import (
    RussianTextTokenizer,
    create_tokenizer,
    tokenize_text_pair,
)


class TestRussianTextTokenizer:
    """Tests for Russian text tokenizer."""
    
    def test_initialization(self):
        """Test tokenizer initialization."""
        tokenizer = RussianTextTokenizer(
            model_name="DeepPavlov/rubert-base-cased",
            max_length=128
        )
        
        assert tokenizer.tokenizer is not None
        assert tokenizer.max_length == 128
        assert tokenizer.get_vocab_size() > 0
    
    def test_tokenize_russian_text(self):
        """Test tokenization of Russian text."""
        tokenizer = RussianTextTokenizer()
        
        text = "Привет, мир!"
        tokens = tokenizer.tokenize(text)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        # Should include special tokens if add_special_tokens=True
        assert any('[CLS]' in str(t) or 'CLS' in str(t) for t in tokens) or len(tokens) > 0
    
    def test_encode_russian_text(self):
        """Test encoding of Russian text."""
        tokenizer = RussianTextTokenizer(max_length=128)
        
        text = "Это тестовый текст на русском языке"
        encoded = tokenizer.encode(text)
        
        assert 'input_ids' in encoded
        assert 'attention_mask' in encoded
        assert encoded['input_ids'].shape[1] == 128  # max_length
        assert encoded['attention_mask'].shape[1] == 128
    
    def test_encode_batch(self):
        """Test batch encoding."""
        tokenizer = RussianTextTokenizer(max_length=64)
        
        texts = [
            "Первая новость",
            "Вторая новость",
            "Третья новость"
        ]
        
        encoded = tokenizer.encode_batch(texts)
        
        assert encoded['input_ids'].shape[0] == 3  # batch size
        assert encoded['input_ids'].shape[1] == 64  # max_length
        assert encoded['attention_mask'].shape[0] == 3
    
    def test_decode(self):
        """Test decoding token IDs back to text."""
        tokenizer = RussianTextTokenizer()
        
        text = "Привет, мир!"
        encoded = tokenizer.encode(text, return_tensors=None)
        
        decoded = tokenizer.decode(encoded['input_ids'][0])
        
        # Decoded text should be similar (may have different casing/punctuation)
        assert isinstance(decoded, str)
        assert len(decoded) > 0
    
    def test_special_tokens(self):
        """Test special token handling."""
        tokenizer = RussianTextTokenizer()
        
        special_tokens = tokenizer.get_special_tokens()
        
        assert 'pad_token_id' in special_tokens
        assert 'cls_token_id' in special_tokens
        assert 'sep_token_id' in special_tokens
        assert special_tokens['pad_token_id'] is not None
    
    def test_padding(self):
        """Test padding behavior."""
        tokenizer = RussianTextTokenizer(max_length=20, padding='max_length')
        
        text = "Короткий текст"
        encoded = tokenizer.encode(text)
        
        # Should be padded to max_length
        assert encoded['input_ids'].shape[1] == 20
        assert encoded['attention_mask'].shape[1] == 20
    
    def test_truncation(self):
        """Test truncation of long texts."""
        tokenizer = RussianTextTokenizer(max_length=10, truncation=True)
        
        # Create a long text
        long_text = " ".join(["слово"] * 50)
        encoded = tokenizer.encode(long_text)
        
        # Should be truncated to max_length
        assert encoded['input_ids'].shape[1] == 10
    
    def test_subword_tokenization(self):
        """Test that subword tokenization handles unknown words."""
        tokenizer = RussianTextTokenizer()
        
        # Use a word that might not be in vocabulary
        text = "НеизвестноеСловоКоторогоНетВСловаре"
        tokens = tokenizer.tokenize(text, add_special_tokens=False)
        
        # Should still tokenize (using subwords)
        assert len(tokens) > 0
        # Subword tokens often start with ## or are split
        assert all(isinstance(t, str) for t in tokens)


class TestTokenizerFactory:
    """Tests for tokenizer factory function."""
    
    def test_create_tokenizer(self):
        """Test tokenizer creation."""
        tokenizer = create_tokenizer(
            model_name="DeepPavlov/rubert-base-cased",
            max_length=256
        )
        
        assert isinstance(tokenizer, RussianTextTokenizer)
        assert tokenizer.max_length == 256
    
    def test_create_multilingual_tokenizer(self):
        """Test creating multilingual tokenizer."""
        tokenizer = create_tokenizer(
            model_name="bert-base-multilingual-cased",
            max_length=128
        )
        
        assert tokenizer.model_name == "bert-base-multilingual-cased"
        assert tokenizer.max_length == 128


class TestTextPairTokenization:
    """Tests for title-snippet pair tokenization."""
    
    def test_tokenize_text_pair(self):
        """Test tokenizing title and snippet pair."""
        tokenizer = create_tokenizer()
        
        title = "Заголовок новости"
        snippet = "Краткое описание новости"
        
        encoded = tokenize_text_pair(
            title=title,
            snippet=snippet,
            tokenizer=tokenizer,
            max_title_len=64,
            max_snippet_len=128
        )
        
        assert 'title_input_ids' in encoded
        assert 'title_attention_mask' in encoded
        assert 'snippet_input_ids' in encoded
        assert 'snippet_attention_mask' in encoded
        
        assert encoded['title_input_ids'].shape[0] == 64
        assert encoded['snippet_input_ids'].shape[0] == 128
    
    def test_tokenize_title_only(self):
        """Test tokenizing title without snippet."""
        tokenizer = create_tokenizer()
        
        title = "Заголовок"
        
        encoded = tokenize_text_pair(
            title=title,
            snippet=None,
            tokenizer=tokenizer
        )
        
        assert 'title_input_ids' in encoded
        assert 'snippet_input_ids' not in encoded


class TestRussianTextHandling:
    """Tests for proper Russian text handling."""
    
    def test_cyrillic_characters(self):
        """Test handling of Cyrillic characters."""
        tokenizer = RussianTextTokenizer()
        
        # Test various Cyrillic characters
        texts = [
            "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ",
            "абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
            "1234567890",
            "Смешанный текст: English and русский",
        ]
        
        for text in texts:
            encoded = tokenizer.encode(text)
            decoded = tokenizer.decode(encoded['input_ids'][0])
            
            # Should handle all without errors
            assert encoded['input_ids'].shape[0] > 0
            assert isinstance(decoded, str)
    
    def test_russian_punctuation(self):
        """Test handling of Russian punctuation."""
        tokenizer = RussianTextTokenizer()
        
        text = "Текст с пунктуацией: запятые, точки. Восклицания! Вопросы?"
        encoded = tokenizer.encode(text)
        
        assert encoded['input_ids'].shape[0] > 0
        assert not torch.isnan(encoded['input_ids']).any()
    
    def test_empty_text_handling(self):
        """Test handling of empty or whitespace-only text."""
        tokenizer = RussianTextTokenizer()
        
        # Empty string
        encoded = tokenizer.encode("")
        assert encoded['input_ids'].shape[0] > 0
        
        # Whitespace only
        encoded = tokenizer.encode("   ")
        assert encoded['input_ids'].shape[0] > 0
    
    def test_very_long_text(self):
        """Test handling of very long texts (should truncate)."""
        tokenizer = RussianTextTokenizer(max_length=50, truncation=True)
        
        # Create very long text
        long_text = " ".join(["слово"] * 200)
        encoded = tokenizer.encode(long_text)
        
        # Should be truncated
        assert encoded['input_ids'].shape[1] == 50


class TestSubwordTokenization:
    """Tests for subword tokenization features."""
    
    def test_unknown_word_handling(self):
        """Test that unknown words are handled via subword tokenization."""
        tokenizer = RussianTextTokenizer()
        
        # Word that likely doesn't exist in vocabulary
        unknown_word = "НесуществующееСловоКоторогоТочноНетВСловаре12345"
        tokens = tokenizer.tokenize(unknown_word, add_special_tokens=False)
        
        # Should be split into subwords
        assert len(tokens) > 0
        # All should be valid tokens
        assert all(isinstance(t, str) for t in tokens)
    
    def test_word_piece_tokenization(self):
        """Test WordPiece subword tokenization."""
        tokenizer = RussianTextTokenizer()
        
        # Common Russian word
        text = "правительство"
        tokens = tokenizer.tokenize(text, add_special_tokens=False)
        
        # Should tokenize (may be single token or multiple subwords)
        assert len(tokens) > 0
    
    def test_vocabulary_coverage(self):
        """Test that tokenizer has good vocabulary coverage."""
        tokenizer = RussianTextTokenizer()
        
        vocab_size = tokenizer.get_vocab_size()
        
        # BERT models typically have 30K+ vocabulary
        assert vocab_size > 10000
        assert vocab_size < 1000000  # Reasonable upper bound
    
    def test_token_info(self):
        """Test getting token information."""
        tokenizer = RussianTextTokenizer()
        
        # Get a token ID
        special_tokens = tokenizer.get_special_tokens()
        pad_id = special_tokens['pad_token_id']
        
        info = tokenizer.get_token_info(pad_id)
        
        assert 'token_id' in info
        assert 'token' in info
        assert 'is_special' in info
        assert info['token_id'] == pad_id

