"""Pytest configuration and fixtures."""

import pytest
import torch
import pandas as pd
import tempfile
import os


@pytest.fixture
def sample_vocab():
    """Create a sample vocabulary for testing."""
    return {
        "#PAD#": 0,
        "#UNKN#": 1,
        "word1": 2,
        "word2": 3,
        "word3": 4,
        "test": 5,
    }


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'href': [f'href_{i}' for i in range(10)],
        'title': [f'Title {i} with words' for i in range(10)],
        'tags': ['tag1,tag2'] * 5 + ['tag2,tag3'] * 5,
        'date': pd.date_range('2018-01-01', periods=10, freq='D')
    })


@pytest.fixture
def sample_tag_mapping():
    """Create a sample tag-to-index mapping."""
    return {
        "tag1": 0,
        "tag2": 1,
        "tag3": 2,
    }


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducibility in tests."""
    torch.manual_seed(42)
    import numpy as np
    np.random.seed(42)

