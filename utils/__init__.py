"""Utility helpers used across API, training scripts, and dashboards.

Important: keep this package lightweight at import time. In production, we want
`uvicorn api.main:app` to import quickly and bind to the port; heavy deps like
transformers/torch should only be imported when actually needed.

This package also prevents ambiguous imports where `import utils` could resolve
to an unrelated third-party PyPI package named `utils`.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "RussianTextTokenizer",
    "create_tokenizer",
    "tokenize_text_pair",
    "prepare_text_for_tokenization",
    "normalise_text",
    "create_vocab",
    "process_tags",
    "build_label_mapping",
    "create_target_encoding",
]

_LAZY: dict[str, tuple[str, str]] = {
    "RussianTextTokenizer": ("utils.tokenization", "RussianTextTokenizer"),
    "create_tokenizer": ("utils.tokenization", "create_tokenizer"),
    "tokenize_text_pair": ("utils.tokenization", "tokenize_text_pair"),
    "prepare_text_for_tokenization": ("utils.russian_text_utils", "prepare_text_for_tokenization"),
    "normalise_text": ("utils.text_processing", "normalise_text"),
    "create_vocab": ("utils.text_processing", "create_vocab"),
    "process_tags": ("utils.data_processing", "process_tags"),
    "build_label_mapping": ("utils.data_processing", "build_label_mapping"),
    "create_target_encoding": ("utils.data_processing", "create_target_encoding"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module 'utils' has no attribute {name!r}")
    module_name, attr_name = _LAZY[name]
    mod = import_module(module_name)
    return getattr(mod, attr_name)


