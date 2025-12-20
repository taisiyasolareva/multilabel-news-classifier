"""Basic text normalization and vocabulary building utilities."""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict

# Keep letters (Latin + Cyrillic), digits, and whitespace.
_CLEAN_RE = re.compile(r"[^0-9a-zA-Z\u0400-\u04FF\s]+", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")


def normalise_text(text: str) -> str:
    """Lowercase, remove punctuation/special chars, and collapse whitespace."""
    s = (text or "").lower()
    s = _CLEAN_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def create_vocab(text: str, vocab_size: int = 50000) -> Dict[str, int]:
    """Create a simple frequency-based vocabulary mapping.

    Always includes:
    - #PAD# -> 0
    - #UNKN# -> 1
    """
    vocab: Dict[str, int] = {"#PAD#": 0, "#UNKN#": 1}
    if vocab_size <= 0:
        return vocab

    tokens = normalise_text(text).split()
    counts = Counter(tokens)

    for word, _ in counts.most_common(max(0, vocab_size)):
        if word in vocab:
            continue
        vocab[word] = len(vocab)

    return vocab


