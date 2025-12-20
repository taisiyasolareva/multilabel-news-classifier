"""Russian text preprocessing helpers.

Keep this module lightweight: it is imported by the FastAPI service at startup.
"""

from __future__ import annotations

import re
from typing import Optional

_WS_RE = re.compile(r"\s+")


def prepare_text_for_tokenization(text: Optional[str]) -> str:
    """Prepare raw text for tokenizer input.

    - Handles None safely
    - Strips surrounding whitespace
    - Collapses internal whitespace/newlines
    """
    if text is None:
        return ""
    # Normalize whitespace and strip.
    s = _WS_RE.sub(" ", str(text)).strip()
    return s


