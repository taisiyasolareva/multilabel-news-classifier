"""Label/tag processing helpers for multi-label classification."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple, Union

import torch


def process_tags(tags: Union[str, Sequence[str], None], sep: str = ",") -> List[str]:
    """Convert raw tags to a list of normalized tag strings."""
    if tags is None:
        return []
    if isinstance(tags, str):
        parts = [t.strip() for t in tags.split(sep)]
        return [p for p in parts if p]
    # Sequence[str]
    out: List[str] = []
    for t in tags:
        if t is None:
            continue
        s = str(t).strip()
        if s:
            out.append(s)
    return out


def build_label_mapping(
    df,
    *,
    tags_col: str = "tags",
    sep: str = ",",
) -> Dict[str, int]:
    """Build a tag->index mapping from a dataframe-like object.

    Expects `df[tags_col]` to contain either comma-separated strings or lists.
    """
    tag_set = set()
    for raw in df[tags_col].tolist():
        tag_set.update(process_tags(raw, sep=sep))
    return {tag: i for i, tag in enumerate(sorted(tag_set))}


def create_target_encoding(
    tag_lists: Iterable[Union[str, Sequence[str], None]],
    label_to_idx: Dict[str, int],
    *,
    sep: str = ",",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a multi-hot target tensor of shape [N, num_labels]."""
    tag_lists = list(tag_lists)
    num_labels = len(label_to_idx)
    y = torch.zeros((len(tag_lists), num_labels), dtype=dtype)
    for i, raw in enumerate(tag_lists):
        for tag in process_tags(raw, sep=sep):
            j = label_to_idx.get(tag)
            if j is not None:
                y[i, j] = 1.0
    return y


