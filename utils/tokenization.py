"""Tokenization utilities used for transformer models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from transformers import AutoTokenizer


@dataclass
class RussianTextTokenizer:
    """Thin wrapper around a HuggingFace tokenizer with sane defaults."""

    model_name: str = "DeepPavlov/rubert-base-cased"
    max_length: int = 128
    padding: Union[bool, str] = "max_length"
    truncation: bool = True

    def __post_init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

    def get_vocab_size(self) -> int:
        return int(getattr(self.tokenizer, "vocab_size", len(self.tokenizer.get_vocab())))

    def get_special_tokens(self) -> Dict[str, Optional[int]]:
        return {
            "pad_token_id": self.tokenizer.pad_token_id,
            "cls_token_id": self.tokenizer.cls_token_id,
            "sep_token_id": self.tokenizer.sep_token_id,
            "unk_token_id": self.tokenizer.unk_token_id,
        }

    def tokenize(self, text: str, add_special_tokens: bool = True) -> List[str]:
        return self.tokenizer.tokenize(text or "", add_special_tokens=add_special_tokens)

    def encode(
        self,
        text: str,
        *,
        max_length: Optional[int] = None,
        padding: Optional[Union[bool, str]] = None,
        truncation: Optional[bool] = None,
        return_tensors: Optional[str] = "pt",
    ) -> Dict[str, Any]:
        """Encode a single text.

        Returns a dict containing `input_ids` and `attention_mask`.
        """
        max_length_eff = max_length or self.max_length
        padding_eff = self.padding if padding is None else padding
        truncation_eff = self.truncation if truncation is None else truncation

        if return_tensors is None:
            enc = self.tokenizer(
                text or "",
                max_length=max_length_eff,
                padding=padding_eff,
                truncation=truncation_eff,
                return_attention_mask=True,
                return_tensors=None,
            )
            # HuggingFace returns lists for a single example; standardize to batch-like shape.
            return {
                "input_ids": [enc["input_ids"]],
                "attention_mask": [enc["attention_mask"]],
            }

        return self.tokenizer(
            text or "",
            max_length=max_length_eff,
            padding=padding_eff,
            truncation=truncation_eff,
            return_attention_mask=True,
            return_tensors=return_tensors,
        )

    def encode_batch(
        self,
        texts: List[str],
        *,
        max_length: Optional[int] = None,
        padding: Optional[Union[bool, str]] = None,
        truncation: Optional[bool] = None,
        return_tensors: str = "pt",
    ) -> Dict[str, Any]:
        max_length_eff = max_length or self.max_length
        padding_eff = self.padding if padding is None else padding
        truncation_eff = self.truncation if truncation is None else truncation
        return self.tokenizer(
            [t or "" for t in texts],
            max_length=max_length_eff,
            padding=padding_eff,
            truncation=truncation_eff,
            return_attention_mask=True,
            return_tensors=return_tensors,
        )

    def decode(self, token_ids: Union[List[int], Any], skip_special_tokens: bool = True) -> str:
        # Avoid importing torch at module import time; handle torch tensors via duck-typing.
        if hasattr(token_ids, "detach") and hasattr(token_ids, "cpu") and hasattr(token_ids, "tolist"):
            token_ids = token_ids.detach().cpu().tolist()
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def get_token_info(self, token_id: int) -> Dict[str, Any]:
        tok = self.tokenizer.convert_ids_to_tokens(int(token_id))
        specials = set(self.tokenizer.all_special_ids)
        return {
            "token_id": int(token_id),
            "token": tok,
            "is_special": int(token_id) in specials,
        }


def create_tokenizer(model_name: str = "DeepPavlov/rubert-base-cased", max_length: int = 128) -> RussianTextTokenizer:
    return RussianTextTokenizer(model_name=model_name, max_length=max_length)


def tokenize_text_pair(
    *,
    title: str,
    snippet: Optional[str],
    tokenizer: RussianTextTokenizer,
    max_title_len: int = 128,
    max_snippet_len: int = 256,
) -> Dict[str, Any]:
    """Tokenize (title, snippet) as two independent sequences (not a single pair encoding)."""
    title_enc = tokenizer.encode(title or "", max_length=max_title_len, return_tensors="pt")
    out: Dict[str, Any] = {
        "title_input_ids": title_enc["input_ids"].squeeze(0),
        "title_attention_mask": title_enc["attention_mask"].squeeze(0),
    }

    if snippet is not None:
        snip_enc = tokenizer.encode(snippet or "", max_length=max_snippet_len, return_tensors="pt")
        out.update(
            {
                "snippet_input_ids": snip_enc["input_ids"].squeeze(0),
                "snippet_attention_mask": snip_enc["attention_mask"].squeeze(0),
            }
        )

    return out


