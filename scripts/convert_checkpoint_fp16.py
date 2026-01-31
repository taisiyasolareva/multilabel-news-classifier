"""
Convert a PyTorch checkpoint to a smaller fp16 "inference checkpoint".

Why:
- Render free tier gives you ~512Mi RAM.
- A float32 transformer checkpoint can exceed that just from weights.
- Saving only the inference state_dict (and converting to fp16) can cut size ~2x.

Usage:
  python scripts/convert_checkpoint_fp16.py \
    --input models/distilmbert_lora_10k_v1.pt \
    --output models/distilmbert_lora_10k_v1_fp16.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch


def _to_fp16_state_dict(sd: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in sd.items():
        if torch.is_tensor(v) and v.is_floating_point():
            out[k] = v.half()
        else:
            out[k] = v
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output", required=True, help="Path to write fp16 inference checkpoint")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(str(in_path), map_location="cpu")

    payload: Dict[str, Any] = {}

    # Preserve useful metadata if present
    if isinstance(ckpt, dict):
        for k in ("model_name", "num_labels", "use_snippet", "tag_to_idx"):
            if k in ckpt:
                payload[k] = ckpt[k]

        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            payload["state_dict"] = _to_fp16_state_dict(ckpt["state_dict"])
        elif "model" in ckpt and hasattr(ckpt["model"], "state_dict"):
            payload["state_dict"] = _to_fp16_state_dict(ckpt["model"].state_dict())
        else:
            # Best effort: if it looks like a raw state_dict, convert it.
            if all(isinstance(k, str) for k in ckpt.keys()):
                payload["state_dict"] = _to_fp16_state_dict(ckpt)  # type: ignore[arg-type]
            else:
                raise ValueError("Unsupported checkpoint format: expected dict with state_dict/model")
    else:
        # If the checkpoint is a full nn.Module, extract its state_dict.
        if hasattr(ckpt, "state_dict"):
            payload["state_dict"] = _to_fp16_state_dict(ckpt.state_dict())
        else:
            raise ValueError("Unsupported checkpoint type (expected dict or nn.Module)")

    torch.save(payload, str(out_path))

    in_mb = in_path.stat().st_size / (1024 * 1024)
    out_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Wrote fp16 inference checkpoint:\n  in:  {in_path} ({in_mb:.1f} MB)\n  out: {out_path} ({out_mb:.1f} MB)")


if __name__ == "__main__":
    main()






