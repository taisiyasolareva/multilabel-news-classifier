"""
Download a model checkpoint to a local path.

Primary use-case: cloud deployment (Render/Railway/Fly) where you don't want to
commit large `.pt` files into git history.

Typical usage (GitHub Release asset):
  python scripts/download_model.py \
    --model-id distilmbert_lora_10k_v1 \
    --url "https://github.com/<user>/<repo>/releases/download/<tag>/distilmbert_lora_10k_v1.pt" \
    --output-path models/distilmbert_lora_10k_v1.pt

Optional: resolve URL from `config/models.yaml` when `--url` is omitted.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Optional

import requests
import yaml


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _resolve_from_registry(model_id: str, registry_path: Path) -> Optional[dict]:
    if not registry_path.exists():
        return None
    data = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return None
    models = data.get("models")
    if not isinstance(models, dict):
        return None
    entry = models.get(model_id)
    return entry if isinstance(entry, dict) else None


def download(url: str, output_path: Path, expected_sha256: Optional[str], force: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        if expected_sha256:
            got = _sha256(output_path)
            if got.lower() == expected_sha256.lower():
                print(f"[download_model] OK (already present): {output_path} sha256={got}")
                return
            raise SystemExit(
                f"[download_model] Refusing to overwrite existing file with mismatched sha256.\n"
                f"  path: {output_path}\n"
                f"  expected: {expected_sha256}\n"
                f"  got: {got}\n"
                f"Use --force to overwrite."
            )
        print(f"[download_model] OK (already present): {output_path}")
        return

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    print(f"[download_model] Downloading:\n  url: {url}\n  to:  {output_path}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length") or 0)
        downloaded = 0
        with tmp_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = 100.0 * downloaded / total
                    print(f"\r[download_model] {pct:5.1f}% ({downloaded/1024/1024:.1f}MB/{total/1024/1024:.1f}MB)", end="")
    print()

    got = _sha256(tmp_path)
    if expected_sha256 and got.lower() != expected_sha256.lower():
        tmp_path.unlink(missing_ok=True)
        raise SystemExit(
            f"[download_model] SHA256 mismatch.\n"
            f"  expected: {expected_sha256}\n"
            f"  got:      {got}\n"
            f"  url:      {url}"
        )

    tmp_path.replace(output_path)
    print(f"[download_model] DONE: {output_path} sha256={got}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True, help="Model id, e.g. distilmbert_lora_10k_v1")
    ap.add_argument("--url", default=None, help="Direct URL to .pt file (overrides registry/env)")
    ap.add_argument(
        "--output-path",
        default=None,
        help="Output checkpoint path. Default: models/<model-id>.pt",
    )
    ap.add_argument(
        "--registry-path",
        default="config/models.yaml",
        help="Optional model registry file (YAML) to resolve URL/sha256",
    )
    ap.add_argument("--sha256", default=None, help="Expected sha256 for integrity check (optional)")
    ap.add_argument("--force", action="store_true", help="Overwrite existing file")

    args = ap.parse_args()

    model_id: str = args.model_id
    output_path = Path(args.output_path) if args.output_path else Path("models") / f"{model_id}.pt"

    # Priority: CLI --url -> env MODEL_URL -> config/models.yaml
    url = args.url or os.environ.get("MODEL_URL")
    expected_sha256 = args.sha256 or os.environ.get("MODEL_SHA256")

    if not url:
        entry = _resolve_from_registry(model_id, Path(args.registry_path))
        if entry:
            url = entry.get("url") or entry.get("checkpoint_url")
            expected_sha256 = expected_sha256 or entry.get("sha256") or entry.get("checkpoint_sha256")

    if not url:
        raise SystemExit(
            "[download_model] No URL provided.\n"
            "Provide --url, or set MODEL_URL env var, or add config/models.yaml entry."
        )

    download(url=url, output_path=output_path, expected_sha256=expected_sha256, force=args.force)


if __name__ == "__main__":
    main()


