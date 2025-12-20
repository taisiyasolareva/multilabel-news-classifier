#!/usr/bin/env python3
"""Evaluate a saved `.pt` checkpoint on the validation split (optionally using a frozen protocol).

Outputs:
- Predictions CSV (for Streamlit Evaluation dashboard): columns `sample_id`, `class_0..`, `target_class_0..`
- Metrics JSON (for model zoo + dashboards), including optional optimized global threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.data_loader import load_data, split_data
from data.transformer_dataset import TransformerNewsDataset
from models.transformer_model import RussianNewsClassifier
from utils.data_processing import create_target_encoding, process_tags
from utils.text_processing import normalise_text
from utils.tokenization import create_tokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _metrics_from_binary(target: torch.Tensor, pred: torch.Tensor) -> dict[str, float]:
    """
    Compute the same family of metrics used in existing `experiments/results/*.json`.
    - precision/recall/f1 are averaged per-sample (like `evaluation.metrics`)
    - exact_match is elementwise accuracy across all labels
    - subset_accuracy is strict set match per sample
    - micro_* are computed globally across all labels
    """
    target = target.float()
    pred = pred.float()

    # Per-sample precision/recall
    tp_per = ((pred == 1) & (target == 1)).sum(dim=1).float()
    pred_pos_per = (pred == 1).sum(dim=1).float()
    true_pos_per = (target == 1).sum(dim=1).float()

    precision = (tp_per / (pred_pos_per + 1e-5)).mean().item()
    recall = (tp_per / (true_pos_per + 1e-5)).mean().item()
    f1 = (2 * precision * recall) / (precision + recall + 1e-5)

    exact_match = (pred == target).float().mean().item()
    subset_accuracy = (pred == target).all(dim=1).float().mean().item()

    tp = ((pred == 1) & (target == 1)).sum().float()
    fp = ((pred == 1) & (target == 0)).sum().float()
    fn = ((pred == 0) & (target == 1)).sum().float()

    micro_precision = (tp / (tp + fp + 1e-5)).item()
    micro_recall = (tp / (tp + fn + 1e-5)).item()
    micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-5)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "exact_match": float(exact_match),
        "subset_accuracy": float(subset_accuracy),
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
    }


@torch.inference_mode()
def _predict_probs(
    *,
    model: RussianNewsClassifier,
    dataset: TransformerNewsDataset,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Return (probs, targets, sample_ids)."""
    model.eval()
    model.to(device)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    probs_list: list[torch.Tensor] = []
    targets_list: list[torch.Tensor] = []
    sample_ids: list[str] = []

    # sample_id preference: href if present, else dataframe index
    if "href" in dataset.df.columns:
        ids = dataset.df["href"].astype(str).tolist()
    else:
        ids = dataset.df.index.astype(str).tolist()

    offset = 0
    for batch in loader:
        bsz = batch["labels"].shape[0]
        sample_ids.extend(ids[offset : offset + bsz])
        offset += bsz

        batch_device: dict[str, torch.Tensor] = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch_device[k] = v.to(device)
        logits = model(
            title_input_ids=batch_device["title_input_ids"],
            title_attention_mask=batch_device["title_attention_mask"],
            snippet_input_ids=batch_device.get("snippet_input_ids"),
            snippet_attention_mask=batch_device.get("snippet_attention_mask"),
        )
        probs = torch.sigmoid(logits).detach().cpu()
        probs_list.append(probs)
        targets_list.append(batch["labels"].detach().cpu())

    probs_all = torch.cat(probs_list, dim=0) if probs_list else torch.empty((0, 0))
    targets_all = torch.cat(targets_list, dim=0) if targets_list else torch.empty((0, 0))
    return probs_all, targets_all, sample_ids


def _optimize_threshold(
    *,
    probs: torch.Tensor,
    target: torch.Tensor,
    metric: str,
    min_t: float = 0.01,
    max_t: float = 0.99,
    step: float = 0.01,
) -> tuple[float, dict[str, float]]:
    if probs.numel() == 0:
        return 0.5, _metrics_from_binary(target, probs)

    if metric not in {"precision", "recall", "f1"}:
        raise ValueError(f"Unknown optimize metric: {metric}")

    best_t = 0.5
    best_val = -1.0
    best_metrics: dict[str, float] = {}

    t = min_t
    while t <= max_t + 1e-9:
        pred = (probs >= t).float()
        m = _metrics_from_binary(target, pred)
        score = m[metric]
        if score > best_val:
            best_val = score
            best_t = float(round(t, 2))
            best_metrics = m
        t = round(t + step, 10)

    return best_t, best_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained model checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to saved `.pt` checkpoint")
    parser.add_argument("--data-path", type=str, default="data/news_data/ria_news.tsv", help="Path to RIA TSV")
    parser.add_argument("--protocol-dir", type=str, default=None, help="Frozen protocol directory (splits.json + tag_to_idx.json)")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Limit validation samples (ignored if protocol-dir is set)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Default global threshold for reporting `metrics`")

    parser.add_argument("--optimize-threshold", action="store_true", help="Search for best global threshold on val set")
    parser.add_argument(
        "--optimize-metric",
        type=str,
        default="f1",
        choices=["precision", "recall", "f1"],
        help="Metric to optimize when --optimize-threshold is set",
    )

    parser.add_argument("--batch-size", type=int, default=16, help="Eval batch size")
    parser.add_argument("--model-id", type=str, default=None, help="Optional model identifier (defaults to checkpoint stem)")
    parser.add_argument("--output-csv", type=str, default=None, help="Write predictions CSV to this path")
    parser.add_argument("--metrics-json", type=str, default=None, help="Write metrics JSON to this path")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint: dict[str, Any] = torch.load(ckpt_path, map_location="cpu")
    tag_to_idx = checkpoint.get("tag_to_idx") or {}
    num_labels = int(checkpoint.get("num_labels") or len(tag_to_idx))
    model_name = checkpoint.get("model_name") or "DeepPavlov/rubert-base-cased"
    use_snippet = bool(checkpoint.get("use_snippet", False))

    model_id = args.model_id or ckpt_path.stem

    logger.info(f"Loading data from {args.data_path}...")
    df_ria, _, _ = load_data(args.data_path)

    logger.info("Processing text...")
    df_ria["title_clean"] = df_ria["title"].apply(normalise_text)
    if "snippet" in df_ria.columns:
        df_ria["snippet_clean"] = df_ria["snippet"].fillna("").apply(normalise_text)

    logger.info("Processing tags...")
    df_ria["tags"] = process_tags(df_ria["tags"])

    logger.info("Splitting data...")
    df_train, df_val, df_test = split_data(
        df_ria,
        train_date_end="2018-10-01",
        val_date_start="2018-10-01",
        val_date_end="2018-12-01",
        test_date_start="2018-12-01",
    )

    protocol_meta: dict[str, Any] | None = None
    if args.protocol_dir:
        protocol_path = Path(args.protocol_dir)
        splits_path = protocol_path / "splits.json"
        mapping_path = protocol_path / "tag_to_idx.json"
        if not splits_path.exists() or not mapping_path.exists():
            raise FileNotFoundError(f"protocol-dir must contain splits.json and tag_to_idx.json: {protocol_path}")

        splits = json.loads(splits_path.read_text(encoding="utf-8"))
        id_col = splits.get("id_column", "href")
        if id_col == "href" and "href" in df_val.columns:
            df_train = df_train[df_train["href"].astype(str).isin(set(splits["train_ids"]))].copy()
            df_val = df_val[df_val["href"].astype(str).isin(set(splits["val_ids"]))].copy()
            df_test = df_test[df_test["href"].astype(str).isin(set(splits["test_ids"]))].copy()
        else:
            train_ids = set(splits["train_ids"])
            val_ids = set(splits["val_ids"])
            test_ids = set(splits["test_ids"])
            df_train = df_train[df_train.index.astype(str).isin(train_ids)].copy()
            df_val = df_val[df_val.index.astype(str).isin(val_ids)].copy()
            df_test = df_test[df_test.index.astype(str).isin(test_ids)].copy()

        tag_to_idx = json.loads(mapping_path.read_text(encoding="utf-8"))
        num_labels = len(tag_to_idx)
        logger.info(
            f"Loaded protocol bundle from {protocol_path} "
            f"(train={len(df_train)}, val={len(df_val)}, test={len(df_test)}, labels={num_labels})"
        )

        protocol_meta = {
            "data_path": args.data_path,
            "data_sha256": _file_sha256(args.data_path),
            "split": {
                "train_date_end": "2018-10-01",
                "val_date_start": "2018-10-01",
                "val_date_end": "2018-12-01",
                "test_date_start": "2018-12-01",
            },
            "limits": {
                "max_train_samples": len(df_train),
                "max_val_samples": len(df_val),
            },
            "label_space": {
                "min_tag_frequency": None,
                "num_labels": num_labels,
            },
        }

    else:
        if args.max_val_samples is not None:
            df_val = df_val.head(args.max_val_samples).copy()

    logger.info(f"Val samples: {len(df_val)}")

    # Encode targets for val set using tag_to_idx
    df_val = df_val.copy()
    df_val["target_tags"] = create_target_encoding(df_val, tag_to_idx)

    tokenizer = create_tokenizer(model_name, max_length=128)
    val_dataset = TransformerNewsDataset(
        df=df_val,
        tokenizer=tokenizer,
        max_title_len=128,
        max_snippet_len=256 if use_snippet else None,
        label_to_idx=tag_to_idx,
    )

    model = RussianNewsClassifier(
        model_name=model_name,
        num_labels=num_labels,
        dropout=float(checkpoint.get("dropout", 0.3)),
        use_snippet=use_snippet,
        freeze_bert=bool(checkpoint.get("freeze_backbone", False)),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)

    device = _pick_device()
    logger.info(f"Evaluating on device: {device}")

    probs, target, sample_ids = _predict_probs(model=model, dataset=val_dataset, batch_size=args.batch_size, device=device)

    # Save predictions CSV for Streamlit dashboards
    if args.output_csv:
        out_csv = Path(args.output_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {"sample_id": sample_ids}
        for j in range(probs.shape[1]):
            data[f"class_{j}"] = probs[:, j].numpy()
        for j in range(target.shape[1]):
            data[f"target_class_{j}"] = target[:, j].numpy()
        pd.DataFrame(data).to_csv(out_csv, index=False)
        logger.info(f"Wrote predictions CSV: {out_csv}")

    # Metrics at requested threshold
    pred_default = (probs >= float(args.threshold)).float()
    metrics_default = _metrics_from_binary(target, pred_default)

    # Sanity stats
    sanity = {
        "avg_true_labels_per_sample": float(target.sum(dim=1).float().mean().item()),
        "avg_pred_labels_per_sample": float(pred_default.sum(dim=1).float().mean().item()),
        "pct_samples_with_any_true_label": float((target.sum(dim=1) > 0).float().mean().item()),
        "pct_samples_with_any_pred_label": float((pred_default.sum(dim=1) > 0).float().mean().item()),
        "prob_min": float(probs.min().item()) if probs.numel() else 0.0,
        "prob_mean": float(probs.mean().item()) if probs.numel() else 0.0,
        "prob_max": float(probs.max().item()) if probs.numel() else 0.0,
    }

    payload: dict[str, Any] = {
        "experiment_id": model_id,
        "checkpoint_path": str(args.checkpoint),
        "data_path": args.data_path,
        "protocol_dir": args.protocol_dir,
        "protocol": protocol_meta,
        "threshold": float(args.threshold),
        "max_val_samples": args.max_val_samples,
        "val_samples": int(target.shape[0]),
        "num_labels": int(target.shape[1]),
        "model_name": model_name,
        "use_snippet": bool(use_snippet),
        "metrics": metrics_default,
        "sanity": sanity,
    }

    if args.optimize_threshold:
        best_t, best_metrics = _optimize_threshold(
            probs=probs,
            target=target,
            metric=args.optimize_metric,
            min_t=0.01,
            max_t=0.99,
            step=0.01,
        )
        payload["optimized_threshold"] = {
            "threshold": float(best_t),
            "metric": args.optimize_metric,
            "metric_value": float(best_metrics[args.optimize_metric]),
            **best_metrics,
        }

    if args.metrics_json:
        out_json = Path(args.metrics_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"Wrote metrics JSON: {out_json}")

    # Print a short summary for terminals
    logger.info(f"Metrics @ threshold={args.threshold}: f1={metrics_default['f1']:.4f}, p={metrics_default['precision']:.4f}, r={metrics_default['recall']:.4f}")
    if args.optimize_threshold:
        opt = payload["optimized_threshold"]
        logger.info(
            f"Optimized threshold={opt['threshold']:.2f} ({opt['metric']}={opt['metric_value']:.4f}) "
            f"f1={opt['f1']:.4f}, p={opt['precision']:.4f}, r={opt['recall']:.4f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())




