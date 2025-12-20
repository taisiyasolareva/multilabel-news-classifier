#!/usr/bin/env python3
"""Training script for the model zoo (protocol + LoRA + W&B logging).

This is the canonical entrypoint we use for fair comparisons:
- supports a frozen protocol bundle via --protocol-dir
- supports PEFT LoRA via --use-lora (adapters merged into base model before saving)
- supports W&B logging via --logger wandb (train/val loss are logged by Lightning)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

try:
    from pytorch_lightning.loggers import WandbLogger
except Exception:  # pragma: no cover
    WandbLogger = None

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.data_loader import load_data, split_data
from data.transformer_dataset import TransformerNewsDataset
from models.transformer_lightning import TransformerClassificationModule
from models.transformer_model import RussianNewsClassifier
from utils.data_processing import build_label_mapping, create_target_encoding, process_tags
from utils.text_processing import normalise_text
from utils.tokenization import create_tokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _parse_bool(x: str) -> bool:
    return str(x).lower() in ("true", "1", "yes", "y")


def _default_lora_target_modules(model_name: str) -> list[str]:
    """
    Pick sensible default target modules for common transformer backbones.
    - BERT/RoBERTa/XLM-R: attention projections are typically named query/key/value
    - DistilBERT: q_lin/k_lin/v_lin
    """
    mn = (model_name or "").lower()
    if "distilbert" in mn:
        return ["q_lin", "k_lin", "v_lin"]
    return ["query", "key", "value"]


def _apply_lora_to_backbone(
    model: RussianNewsClassifier,
    *,
    model_name: str,
    r: int,
    alpha: int,
    dropout: float,
    target_modules: list[str],
) -> dict:
    """Apply LoRA adapters to model.bert using PEFT."""
    try:
        from peft import LoraConfig, TaskType, get_peft_model  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "LoRA requested but `peft` is not installed. Install it with `pip install peft accelerate`."
        ) from e

    lora_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,  # we wrap AutoModel (not AutoModelForSequenceClassification)
        r=int(r),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        target_modules=target_modules,
        bias="none",
    )

    model.bert = get_peft_model(model.bert, lora_cfg)
    if hasattr(model.bert, "print_trainable_parameters"):
        model.bert.print_trainable_parameters()

    return {
        "enabled": True,
        "r": int(r),
        "alpha": int(alpha),
        "dropout": float(dropout),
        "target_modules": target_modules,
        "merged_into_base": True,  # we attempt to merge before saving final .pt
    }


def train_model(
    *,
    data_path: str = "data/news_data/ria_news.tsv",
    output_path: str = "models/best_model.pt",
    model_name: str = "DeepPavlov/rubert-base-cased",
    epochs: int = 3,
    batch_size: int = 16,
    accumulate_grad_batches: int = 4,
    learning_rate: float = 2e-5,
    use_snippet: bool = False,
    freeze_backbone: bool = False,
    max_title_len: int = 128,
    max_snippet_len: int = 256,
    min_tag_frequency: int = 30,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    num_workers: int = 0,
    protocol_dir: str | None = None,
    use_lora: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: str | None = None,
    logger_backend: str = "csv",
    wandb_project: str = "russian-news-classification",
    wandb_run_name: str | None = None,
    wandb_mode: str = "online",
) -> tuple[TransformerClassificationModule, dict]:
    """Train a transformer multi-label classifier and save a `.pt` checkpoint."""
    pl.seed_everything(42)

    output_path_p = Path(output_path)
    output_dir = output_path_p.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading data from {data_path}...")
    df_ria, _, _ = load_data(data_path)
    logger.info(f"Loaded {len(df_ria)} articles")

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

    tag_to_idx: dict | None = None
    if protocol_dir:
        protocol_path = Path(protocol_dir)
        splits_path = protocol_path / "splits.json"
        mapping_path = protocol_path / "tag_to_idx.json"
        if not splits_path.exists() or not mapping_path.exists():
            raise FileNotFoundError(f"protocol_dir must contain splits.json and tag_to_idx.json: {protocol_path}")

        splits = json.loads(splits_path.read_text(encoding="utf-8"))
        id_col = splits.get("id_column", "href")
        if id_col == "href" and "href" in df_train.columns:
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
        logger.info(
            f"Loaded protocol bundle from {protocol_path} "
            f"(train={len(df_train)}, val={len(df_val)}, test={len(df_test)}, labels={len(tag_to_idx)})"
        )

    if protocol_dir is None and max_train_samples is not None:
        df_train = df_train.head(max_train_samples).copy()
        logger.info(f"Limited training set to {max_train_samples} samples")
    if protocol_dir is None and max_val_samples is not None:
        df_val = df_val.head(max_val_samples).copy()
        logger.info(f"Limited validation set to {max_val_samples} samples")

    logger.info(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

    # Build mapping from TRAIN ONLY (avoid leakage + fair comparison).
    if tag_to_idx is None:
        logger.info("Building label mapping from TRAIN split only...")
        tag_to_idx = build_label_mapping(df_train, min_frequency=min_tag_frequency)
    num_labels = len(tag_to_idx)
    logger.info(f"Using {num_labels} labels (min_tag_frequency={min_tag_frequency})")

    # Encode targets for each split using the SAME mapping
    df_train = df_train.copy()
    df_val = df_val.copy()
    df_test = df_test.copy()
    df_train["target_tags"] = create_target_encoding(df_train, tag_to_idx)
    df_val["target_tags"] = create_target_encoding(df_val, tag_to_idx)
    df_test["target_tags"] = create_target_encoding(df_test, tag_to_idx)

    logger.info(f"Creating tokenizer: {model_name}")
    tokenizer = create_tokenizer(model_name, max_length=max_title_len)

    logger.info("Creating datasets...")
    train_dataset = TransformerNewsDataset(
        df=df_train,
        tokenizer=tokenizer,
        max_title_len=max_title_len,
        max_snippet_len=max_snippet_len if use_snippet else None,
        label_to_idx=tag_to_idx,
    )
    val_dataset = TransformerNewsDataset(
        df=df_val,
        tokenizer=tokenizer,
        max_title_len=max_title_len,
        max_snippet_len=max_snippet_len if use_snippet else None,
        label_to_idx=tag_to_idx,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
    )

    logger.info(f"Creating model with {num_labels} labels, use_snippet={use_snippet}...")
    model = RussianNewsClassifier(
        model_name=model_name,
        num_labels=num_labels,
        dropout=0.3,
        use_snippet=use_snippet,
        freeze_bert=freeze_backbone,
    )

    lora_meta: dict = {"enabled": False}
    if use_lora:
        if freeze_backbone:
            logger.warning(
                "Both --freeze-backbone and --use-lora were set. LoRA expects backbone trainable; proceeding anyway."
            )
        targets = (
            [t.strip() for t in (lora_target_modules or "").split(",") if t.strip()]
            or _default_lora_target_modules(model_name)
        )
        logger.info(f"Enabling LoRA on backbone: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}, targets={targets}")
        lora_meta = _apply_lora_to_backbone(
            model,
            model_name=model_name,
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_modules=targets,
        )

    num_training_steps = len(train_loader) * epochs
    lightning_module = TransformerClassificationModule(
        model=model,
        learning_rate=learning_rate,
        warmup_steps=500,
        weight_decay=0.01,
        num_training_steps=num_training_steps,
        use_snippet=use_snippet,
    )

    # Setup logging. NOTE: train_loss/val_loss are logged inside TransformerClassificationModule.
    logger_instance = CSVLogger(save_dir="logs/")
    if logger_backend == "wandb":
        if WandbLogger is None:
            raise RuntimeError(
                "WandbLogger unavailable. Ensure `wandb` and the pytorch-lightning wandb integration are installed."
            )
        os.environ["WANDB_MODE"] = wandb_mode
        logger_instance = WandbLogger(
            project=wandb_project,
            name=wandb_run_name or output_path_p.stem,
            log_model=False,
        )

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename="checkpoint-{epoch:02d}-{val_f1:.3f}",
        monitor="val_f1",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    early_stopping = EarlyStopping(
        monitor="val_f1",
        mode="max",
        patience=3,
        verbose=True,
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        logger=logger_instance,
        callbacks=[checkpoint_callback, early_stopping],
        accelerator="auto",
        devices="auto",
        gradient_clip_val=1.0,
        accumulate_grad_batches=accumulate_grad_batches,
        log_every_n_steps=50,
        enable_progress_bar=True,
    )

    logger.info("Starting training...")
    trainer.fit(lightning_module, train_loader, val_loader)
    logger.info("Training complete!")

    best_model_path = checkpoint_callback.best_model_path
    logger.info(f"Best model checkpoint: {best_model_path}")

    best_module = TransformerClassificationModule.load_from_checkpoint(best_model_path, model=model)

    # If LoRA is enabled, merge adapters into base weights so inference does NOT require PEFT.
    if use_lora and hasattr(best_module.model.bert, "merge_and_unload"):
        try:
            best_module.model.bert = best_module.model.bert.merge_and_unload()
            logger.info("Merged LoRA adapters into base backbone weights for saving.")
        except Exception as e:
            logger.warning(
                f"Failed to merge LoRA adapters into base weights: {e}. Saving adapter-wrapped state_dict instead."
            )
            lora_meta["merged_into_base"] = False

    logger.info(f"Saving model to {output_path}...")
    save_dict = {
        "state_dict": best_module.model.state_dict(),
        "num_labels": num_labels,
        "tag_to_idx": tag_to_idx,
        "model_name": model_name,
        "dropout": 0.3,
        "use_snippet": use_snippet,
        "freeze_backbone": freeze_backbone,
        "protocol_dir": protocol_dir,
        "lora": lora_meta,
    }
    torch.save(save_dict, output_path)
    logger.info(f"Model saved successfully to {output_path}")

    return best_module, tag_to_idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Russian news classification model (protocol + LoRA + W&B)")
    parser.add_argument("--data-path", type=str, default="data/news_data/ria_news.tsv", help="Path to training data TSV file")
    parser.add_argument("--output-path", type=str, default="models/best_model.pt", help="Path to save trained model")
    parser.add_argument("--model-name", type=str, default="DeepPavlov/rubert-base-cased", help="HuggingFace model name or local path")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--accumulate-grad-batches", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--use-snippet", type=_parse_bool, default=False, help="Use snippets in addition to titles")
    parser.add_argument("--freeze-backbone", type=_parse_bool, default=False, help="Freeze transformer backbone (trains only head)")
    parser.add_argument("--max-title-len", type=int, default=128, help="Max title token length")
    parser.add_argument("--max-snippet-len", type=int, default=256, help="Max snippet token length (if snippets enabled)")
    parser.add_argument("--min-tag-frequency", type=int, default=30, help="Min tag frequency (used only when protocol-dir is not provided)")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Limit training samples (only when protocol-dir is not provided)")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Limit validation samples (only when protocol-dir is not provided)")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader num_workers")
    parser.add_argument("--protocol-dir", type=str, default=None, help="Frozen protocol directory with splits.json + tag_to_idx.json")

    parser.add_argument("--use-lora", type=_parse_bool, default=False, help="Enable LoRA (PEFT) adapters on transformer backbone")
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--lora-target-modules", type=str, default=None, help="Comma-separated target module names (optional)")

    parser.add_argument(
        "--logger",
        type=str,
        default="csv",
        choices=["csv", "wandb"],
        help="Logger backend (wandb requires wandb installed and WANDB_API_KEY configured)",
    )
    parser.add_argument("--wandb-project", type=str, default="russian-news-classification", help="W&B project (when --logger wandb)")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name (defaults to output checkpoint stem)")
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"], help="W&B mode")

    args = parser.parse_args()

    train_model(
        data_path=args.data_path,
        output_path=args.output_path,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        accumulate_grad_batches=args.accumulate_grad_batches,
        learning_rate=args.learning_rate,
        use_snippet=args.use_snippet,
        freeze_backbone=args.freeze_backbone,
        max_title_len=args.max_title_len,
        max_snippet_len=args.max_snippet_len,
        min_tag_frequency=args.min_tag_frequency,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        num_workers=args.num_workers,
        protocol_dir=args.protocol_dir,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        logger_backend=args.logger,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_mode=args.wandb_mode,
    )




