#!/usr/bin/env python3
"""Generate evaluation-ready predictions CSV from tiny validation set.

Loads the 10-sample validation set, runs model inference, and creates
a CSV with predictions and ground truth in the format expected by the
evaluation dashboard.
"""

import logging
import sys
from pathlib import Path

import torch
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.transformer_model import RussianNewsClassifier
from utils.tokenization import create_tokenizer, tokenize_text_pair
from utils.text_processing import normalise_text
from utils.data_processing import process_tags

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(checkpoint_path: Path, device: torch.device):
    """Load model and label mapping from saved checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint from {checkpoint_path}...")
    save_dict = torch.load(checkpoint_path, map_location=device)

    state_dict = save_dict["state_dict"]
    num_labels = save_dict["num_labels"]
    tag_to_idx = save_dict["tag_to_idx"]
    model_name = save_dict.get("model_name", "DeepPavlov/rubert-base-cased")
    dropout = save_dict.get("dropout", 0.3)
    use_snippet = save_dict.get("use_snippet", False)

    model = RussianNewsClassifier(
        model_name=model_name,
        num_labels=num_labels,
        dropout=dropout,
        use_snippet=use_snippet,
        freeze_bert=False,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    logger.info(
        f"Loaded model: {model_name} | num_labels={num_labels} | use_snippet={use_snippet}"
    )

    return model, tag_to_idx


def tags_to_binary(tags_str: str, tag_to_idx: dict) -> np.ndarray:
    """Convert comma-separated tags string to binary multi-label vector."""
    if pd.isna(tags_str) or not tags_str:
        return np.zeros(len(tag_to_idx), dtype=int)
    
    # Process tags (normalize, lowercase, split)
    tags_processed = process_tags(pd.Series([tags_str])).iloc[0]
    if not tags_processed:
        return np.zeros(len(tag_to_idx), dtype=int)
    
    tag_list = [t.strip() for t in tags_processed.split(',') if t.strip()]
    
    # Create binary vector
    binary = np.zeros(len(tag_to_idx), dtype=int)
    for tag in tag_list:
        if tag in tag_to_idx:
            idx = tag_to_idx[tag]
            binary[idx] = 1
    
    return binary


def generate_predictions(
    val_csv_path: Path = Path("data/tiny_val.csv"),
    checkpoint_path: Path = Path("models/best_model.pt"),
    output_path: Path = Path("experiments/tiny_eval_predictions.csv"),
    use_binary_predictions: bool = False,  # If True, use 0/1; if False, use probabilities
    threshold: float = 0.5,
):
    """Generate evaluation predictions CSV from validation set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model, tag_to_idx = load_model(checkpoint_path, device)
    num_labels = len(tag_to_idx)
    
    # Create tokenizer
    tokenizer = create_tokenizer(model_name=model.model_name, max_length=128)
    
    # Load validation data
    logger.info(f"Loading validation data from {val_csv_path}...")
    df_val = pd.read_csv(val_csv_path)
    logger.info(f"Loaded {len(df_val)} validation samples")
    
    # Process tags column
    if 'tags' in df_val.columns:
        df_val['tags'] = process_tags(df_val['tags'])
    
    # Prepare data structures
    sample_ids = []
    predictions_list = []
    targets_list = []
    
    # Process each sample
    logger.info("Generating predictions...")
    for idx, row in df_val.iterrows():
        sample_id = row.get('id', idx)
        title = str(row['title'])
        snippet = str(row['snippet']) if pd.notna(row.get('snippet')) else None
        
        # Normalize text
        title_clean = normalise_text(title)
        snippet_clean = normalise_text(snippet) if snippet else None
        
        # Tokenize
        encoded = tokenize_text_pair(
            title=title_clean,
            snippet=snippet_clean,
            tokenizer=tokenizer,
            max_title_len=128,
            max_snippet_len=256 if snippet_clean else None,
        )
        
        # Prepare inputs
        title_input_ids = encoded['title_input_ids'].unsqueeze(0).to(device)
        title_attention_mask = encoded['title_attention_mask'].unsqueeze(0).to(device)
        
        snippet_input_ids = None
        snippet_attention_mask = None
        if snippet_clean and 'snippet_input_ids' in encoded:
            snippet_input_ids = encoded['snippet_input_ids'].unsqueeze(0).to(device)
            snippet_attention_mask = encoded['snippet_attention_mask'].unsqueeze(0).to(device)
        
        # Get model predictions
        with torch.no_grad():
            logits = model(
                title_input_ids=title_input_ids,
                title_attention_mask=title_attention_mask,
                snippet_input_ids=snippet_input_ids,
                snippet_attention_mask=snippet_attention_mask,
            )
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        # Convert to binary if requested
        if use_binary_predictions:
            preds = (probs >= threshold).astype(int)
        else:
            preds = probs
        
        # Get ground truth
        tags_str = row.get('tags', '')
        targets = tags_to_binary(tags_str, tag_to_idx)
        
        sample_ids.append(sample_id)
        predictions_list.append(preds)
        targets_list.append(targets)
    
    # Create DataFrame
    logger.info("Creating evaluation DataFrame...")
    
    # Build column names
    class_cols = [f'class_{i}' for i in range(num_labels)]
    target_cols = [f'target_class_{i}' for i in range(num_labels)]
    
    # Create data dictionary
    data = {'sample_id': sample_ids}
    
    # Add prediction columns
    for i, col in enumerate(class_cols):
        data[col] = [pred[i] for pred in predictions_list]
    
    # Add target columns
    for i, col in enumerate(target_cols):
        data[col] = [target[i] for target in targets_list]
    
    df_eval = pd.DataFrame(data)
    
    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_eval.to_csv(output_path, index=False)
    logger.info(f"Saved evaluation predictions to {output_path}")
    logger.info(f"DataFrame shape: {df_eval.shape}")
    logger.info(f"Columns: {list(df_eval.columns[:5])}... (showing first 5)")
    
    return df_eval


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate evaluation predictions CSV")
    parser.add_argument(
        "--val-csv",
        type=str,
        default="data/tiny_val.csv",
        help="Path to validation CSV file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/best_model.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/tiny_eval_predictions.csv",
        help="Output CSV path"
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Use binary predictions (0/1) instead of probabilities"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for binary predictions (if --binary is used)"
    )
    
    args = parser.parse_args()
    
    generate_predictions(
        val_csv_path=Path(args.val_csv),
        checkpoint_path=Path(args.checkpoint),
        output_path=Path(args.output),
        use_binary_predictions=args.binary,
        threshold=args.threshold,
    )



