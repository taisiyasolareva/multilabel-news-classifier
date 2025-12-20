"""Script to generate reference statistics for drift detection."""

import logging
import argparse
from pathlib import Path
import pandas as pd

from data.data_loader import load_data
from utils.text_processing import normalise_text
from monitoring.data_drift import DataDriftDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_reference_stats(
    data_path: str,
    output_path: str = "monitoring/reference_stats.json",
) -> None:
    """
    Generate reference statistics from training data.
    
    Args:
        data_path: Path to training data TSV file
        output_path: Path to save reference statistics
    """
    logger.info(f"Loading data from {data_path}")
    
    # Load data
    df, _, _ = load_data(data_path)
    
    # Process text
    df['title_clean'] = df['title'].apply(normalise_text)
    if 'snippet' in df.columns:
        df['snippet_clean'] = df['snippet'].fillna("").apply(normalise_text)
    else:
        df['snippet_clean'] = ""
    
    # Create drift detector
    detector = DataDriftDetector(reference_data=df[['title_clean', 'snippet_clean']])
    
    # Save reference statistics
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    detector.save_reference_stats(str(output_path))
    
    logger.info(f"Reference statistics saved to {output_path}")
    logger.info(f"Statistics computed for {len(df)} samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate reference statistics")
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to training data TSV file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="monitoring/reference_stats.json",
        help="Output path for reference statistics"
    )
    
    args = parser.parse_args()
    
    generate_reference_stats(
        data_path=args.data_path,
        output_path=args.output,
    )




