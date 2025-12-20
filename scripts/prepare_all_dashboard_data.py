#!/usr/bin/env python3
"""Master script to prepare ALL data files needed for all Streamlit dashboards.

This script generates:
1. Evaluation Dashboard: predictions CSV
2. Analytics Dashboard: category, thread, and sentiment counts CSVs
3. Model Comparison Dashboard: experiment results (if available)

Usage:
    python scripts/prepare_all_dashboard_data.py --checkpoint models/best_model_v2.pt
"""

import sys
import argparse
import subprocess
from pathlib import Path
import pandas as pd
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from analysis.sentiment_analyzer import SentimentAnalyzer
from data.data_loader import load_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def prepare_evaluation_predictions(
    checkpoint_path: str,
    data_path: str,
    output_path: str,
    max_val_samples: int = None,
    threshold: float = 0.5
):
    """Generate predictions CSV for Evaluation Dashboard."""
    logger.info("=" * 60)
    logger.info("1. Preparing Evaluation Dashboard Data")
    logger.info("=" * 60)
    
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "evaluate.py"),
        "--checkpoint", checkpoint_path,
        "--data-path", data_path,
        "--threshold", str(threshold),
        "--output-csv", output_path
    ]
    
    if max_val_samples:
        cmd.extend(["--max-val-samples", str(max_val_samples)])
    
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        logger.info(f"✅ Evaluation predictions saved to: {output_path}")
        return True
    else:
        logger.error(f"❌ Failed to generate evaluation predictions:")
        logger.error(result.stderr)
        return False


def prepare_category_analytics_csv(
    ria_path: str,
    output_path: str,
    max_samples: int = None
):
    """Prepare CSV for Category Analytics tab."""
    logger.info(f"Loading RIA news from {ria_path}")
    df = pd.read_csv(ria_path, sep='\t')
    df = df[~df.tags.isnull()]
    
    if max_samples:
        df = df.head(max_samples)
    
    # Use title + snippet as text
    df['text'] = df.apply(
        lambda row: f"{row['title']} {row.get('snippet', '')}".strip(),
        axis=1
    )
    
    # Use first tag as category
    df['category'] = df['tags'].apply(
        lambda x: str(x).split(',')[0].strip() if pd.notna(x) else 'unknown'
    )
    
    output_df = df[['category', 'text']].copy()
    output_df.to_csv(output_path, index=False)
    logger.info(f"✅ Category analytics CSV: {output_path} ({len(output_df)} rows)")
    return output_path


def prepare_thread_analysis_csv(
    vk_comments_path: str,
    output_path: str,
    max_samples: int = None
):
    """Prepare CSV for Thread Analysis tab."""
    if not Path(vk_comments_path).exists():
        logger.warning(f"⚠️  VK comments file not found: {vk_comments_path}")
        return None
    
    logger.info(f"Loading VK comments from {vk_comments_path}")
    df = pd.read_csv(vk_comments_path, sep='\t')
    df = df[~df.text.isnull()]
    
    if max_samples:
        df = df.head(max_samples)
    
    df['news_id'] = df['post_id'].astype(str)
    output_df = df[['news_id', 'text']].copy()
    output_df.to_csv(output_path, index=False)
    logger.info(f"✅ Thread analysis CSV: {output_path} ({len(output_df)} rows)")
    return output_path


def prepare_predictive_intervals_csv(
    vk_comments_path: str,
    output_path: str,
    max_news_items: int = 50,
    max_comments_per_item: int = 1000
):
    """Prepare CSV for Predictive Intervals tab (requires sentiment analysis)."""
    if not Path(vk_comments_path).exists():
        logger.warning(f"⚠️  VK comments file not found: {vk_comments_path}")
        return None
    
    logger.info(f"Loading VK comments from {vk_comments_path}")
    df_comments = pd.read_csv(vk_comments_path, sep='\t')
    df_comments = df_comments[~df_comments.text.isnull()]
    
    # Limit comments per news item
    df_comments = df_comments.groupby('post_id').head(max_comments_per_item)
    
    # Get unique news items
    news_ids = df_comments['post_id'].unique()[:max_news_items]
    logger.info(f"Analyzing sentiment for {len(news_ids)} news items...")
    logger.info("⚠️  This step is slow - analyzing sentiment for comments...")
    
    # Initialize sentiment analyzer
    analyzer = SentimentAnalyzer()
    
    results = []
    for i, news_id in enumerate(news_ids):
        if (i + 1) % 10 == 0:
            logger.info(f"Processing {i+1}/{len(news_ids)}...")
        
        # Get comments for this news item
        comments = df_comments[df_comments['post_id'] == news_id]['text'].tolist()
        
        if not comments:
            continue
        
        # Analyze sentiment in batches
        sentiments = []
        batch_size = 50
        for j in range(0, len(comments), batch_size):
            batch = comments[j:j+batch_size]
            try:
                batch_results = analyzer.analyze_batch(batch)
                sentiments.extend(batch_results)
            except Exception as e:
                logger.warning(f"Error analyzing batch for news_id {news_id}: {e}")
                continue
        
        # Count sentiments
        positive_count = sum(1 for s in sentiments if s.get('label') == 'POSITIVE')
        negative_count = sum(1 for s in sentiments if s.get('label') == 'NEGATIVE')
        neutral_count = sum(1 for s in sentiments if s.get('label') == 'NEUTRAL')
        
        results.append({
            'id': str(news_id),
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count
        })
    
    if not results:
        logger.warning("⚠️  No sentiment results generated")
        return None
    
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_path, index=False)
    logger.info(f"✅ Predictive intervals CSV: {output_path} ({len(output_df)} rows)")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Prepare ALL data files for Streamlit dashboards",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all data with default settings
  python scripts/prepare_all_dashboard_data.py --checkpoint models/best_model_v2.pt
  
  # Quick test with limited samples (faster)
  python scripts/prepare_all_dashboard_data.py \\
      --checkpoint models/best_model_v2.pt \\
      --max-val-samples 100 \\
      --max-samples 500 \\
      --max-news-items 10 \\
      --skip-sentiment
  
  # Full dataset (slow, especially sentiment analysis)
  python scripts/prepare_all_dashboard_data.py \\
      --checkpoint models/best_model_v2.pt \\
      --skip-sentiment  # Skip slow sentiment analysis
        """
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/news_data/ria_news.tsv",
        help="Path to RIA news TSV file"
    )
    parser.add_argument(
        "--vk-comments-path",
        type=str,
        default="data/vk_comments.tsv",
        help="Path to VK comments TSV file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments",
        help="Output directory for all CSV files"
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help="Maximum validation samples for evaluation (for testing)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples for category/thread analytics (for testing)"
    )
    parser.add_argument(
        "--max-news-items",
        type=int,
        default=50,
        help="Maximum news items for predictive intervals (sentiment is slow)"
    )
    parser.add_argument(
        "--max-comments-per-item",
        type=int,
        default=1000,
        help="Maximum comments per news item for sentiment analysis"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for evaluation predictions"
    )
    parser.add_argument(
        "--skip-sentiment",
        action="store_true",
        help="Skip sentiment analysis (slow step)"
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip evaluation predictions generation"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Preparing ALL Dashboard Data Files")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info("")
    
    results = {}
    
    # 1. Evaluation Dashboard: Predictions CSV
    if not args.skip_evaluation:
        eval_output = output_dir / "dashboard_eval_predictions.csv"
        success = prepare_evaluation_predictions(
            args.checkpoint,
            args.data_path,
            str(eval_output),
            max_val_samples=args.max_val_samples,
            threshold=args.threshold
        )
        results['evaluation'] = str(eval_output) if success else None
    else:
        logger.info("Skipping evaluation predictions (--skip-evaluation)")
        results['evaluation'] = None
    
    logger.info("")
    
    # 2. Analytics Dashboard: Category Analytics CSV
    logger.info("=" * 60)
    logger.info("2. Preparing Analytics Dashboard Data")
    logger.info("=" * 60)
    
    category_output = output_dir / "analytics_category_data.csv"
    prepare_category_analytics_csv(
        args.data_path,
        str(category_output),
        max_samples=args.max_samples
    )
    results['category_analytics'] = str(category_output)
    
    logger.info("")
    
    # 3. Analytics Dashboard: Thread Analysis CSV
    thread_output = output_dir / "analytics_thread_data.csv"
    vk_comments_exists = Path(args.vk_comments_path).exists()
    
    if vk_comments_exists:
        thread_path = prepare_thread_analysis_csv(
            args.vk_comments_path,
            str(thread_output),
            max_samples=args.max_samples
        )
        results['thread_analysis'] = thread_path
    else:
        logger.warning(f"⚠️  VK comments file not found: {args.vk_comments_path}")
        logger.warning("   Thread Analysis CSV requires VK comments data")
        logger.warning("   Download from: https://drive.google.com/drive/folders/11oCcLplWtp_qm-WuEbfCFP_Mz5K_z3ps")
        results['thread_analysis'] = None
    
    logger.info("")
    
    # 4. Analytics Dashboard: Predictive Intervals CSV (sentiment analysis)
    if not args.skip_sentiment:
        if vk_comments_exists:
            sentiment_output = output_dir / "analytics_sentiment_counts.csv"
            sentiment_path = prepare_predictive_intervals_csv(
                args.vk_comments_path,
                str(sentiment_output),
                max_news_items=args.max_news_items,
                max_comments_per_item=args.max_comments_per_item
            )
            results['predictive_intervals'] = sentiment_path
        else:
            logger.warning(f"⚠️  VK comments file not found: {args.vk_comments_path}")
            logger.warning("   Predictive Intervals CSV requires VK comments data")
            logger.warning("   Download from: https://drive.google.com/drive/folders/11oCcLplWtp_qm-WuEbfCFP_Mz5K_z3ps")
            results['predictive_intervals'] = None
    else:
        logger.info("Skipping predictive intervals (--skip-sentiment)")
        results['predictive_intervals'] = None
    
    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("✅ Dashboard Data Preparation Complete!")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Generated files:")
    logger.info("")
    
    if results['evaluation']:
        logger.info(f"📊 Evaluation Dashboard:")
        logger.info(f"   {results['evaluation']}")
        logger.info("")
    
    logger.info(f"📈 Analytics Dashboard:")
    logger.info(f"   Category Analytics: {results['category_analytics']}")
    if results['thread_analysis']:
        logger.info(f"   Thread Analysis: {results['thread_analysis']}")
    else:
        logger.info(f"   Thread Analysis: (missing - requires VK comments file)")
    if results['predictive_intervals']:
        logger.info(f"   Predictive Intervals: {results['predictive_intervals']}")
    else:
        if args.skip_sentiment:
            logger.info(f"   Predictive Intervals: (skipped - use without --skip-sentiment to generate)")
        else:
            logger.info(f"   Predictive Intervals: (missing - requires VK comments file)")
    logger.info("")
    
    logger.info("📝 Sentiment Dashboard:")
    logger.info("   No CSV needed - uses FastAPI endpoint")
    logger.info("")
    
    logger.info("📊 Model Comparison Dashboard:")
    logger.info("   Uses experiment tracker or upload CSV manually")
    logger.info("")
    
    logger.info("=" * 60)
    logger.info("Next Steps:")
    logger.info("=" * 60)
    logger.info("1. Start FastAPI server:")
    logger.info("   uvicorn api.main:app --reload --port 8000")
    logger.info("")
    logger.info("2. Run dashboards:")
    logger.info("   streamlit run dashboards/evaluation_dashboard.py")
    logger.info("   streamlit run dashboards/analytics_dashboard.py")
    logger.info("   streamlit run dashboards/sentiment_dashboard.py")
    logger.info("   streamlit run dashboards/model_comparison_dashboard.py")
    logger.info("")
    logger.info("3. Upload the generated CSV files in each dashboard")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

