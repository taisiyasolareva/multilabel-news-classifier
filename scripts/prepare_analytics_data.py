#!/usr/bin/env python3
"""Prepare CSV files for analytics dashboard from original dataset.

This script generates the required CSV formats for:
1. Predictive Intervals: sentiment counts per news item
2. Category Analytics: news articles with categories
3. Thread Analysis: comments with news IDs
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from analysis.sentiment_analyzer import SentimentAnalyzer
from data.data_loader import load_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_category_analytics_csv(
    ria_path: str,
    output_path: str = "data/analytics_category_data.csv",
    max_samples: int = None
):
    """
    Prepare CSV for Category Analytics tab.
    
    Required columns: category, text
    """
    logger.info(f"Loading RIA news from {ria_path}")
    df = pd.read_csv(ria_path, sep='\t')
    df = df[~df.tags.isnull()]
    
    if max_samples:
        df = df.head(max_samples)
    
    # Use title + snippet as text, or just title if snippet is missing
    df['text'] = df.apply(
        lambda row: f"{row['title']} {row.get('snippet', '')}".strip(),
        axis=1
    )
    
    # Use first tag as category (or 'unknown' if no tags)
    df['category'] = df['tags'].apply(
        lambda x: str(x).split(',')[0].strip() if pd.notna(x) else 'unknown'
    )
    
    # Select only required columns
    output_df = df[['category', 'text']].copy()
    
    output_df.to_csv(output_path, index=False)
    logger.info(f"✅ Created category analytics CSV: {output_path} ({len(output_df)} rows)")
    
    return output_path


def prepare_thread_analysis_csv(
    vk_comments_path: str,
    output_path: str = "data/analytics_thread_data.csv",
    max_samples: int = None
):
    """
    Prepare CSV for Thread Analysis tab.
    
    Required columns: news_id, text
    """
    logger.info(f"Loading VK comments from {vk_comments_path}")
    df = pd.read_csv(vk_comments_path, sep='\t')
    df = df[~df.text.isnull()]
    
    if max_samples:
        df = df.head(max_samples)
    
    # Use post_id as news_id
    df['news_id'] = df['post_id'].astype(str)
    
    # Select only required columns
    output_df = df[['news_id', 'text']].copy()
    
    output_df.to_csv(output_path, index=False)
    logger.info(f"✅ Created thread analysis CSV: {output_path} ({len(output_df)} rows)")
    
    return output_path


def prepare_predictive_intervals_csv(
    vk_comments_path: str,
    vk_news_path: str = None,
    output_path: str = "data/analytics_sentiment_counts.csv",
    max_news_items: int = 50,
    max_comments_per_item: int = 1000
):
    """
    Prepare CSV for Predictive Intervals tab.
    
    Required columns: id, positive_count, negative_count, neutral_count
    
    This analyzes comments with sentiment to get counts per news item.
    """
    logger.info(f"Loading VK comments from {vk_comments_path}")
    df_comments = pd.read_csv(vk_comments_path, sep='\t')
    df_comments = df_comments[~df_comments.text.isnull()]
    
    # Limit comments per news item
    df_comments = df_comments.groupby('post_id').head(max_comments_per_item)
    
    # Get unique news items
    news_ids = df_comments['post_id'].unique()[:max_news_items]
    logger.info(f"Analyzing sentiment for {len(news_ids)} news items...")
    
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
            batch_results = analyzer.analyze_batch(batch)
            sentiments.extend(batch_results)
        
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
    
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_path, index=False)
    logger.info(f"✅ Created predictive intervals CSV: {output_path} ({len(output_df)} rows)")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Prepare CSV files for analytics dashboard")
    parser.add_argument(
        "--ria-path",
        type=str,
        default="data/news_data/ria_news.tsv",
        help="Path to RIA news TSV file"
    )
    parser.add_argument(
        "--vk-comments-path",
        type=str,
        default="data/news_data/vk_comments.tsv",
        help="Path to VK comments TSV file"
    )
    parser.add_argument(
        "--vk-news-path",
        type=str,
        default=None,
        help="Path to VK news TSV file (optional)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for CSV files"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples for category analytics (for testing)"
    )
    parser.add_argument(
        "--max-news-items",
        type=int,
        default=50,
        help="Maximum news items for predictive intervals (sentiment analysis is slow)"
    )
    parser.add_argument(
        "--max-comments-per-item",
        type=int,
        default=1000,
        help="Maximum comments per news item for sentiment analysis"
    )
    parser.add_argument(
        "--skip-sentiment",
        action="store_true",
        help="Skip sentiment analysis (slow step)"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Preparing Analytics Dashboard CSV Files")
    logger.info("=" * 60)
    
    # 1. Category Analytics CSV
    logger.info("\n1. Preparing Category Analytics CSV...")
    category_path = output_dir / "analytics_category_data.csv"
    prepare_category_analytics_csv(
        args.ria_path,
        str(category_path),
        max_samples=args.max_samples
    )
    
    # 2. Thread Analysis CSV
    logger.info("\n2. Preparing Thread Analysis CSV...")
    if Path(args.vk_comments_path).exists():
        thread_path = output_dir / "analytics_thread_data.csv"
        prepare_thread_analysis_csv(
            args.vk_comments_path,
            str(thread_path),
            max_samples=args.max_samples
        )
    else:
        logger.warning(f"⚠️  VK comments file not found: {args.vk_comments_path}")
        logger.warning("   Skipping thread analysis CSV")
    
    # 3. Predictive Intervals CSV (requires sentiment analysis - slow!)
    if not args.skip_sentiment:
        logger.info("\n3. Preparing Predictive Intervals CSV (sentiment analysis)...")
        logger.info("   ⚠️  This step is slow - analyzing sentiment for comments...")
        if Path(args.vk_comments_path).exists():
            sentiment_path = output_dir / "analytics_sentiment_counts.csv"
            prepare_predictive_intervals_csv(
                args.vk_comments_path,
                args.vk_news_path,
                str(sentiment_path),
                max_news_items=args.max_news_items,
                max_comments_per_item=args.max_comments_per_item
            )
        else:
            logger.warning(f"⚠️  VK comments file not found: {args.vk_comments_path}")
            logger.warning("   Skipping predictive intervals CSV")
    else:
        logger.info("\n3. Skipping Predictive Intervals CSV (--skip-sentiment flag)")
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ All CSV files prepared!")
    logger.info("=" * 60)
    logger.info("\nNext steps:")
    logger.info("1. Open Streamlit analytics dashboard:")
    logger.info("   streamlit run dashboards/analytics_dashboard.py")
    logger.info("2. Upload the generated CSV files in each tab:")
    logger.info(f"   - Category Analytics: {output_dir}/analytics_category_data.csv")
    if Path(args.vk_comments_path).exists():
        logger.info(f"   - Thread Analysis: {output_dir}/analytics_thread_data.csv")
        if not args.skip_sentiment:
            logger.info(f"   - Predictive Intervals: {output_dir}/analytics_sentiment_counts.csv")


if __name__ == "__main__":
    main()



