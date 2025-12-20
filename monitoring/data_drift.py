"""Data drift detection utilities."""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats
from collections import defaultdict
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class DataDriftDetector:
    """
    Detect data drift in input features.
    
    Compares current data distribution to reference (training) distribution.
    """

    def __init__(
        self,
        reference_data: Optional[pd.DataFrame] = None,
        reference_stats: Optional[Dict] = None,
        drift_threshold: float = 0.05,
    ):
        """
        Initialize data drift detector.
        
        Args:
            reference_data: Reference DataFrame (training data)
            reference_stats: Pre-computed reference statistics
            drift_threshold: Threshold for drift detection (5%)
        """
        self.drift_threshold = drift_threshold
        self.reference_stats = reference_stats or {}
        
        if reference_data is not None:
            self.reference_stats = self._compute_statistics(reference_data)
        
        # Store current data for comparison
        self.current_data: List[Dict] = []
        
        logger.info("DataDriftDetector initialized")

    def _compute_statistics(self, data: pd.DataFrame) -> Dict:
        """
        Compute statistics for reference data.
        
        Args:
            data: DataFrame with features
            
        Returns:
            Dictionary of statistics
        """
        stats_dict = {}
        
        # Text length statistics
        if "title" in data.columns:
            title_lengths = data["title"].str.len()
            stats_dict["title_length"] = {
                "mean": float(title_lengths.mean()),
                "std": float(title_lengths.std()),
                "min": float(title_lengths.min()),
                "max": float(title_lengths.max()),
            }
        
        if "snippet" in data.columns:
            snippet_lengths = data["snippet"].str.len()
            stats_dict["snippet_length"] = {
                "mean": float(snippet_lengths.mean()),
                "std": float(snippet_lengths.std()),
                "min": float(snippet_lengths.min()),
                "max": float(snippet_lengths.max()),
            }
        
        # Word count statistics
        if "title" in data.columns:
            word_counts = data["title"].str.split().str.len()
            stats_dict["title_word_count"] = {
                "mean": float(word_counts.mean()),
                "std": float(word_counts.std()),
            }
        
        # Character distribution (for Russian text)
        if "title" in data.columns:
            cyrillic_ratio = data["title"].apply(
                lambda x: len([c for c in str(x) if '\u0400' <= c <= '\u04FF']) / max(len(str(x)), 1)
            )
            stats_dict["cyrillic_ratio"] = {
                "mean": float(cyrillic_ratio.mean()),
                "std": float(cyrillic_ratio.std()),
            }
        
        return stats_dict

    def record_sample(
        self,
        title: str,
        snippet: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Record a sample for drift detection.
        
        Args:
            title: Article title
            snippet: Optional snippet
            metadata: Optional metadata
        """
        sample = {
            "timestamp": datetime.now().isoformat(),
            "title": title,
            "snippet": snippet,
            "metadata": metadata or {},
        }
        
        self.current_data.append(sample)
        
        # Keep only recent samples (last 1000)
        if len(self.current_data) > 1000:
            self.current_data = self.current_data[-1000:]

    def detect_drift(
        self,
        window_size: int = 100,
    ) -> Tuple[bool, Dict]:
        """
        Detect data drift in recent samples.
        
        Args:
            window_size: Number of recent samples to analyze
            
        Returns:
            Tuple of (has_drift, drift_info)
        """
        if len(self.current_data) < window_size:
            return False, {"error": "Insufficient data"}
        
        if not self.reference_stats:
            return False, {"error": "No reference statistics"}
        
        # Get recent samples
        recent = self.current_data[-window_size:]
        recent_df = pd.DataFrame(recent)
        
        # Compute current statistics
        current_stats = self._compute_statistics(recent_df)
        
        # Compare with reference
        drift_info = {}
        has_drift = False
        
        for feature, ref_stats in self.reference_stats.items():
            if feature not in current_stats:
                continue
            
            curr_stats = current_stats[feature]
            
            # Compare means (relative change)
            if "mean" in ref_stats and "mean" in curr_stats:
                ref_mean = ref_stats["mean"]
                curr_mean = curr_stats["mean"]
                
                if ref_mean > 0:
                    relative_change = abs(curr_mean - ref_mean) / ref_mean
                    
                    drift_info[feature] = {
                        "reference_mean": ref_mean,
                        "current_mean": curr_mean,
                        "relative_change": relative_change,
                        "drifted": relative_change > self.drift_threshold,
                    }
                    
                    if relative_change > self.drift_threshold:
                        has_drift = True
                        logger.warning(
                            f"Data drift detected in {feature}: "
                            f"relative change {relative_change:.2%} "
                            f"(ref: {ref_mean:.2f}, curr: {curr_mean:.2f})"
                        )
        
        return has_drift, drift_info

    def statistical_test(
        self,
        feature: str,
        window_size: int = 100,
    ) -> Dict:
        """
        Perform statistical test for drift (KS test).
        
        Args:
            feature: Feature to test
            window_size: Number of recent samples
            
        Returns:
            Dictionary with test results
        """
        if len(self.current_data) < window_size:
            return {"error": "Insufficient data"}
        
        recent = self.current_data[-window_size:]
        recent_df = pd.DataFrame(recent)
        
        # Extract feature values
        if feature == "title_length":
            current_values = recent_df["title"].str.len().values
        elif feature == "snippet_length":
            current_values = recent_df["snippet"].str.len().values
        elif feature == "title_word_count":
            current_values = recent_df["title"].str.split().str.len().values
        else:
            return {"error": f"Unknown feature: {feature}"}
        
        # Get reference statistics
        if feature not in self.reference_stats:
            return {"error": f"No reference stats for {feature}"}
        
        ref_stats = self.reference_stats[feature]
        
        # Generate reference distribution (normal approximation)
        if "mean" in ref_stats and "std" in ref_stats:
            ref_mean = ref_stats["mean"]
            ref_std = ref_stats["std"]
            
            # Kolmogorov-Smirnov test
            # Create reference distribution sample
            ref_sample = np.random.normal(ref_mean, ref_std, size=1000)
            
            # Perform KS test
            statistic, p_value = stats.ks_2samp(current_values, ref_sample)
            
            return {
                "feature": feature,
                "ks_statistic": float(statistic),
                "p_value": float(p_value),
                "drifted": p_value < 0.05,  # Significant drift
                "reference_mean": ref_mean,
                "current_mean": float(np.mean(current_values)),
            }
        
        return {"error": "Insufficient reference statistics"}

    def save_reference_stats(self, path: str) -> None:
        """Save reference statistics to file."""
        with open(path, 'w') as f:
            json.dump(self.reference_stats, f, indent=2)
        logger.info(f"Reference statistics saved to {path}")

    def load_reference_stats(self, path: str) -> None:
        """Load reference statistics from file."""
        with open(path) as f:
            self.reference_stats = json.load(f)
        logger.info(f"Reference statistics loaded from {path}")




