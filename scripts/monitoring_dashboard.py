"""Simple monitoring dashboard script."""

import logging
import json
from pathlib import Path
from datetime import datetime, timedelta

from monitoring.performance_monitor import PerformanceMonitor
from monitoring.data_drift import DataDriftDetector
from monitoring.prediction_logger import PredictionLogger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def print_dashboard() -> None:
    """Print monitoring dashboard."""
    print("=" * 80)
    print("MONITORING DASHBOARD")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Performance metrics
    print("PERFORMANCE METRICS")
    print("-" * 80)
    try:
        monitor = PerformanceMonitor()
        recent_metrics = monitor.get_recent_metrics()
        
        if recent_metrics:
            print(f"Recent Metrics (last {monitor.window_size} predictions):")
            print(f"  Precision: {recent_metrics.get('precision', 0):.3f}")
            print(f"  Recall:    {recent_metrics.get('recall', 0):.3f}")
            print(f"  F1 Score:  {recent_metrics.get('f1', 0):.3f}")
            print(f"  Exact Match: {recent_metrics.get('exact_match', 0):.3f}")
            print(f"  Count:     {recent_metrics.get('count', 0)}")
        else:
            print("  No metrics available")
        
        # Performance report
        report = monitor.generate_report()
        print(f"\nTotal Predictions: {report.get('total_predictions', 0)}")
        print(f"Predictions with Metrics: {report.get('predictions_with_metrics', 0)}")
        
    except Exception as e:
        print(f"  Error: {e}")
    
    print()
    
    # Data drift
    print("DATA DRIFT DETECTION")
    print("-" * 80)
    try:
        detector = DataDriftDetector()
        ref_stats_path = Path("monitoring/reference_stats.json")
        
        if ref_stats_path.exists():
            detector.load_reference_stats(str(ref_stats_path))
            has_drift, drift_info = detector.detect_drift()
            
            if has_drift:
                print("⚠️  DRIFT DETECTED!")
                for feature, info in drift_info.items():
                    if info.get("drifted"):
                        print(f"  {feature}:")
                        print(f"    Reference: {info.get('reference_mean', 0):.2f}")
                        print(f"    Current:   {info.get('current_mean', 0):.2f}")
                        print(f"    Change:    {info.get('relative_change', 0):.2%}")
            else:
                print("✓ No drift detected")
        else:
            print("  Reference statistics not found")
            print("  Run: python scripts/generate_reference_stats.py --data-path <path>")
    
    except Exception as e:
        print(f"  Error: {e}")
    
    print()
    
    # Prediction logs
    print("PREDICTION LOGS")
    print("-" * 80)
    try:
        logger = PredictionLogger()
        analysis = logger.analyze_logs(days=7)
        
        print(f"Total Predictions (7 days): {analysis.get('total_predictions', 0)}")
        print(f"Unique Titles: {analysis.get('unique_titles', 0)}")
        
        avg_latency = analysis.get('avg_latency_ms')
        if avg_latency:
            print(f"Average Latency: {avg_latency:.2f} ms")
        
        model_versions = analysis.get('model_versions', {})
        if model_versions:
            print("Model Versions:")
            for version, count in model_versions.items():
                print(f"  {version}: {count} predictions")
    
    except Exception as e:
        print(f"  Error: {e}")
    
    print()
    print("=" * 80)


if __name__ == "__main__":
    print_dashboard()




