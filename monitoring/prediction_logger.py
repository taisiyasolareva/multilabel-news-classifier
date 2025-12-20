"""Prediction logging utilities."""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
from pathlib import Path
import csv

logger = logging.getLogger(__name__)


class PredictionLogger:
    """
    Log predictions for analysis and debugging.
    
    Stores predictions with metadata for later analysis.
    """

    def __init__(
        self,
        log_dir: str = "monitoring/predictions",
        max_logs: int = 10000,
        log_format: str = "json",  # "json" or "csv"
    ):
        """
        Initialize prediction logger.
        
        Args:
            log_dir: Directory to store logs
            max_logs: Maximum number of logs to keep
            log_format: Log format ("json" or "csv")
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_logs = max_logs
        self.log_format = log_format
        
        # Current log file (rotates daily)
        self.current_log_file = self._get_log_file_path()
        
        logger.info(f"PredictionLogger initialized: log_dir={log_dir}, format={log_format}")

    def _get_log_file_path(self) -> Path:
        """Get log file path for current date."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        if self.log_format == "json":
            return self.log_dir / f"predictions_{date_str}.jsonl"
        else:
            return self.log_dir / f"predictions_{date_str}.csv"

    def log_prediction(
        self,
        input_data: Dict[str, Any],
        prediction: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log a prediction.
        
        Args:
            input_data: Input data (title, snippet, etc.)
            prediction: Prediction results (tags, scores)
            metadata: Optional metadata (model_version, latency, etc.)
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "input": input_data,
            "prediction": prediction,
            "metadata": metadata or {},
        }
        
        # Rotate log file if date changed
        new_log_file = self._get_log_file_path()
        if new_log_file != self.current_log_file:
            self.current_log_file = new_log_file
        
        # Write log entry
        if self.log_format == "json":
            self._write_json_log(log_entry)
        else:
            self._write_csv_log(log_entry)

    def _write_json_log(self, entry: Dict) -> None:
        """Write JSON log entry."""
        try:
            with open(self.current_log_file, 'a') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"Failed to write JSON log: {e}")

    def _write_csv_log(self, entry: Dict) -> None:
        """Write CSV log entry."""
        try:
            file_exists = self.current_log_file.exists()
            
            with open(self.current_log_file, 'a', newline='') as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "timestamp",
                        "title",
                        "snippet",
                        "predicted_tags",
                        "prediction_scores",
                        "model_version",
                        "latency_ms",
                    ],
                )
                
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow({
                    "timestamp": entry["timestamp"],
                    "title": entry["input"].get("title", ""),
                    "snippet": entry["input"].get("snippet", ""),
                    "predicted_tags": ",".join(entry["prediction"].get("tags", [])),
                    "prediction_scores": json.dumps(entry["prediction"].get("scores", {})),
                    "model_version": entry["metadata"].get("model_version", ""),
                    "latency_ms": entry["metadata"].get("latency_ms", ""),
                })
        except Exception as e:
            logger.error(f"Failed to write CSV log: {e}")

    def get_recent_logs(
        self,
        limit: int = 100,
        date: Optional[str] = None,
    ) -> List[Dict]:
        """
        Get recent prediction logs.
        
        Args:
            limit: Maximum number of logs to return
            date: Date string (YYYY-MM-DD), defaults to today
            
        Returns:
            List of log entries
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        log_file = self.log_dir / f"predictions_{date}.jsonl"
        
        if not log_file.exists():
            return []
        
        logs = []
        try:
            with open(log_file) as f:
                lines = f.readlines()
                # Get last N lines
                for line in lines[-limit:]:
                    logs.append(json.loads(line))
        except Exception as e:
            logger.error(f"Failed to read logs: {e}")
        
        return logs

    def analyze_logs(
        self,
        days: int = 7,
    ) -> Dict:
        """
        Analyze prediction logs.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary with analysis results
        """
        from datetime import timedelta
        
        analysis = {
            "total_predictions": 0,
            "unique_titles": set(),
            "avg_latency_ms": [],
            "model_versions": defaultdict(int),
            "date_range": {},
        }
        
        for i in range(days):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            logs = self.get_recent_logs(limit=10000, date=date)
            
            analysis["total_predictions"] += len(logs)
            
            for log in logs:
                # Unique titles
                if "input" in log and "title" in log["input"]:
                    analysis["unique_titles"].add(log["input"]["title"])
                
                # Latency
                if "metadata" in log and "latency_ms" in log["metadata"]:
                    latency = log["metadata"]["latency_ms"]
                    if latency:
                        analysis["avg_latency_ms"].append(latency)
                
                # Model versions
                if "metadata" in log and "model_version" in log["metadata"]:
                    version = log["metadata"]["model_version"]
                    analysis["model_versions"][version] += 1
        
        # Convert set to count
        analysis["unique_titles"] = len(analysis["unique_titles"])
        
        # Calculate average latency
        if analysis["avg_latency_ms"]:
            analysis["avg_latency_ms"] = sum(analysis["avg_latency_ms"]) / len(analysis["avg_latency_ms"])
        else:
            analysis["avg_latency_ms"] = None
        
        return analysis




