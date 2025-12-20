"""Monitoring endpoints for FastAPI."""

from typing import Dict, Optional
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from monitoring.performance_monitor import PerformanceMonitor
from monitoring.data_drift import DataDriftDetector
from monitoring.prediction_logger import PredictionLogger

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


class MonitoringResponse(BaseModel):
    """Monitoring response model."""
    status: str
    data: Dict


@router.get("/performance", response_model=MonitoringResponse)
async def get_performance_metrics(
    window: Optional[int] = None,
) -> MonitoringResponse:
    """
    Get performance metrics.
    
    Args:
        window: Number of recent predictions to analyze
        
    Returns:
        Performance metrics
    """
    try:
        monitor = PerformanceMonitor()
        metrics = monitor.get_recent_metrics(window=window)
        report = monitor.generate_report()
        
        return MonitoringResponse(
            status="success",
            data={
                "metrics": metrics,
                "report": report,
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/drift", response_model=MonitoringResponse)
async def check_data_drift(
    window: int = 100,
) -> MonitoringResponse:
    """
    Check for data drift.
    
    Args:
        window: Number of recent samples to analyze
        
    Returns:
        Drift detection results
    """
    try:
        detector = DataDriftDetector()
        ref_stats_path = "monitoring/reference_stats.json"
        if Path(ref_stats_path).exists():
            detector.load_reference_stats(ref_stats_path)
        
        has_drift, drift_info = detector.detect_drift(window_size=window)
        
        return MonitoringResponse(
            status="success",
            data={
                "has_drift": has_drift,
                "drift_info": drift_info,
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/predictions", response_model=MonitoringResponse)
async def get_prediction_logs(
    limit: int = 100,
    date: Optional[str] = None,
) -> MonitoringResponse:
    """
    Get recent prediction logs.
    
    Args:
        limit: Maximum number of logs to return
        date: Date string (YYYY-MM-DD)
        
    Returns:
        Prediction logs
    """
    try:
        logger = PredictionLogger()
        logs = logger.get_recent_logs(limit=limit, date=date)
        analysis = logger.analyze_logs(days=7)
        
        return MonitoringResponse(
            status="success",
            data={
                "logs": logs,
                "analysis": analysis,
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=MonitoringResponse)
async def monitoring_health() -> MonitoringResponse:
    """Check monitoring system health."""
    try:
        from pathlib import Path
        
        health = {
            "prediction_logger": Path("monitoring/predictions").exists(),
            "performance_monitor": Path("monitoring/performance_metrics.json").exists(),
            "drift_detector": Path("monitoring/reference_stats.json").exists(),
        }
        
        return MonitoringResponse(
            status="success",
            data={"health": health},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




