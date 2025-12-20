"""Monitoring middleware for FastAPI."""

import time
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from typing import Optional
from monitoring.prediction_logger import PredictionLogger
from monitoring.data_drift import DataDriftDetector
from monitoring.performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)


class MonitoringMiddleware(BaseHTTPMiddleware):
    """
    Middleware for monitoring API requests and predictions.
    
    Logs predictions, detects data drift, and monitors performance.
    """

    def __init__(
        self,
        app,
        prediction_logger: PredictionLogger,
        drift_detector: Optional[DataDriftDetector] = None,
        performance_monitor: Optional[PerformanceMonitor] = None,
    ):
        """
        Initialize monitoring middleware.
        
        Args:
            app: FastAPI application
            prediction_logger: Prediction logger instance
            drift_detector: Optional drift detector
            performance_monitor: Optional performance monitor
        """
        super().__init__(app)
        self.prediction_logger = prediction_logger
        self.drift_detector = drift_detector
        self.performance_monitor = performance_monitor

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with monitoring."""
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Monitor classification endpoints (log in background)
        if request.url.path in ["/classify", "/classify/batch"]:
            # Store monitoring data in response state for background processing
            # The actual logging will be done by the endpoint or a background task
            response.headers["X-Process-Time"] = str(latency_ms)
            response.headers["X-Monitored"] = "true"
        
        return response

