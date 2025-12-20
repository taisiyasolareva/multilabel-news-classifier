"""Script to start the FastAPI server."""

import argparse
import uvicorn
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start FastAPI server")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Log level"
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    model_path = Path("models/best_model.pt")
    if not model_path.exists():
        print(f"Warning: Model file not found at {model_path}")
        print("API will start but model must be loaded via /model/reload endpoint")
    
    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level=args.log_level,
    )

