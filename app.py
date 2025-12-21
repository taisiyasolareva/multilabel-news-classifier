"""
HuggingFace Spaces entry point for News Classification API.

This file is required by HuggingFace Spaces. It imports and exposes the FastAPI app
from api/main.py. Spaces will automatically run this with uvicorn.
"""
import os
import sys
from pathlib import Path

# Ensure the app directory is in Python path
app_dir = Path(__file__).parent
sys.path.insert(0, str(app_dir))

# Import the FastAPI app from api.main
from api.main import app

# HuggingFace Spaces expects the app to be accessible via the 'app' variable
# For Docker Spaces, we need to run uvicorn directly when executed
__all__ = ["app"]

if __name__ == "__main__":
    import uvicorn
    # HuggingFace Spaces sets PORT env var (typically 7860)
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(
        "app:app",  # Use string reference so it works with app variable
        host="0.0.0.0",
        port=port,
        log_level="info"
    )

