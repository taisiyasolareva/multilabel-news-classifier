"""
HuggingFace Spaces entry point for Russian News Classification API.

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

# HuggingFace Spaces automatically handles PORT, but we can set it here as fallback
# Spaces expects the app to be accessible via the 'app' variable
__all__ = ["app"]

if __name__ == "__main__":
    # For local testing
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

