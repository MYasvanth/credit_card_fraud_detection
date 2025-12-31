"""FastAPI Model Serving script for credit card fraud detection."""

import sys
from pathlib import Path
import uvicorn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.main import app
from src.utils.constants import API_HOST, API_PORT
from src.utils.logger import logger

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Serve FastAPI model for fraud detection')
    parser.add_argument('--model_name', default='credit_card_fraud_detector', help='Model name in registry')
    parser.add_argument('--stage', default='Production', help='Model stage to serve')
    parser.add_argument('--host', default=API_HOST, help='Host to bind to')
    parser.add_argument('--port', type=int, default=API_PORT, help='Port to bind to')

    args = parser.parse_args()

    logger.info(f"Starting FastAPI server on {args.host}:{args.port}")
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info"
    )
