"""BentoML model packaging and deployment utilities."""

import bentoml
import joblib
from pathlib import Path
from typing import Dict, Any

from ..utils.logger import logger
from ..utils.constants import MODELS_PATH


class BentoBuilder:
    """Build and manage BentoML models."""
    
    def __init__(self):
        self.models_path = Path(MODELS_PATH)
    
    def save_model_to_bento(self, model_name: str = "fraud_detection_model", 
                           model_version: str = None) -> str:
        """Save trained model to BentoML model store."""
        model_path = self.models_path / "best_model.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load model
        model = joblib.load(model_path)
        
        # Save to BentoML
        bento_model = bentoml.sklearn.save_model(
            model_name,
            model,
            labels={
                "framework": "sklearn",
                "task": "fraud_detection",
                "version": model_version or "latest"
            },
            metadata={
                "model_path": str(model_path),
                "features": "V1-V28,Amount"
            }
        )
        
        logger.info(f"Model saved to BentoML: {bento_model}")
        return str(bento_model)
    
    def build_bento(self, service_file: str = "service.py") -> str:
        """Build BentoML service."""
        import subprocess
        
        try:
            result = subprocess.run(
                ["bentoml", "build", "-f", "bentofile.yaml"],
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info(f"Bento built successfully: {result.stdout}")
            return result.stdout.strip()
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Bento build failed: {e.stderr}")
            raise
    
    def containerize_bento(self, bento_tag: str) -> str:
        """Containerize BentoML service."""
        import subprocess
        
        try:
            result = subprocess.run(
                ["bentoml", "containerize", bento_tag],
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info(f"Bento containerized: {result.stdout}")
            return result.stdout.strip()
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Containerization failed: {e.stderr}")
            raise


def build_and_deploy_bento():
    """Complete BentoML build and deployment pipeline."""
    builder = BentoBuilder()
    
    # Save model to BentoML
    model_tag = builder.save_model_to_bento()
    logger.info(f"Model saved: {model_tag}")
    
    # Build Bento
    bento_tag = builder.build_bento()
    logger.info(f"Bento built: {bento_tag}")
    
    # Containerize
    container_tag = builder.containerize_bento(bento_tag)
    logger.info(f"Container built: {container_tag}")
    
    return {
        "model_tag": model_tag,
        "bento_tag": bento_tag,
        "container_tag": container_tag
    }