"""Deploy model using BentoML."""

import argparse
from pathlib import Path

from src.deployment.bento_builder import build_and_deploy_bento
from src.utils.logger import logger


def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description="Deploy fraud detection model with BentoML")
    parser.add_argument("--build-only", action="store_true", help="Only build, don't containerize")
    
    args = parser.parse_args()
    
    try:
        if args.build_only:
            from src.deployment.bento_builder import BentoBuilder
            builder = BentoBuilder()
            
            # Save model and build
            model_tag = builder.save_model_to_bento()
            bento_tag = builder.build_bento()
            
            logger.info(f"Build completed - Model: {model_tag}, Bento: {bento_tag}")
        else:
            # Full deployment
            result = build_and_deploy_bento()
            logger.info(f"Deployment completed: {result}")
            
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        raise


if __name__ == "__main__":
    main()