"""Complete MLOps pipeline with monitoring and deployment."""

import argparse
from pathlib import Path

from scripts.run_monitoring import run_monitoring_pipeline
from scripts.deploy_bento import build_and_deploy_bento
from src.utils.logger import logger


def run_complete_pipeline(mode: str = "all"):
    """Run complete MLOps pipeline."""
    
    if mode in ["all", "monitoring"]:
        logger.info("Running monitoring pipeline...")
        
        # Use sample data paths
        reference_data = "data/raw/creditcard.csv"
        current_data = "data/raw/creditcard.csv"  # In production, this would be new data
        
        if Path(reference_data).exists():
            monitoring_results = run_monitoring_pipeline(reference_data, current_data)
            logger.info("Monitoring completed")
        else:
            logger.warning("Reference data not found, skipping monitoring")
    
    if mode in ["all", "deploy"]:
        logger.info("Running BentoML deployment...")
        
        try:
            deployment_results = build_and_deploy_bento()
            logger.info(f"Deployment completed: {deployment_results}")
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
    
    logger.info("MLOps pipeline completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MLOps pipeline")
    parser.add_argument("--mode", choices=["all", "monitoring", "deploy"], 
                       default="all", help="Pipeline mode")
    
    args = parser.parse_args()
    run_complete_pipeline(args.mode)