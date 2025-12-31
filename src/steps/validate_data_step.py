"""ZenML step for data validation."""

from typing import Tuple, Dict, Any
import pandas as pd
from zenml import step
from zenml.logger import get_logger

from ..data.validator import DataValidator
from ..utils.constants import TARGET_COLUMN

logger = get_logger(__name__)


@step
def validate_data_step(
    data: pd.DataFrame,
    validation_config: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validate input data using comprehensive validation rules.
    
    Args:
        data: Input DataFrame to validate
        validation_config: Configuration for validation rules
        
    Returns:
        Tuple of validated data and validation report
    """
    validator = DataValidator(
        required_columns=validation_config.get("required_columns", []),
        target_column=validation_config.get("target_column", TARGET_COLUMN),
        numerical_columns=validation_config.get("numerical_columns", []),
        categorical_columns=validation_config.get("categorical_columns", [])
    )
    
    # Perform validation
    is_valid, validation_report = validator.validate(data)
    
    if not is_valid:
        logger.warning("Data validation failed with issues:")
        for issue in validation_report.get("issues", []):
            logger.warning(f"- {issue}")
    else:
        logger.info("Data validation passed successfully")
    
    # Log validation statistics
    logger.info(f"Data shape: {data.shape}")
    logger.info(f"Missing values: {data.isnull().sum().sum()}")
    logger.info(f"Duplicate rows: {data.duplicated().sum()}")
    
    return data, validation_report