import pandas as pd
from typing import Dict, List, Any
from src.utils.logger import logger


class DataValidator:
    """Data validation class for checking data integrity."""

    def __init__(self, required_columns: List[str], target_column: str,
                 numerical_columns: List[str] = None, categorical_columns: List[str] = None):
        """
        Initialize the DataValidator.

        Args:
            required_columns: List of columns that must be present
            target_column: Name of the target column
            numerical_columns: List of numerical columns
            categorical_columns: List of categorical columns
        """
        self.required_columns = required_columns
        self.target_column = target_column
        self.numerical_columns = numerical_columns or []
        self.categorical_columns = categorical_columns or []

    def validate(self, df: pd.DataFrame) -> tuple[bool, Dict[str, Any]]:
        """
        Validate data integrity by checking for missing values, data types, and schema consistency.

        Args:
            df: Input DataFrame to validate.

        Returns:
            Tuple of (is_valid, validation_report)
        """
        logger.info("Starting data validation")
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "issues": [],  # For compatibility with step
            "summary": {}
        }

        # Check if DataFrame is empty
        if df.empty:
            validation_results["is_valid"] = False
            validation_results["errors"].append("DataFrame is empty")
            validation_results["issues"].append("DataFrame is empty")
            logger.error("DataFrame is empty")
            return False, validation_results

        # Check for required columns
        missing_columns = [col for col in self.required_columns if col not in df.columns]
        if missing_columns:
            validation_results["is_valid"] = False
            error_msg = f"Missing required columns: {missing_columns}"
            validation_results["errors"].append(error_msg)
            validation_results["issues"].append(error_msg)
            logger.error(error_msg)
            return False, validation_results

        # Check for missing values
        missing_values = df.isnull().sum()
        total_missing = missing_values.sum()
        if total_missing > 0:
            warning_msg = f"Found {total_missing} missing values"
            validation_results["warnings"].append(warning_msg)
            validation_results["issues"].append(warning_msg)
            logger.warning(f"{warning_msg} in columns: {missing_values[missing_values > 0].index.tolist()}")

        # Check target column data type
        if self.target_column in df.columns and not df[self.target_column].dtype in ['int64', 'int32', 'float64']:
            warning_msg = f"Target column '{self.target_column}' is not numeric"
            validation_results["warnings"].append(warning_msg)
            validation_results["issues"].append(warning_msg)
            logger.warning(warning_msg)

        # Check for duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            warning_msg = f"Found {duplicates} duplicate rows"
            validation_results["warnings"].append(warning_msg)
            validation_results["issues"].append(warning_msg)
            logger.warning(warning_msg)

        # Summary
        validation_results["summary"] = {
            "shape": df.shape,
            "columns": len(df.columns),
            "total_missing": int(total_missing),
            "duplicates": int(duplicates)
        }

        logger.info("Data validation completed")
        return validation_results["is_valid"], validation_results


def validate_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Legacy function for backward compatibility.
    """
    validator = DataValidator(
        required_columns=['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                          'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                          'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount', 'Class'],
        target_column='Class'
    )
    return validator.validate(df)[1]
