# Comprehensive Testing Plan for Credit Card Fraud Detection Project

## Tasks

- [ ] Add unit tests for MLflowManager in src/utils/mlflow_manager.py
  - Test experiment creation and setup
  - Test start run methods (training, ab_test, comparison, monitoring, retraining)
  - Test model logging and metadata logging
  - Test model version comparison and performance history retrieval
  - Test experiment run retrieval, export and cleanup
  - Use mocks to isolate MLflow dependencies

- [ ] Add CLI tests for scripts/mlflow_dashboard.py
  - Test commands: summary, registry, compare, export, cleanup, history
  - Use pytest capturing or subprocess to verify output correctness
  - Test error and edge cases

- [ ] Consider adding integration tests for end-to-end data pipeline with model registry
  - Exercise pipeline runs that log models to MLflow
  - Verify model registry updates and metadata

- [ ] Ensure pytest runs all tests including new ones
- [ ] Integrate coverage reporting to identify gaps

## Next Steps

- Implement unit tests for MLflowManager class
- Implement CLI tests for MLflow dashboard script
