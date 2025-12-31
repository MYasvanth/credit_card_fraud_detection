# Credit Card Fraud Detection - Usage Guide

This guide explains how to run the different ML components in your fraud detection system.

## Prerequisites

1. **Activate your virtual environment:**
   ```bash
   venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure you have trained models in MLflow:**
   ```bash
   python scripts/train_with_mlflow.py
   ```

## Component Usage

### 1. Model Serving

Start a REST API server to serve your trained model for real-time predictions.

**Basic Usage:**
```bash
python run_components.py serve
```

**Advanced Usage:**
```bash
python run_components.py serve --model_name credit_card_fraud_detector --stage Production --host 0.0.0.0 --port 5000
```

**API Endpoints:**
- `GET /health` - Health check
- `POST /predict` - Single prediction
- `POST /batch_predict` - Batch predictions
- `GET /model_info` - Model information

**Example API Call:**
```bash
curl -X POST "http://localhost:5000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "features": {
         "V1": -1.359807134,
         "V2": -0.072781173,
         "V3": 2.536346738,
         "Amount": 149.62,
         "Time": 0
       }
     }'
```

### 2. A/B Testing

Compare two models by simulating traffic routing and measuring performance differences.

**Basic Usage:**
```bash
python run_components.py ab-test --model-a model_v1 --model-b model_v2
```

**Advanced Usage:**
```bash
python run_components.py ab-test \
  --model-a credit_card_fraud_detector \
  --model-b credit_card_fraud_detector_v2 \
  --test-data data/raw/creditcard.csv \
  --traffic-split 0.3 \
  --n-simulations 2000 \
  --output results/ab_test_results.json
```

**Parameters:**
- `--model-a`: Control model name
- `--model-b`: Treatment model name  
- `--traffic-split`: Fraction of traffic to model B (0.5 = 50/50)
- `--n-simulations`: Number of prediction simulations
- `--output`: Save results to JSON file

### 3. Model Comparison

Compare multiple models side-by-side with comprehensive metrics and visualizations.

**Basic Usage:**
```bash
python run_components.py compare --models model_v1 model_v2 model_v3
```

**Advanced Usage:**
```bash
python run_components.py compare \
  --models credit_card_fraud_detector model_v2 baseline_model \
  --test-data data/raw/creditcard.csv \
  --output-dir reports/comparison
```

**Outputs:**
- Detailed metrics comparison table
- ROC curves comparison plot
- Precision-Recall curves plot
- Statistical significance tests
- MLflow experiment logging

### 4. Automated Retraining Pipeline

Trigger retraining based on performance degradation or data drift detection.

**Performance-Based Retraining:**
```bash
python run_components.py retrain \
  --current-f1 0.82 \
  --baseline-f1 0.85 \
  --data-path data/raw/creditcard.csv
```

**Drift-Based Retraining:**
```bash
python run_components.py retrain \
  --current-f1 0.85 \
  --baseline-f1 0.85 \
  --drift-detected \
  --data-path data/raw/creditcard.csv \
  --quick
```

**Parameters:**
- `--current-f1`: Current model F1 score
- `--baseline-f1`: Baseline F1 score threshold
- `--drift-detected`: Flag indicating data drift
- `--quick`: Use quick training mode
- `--data-path`: Path to new training data

## Integrated Pipeline Workflows

### Complete ML Pipeline

Run the full training and evaluation pipeline:

```bash
# 1. Train initial model
python scripts/run_pipeline.py train --quick

# 2. Start model serving
python run_components.py serve &

# 3. Compare with previous models
python run_components.py compare --models credit_card_fraud_detector baseline_model

# 4. Run A/B test if needed
python run_components.py ab-test --model-a baseline_model --model-b credit_card_fraud_detector
```

### Monitoring and Retraining Workflow

```bash
# 1. Run monitoring pipeline
python scripts/run_pipeline.py monitor \
  --reference-data data/raw/creditcard.csv \
  --current-data data/raw/creditcard.csv \
  --predictions data/inference_samples/predictions.csv \
  --baseline-f1 0.85

# 2. Trigger retraining if performance degrades
python run_components.py retrain --current-f1 0.82 --baseline-f1 0.85
```

## Configuration Files

The system uses YAML configuration files in the `configs/` directory:

- `configs/model.yaml` - Model training parameters
- `configs/data.yaml` - Data processing settings
- `configs/deployment.yaml` - Deployment configurations
- `configs/monitoring.yaml` - Monitoring thresholds

## Enhanced MLflow Integration

The system uses an enhanced MLflow setup with organized experiments:

### 🎯 **Experiment Organization:**
- `credit_card_fraud_detection` - Model training
- `credit_card_fraud_ab_testing` - A/B testing results  
- `credit_card_fraud_model_comparison` - Model comparisons
- `credit_card_fraud_monitoring` - Performance monitoring
- `credit_card_fraud_retraining` - Automated retraining

### 📊 **MLflow Dashboard:**
```bash
# View experiment summary
python scripts/mlflow_dashboard.py summary

# View model registry
python scripts/mlflow_dashboard.py registry

# Compare Production vs Staging
python scripts/mlflow_dashboard.py compare

# View model performance history
python scripts/mlflow_dashboard.py history --model-name credit_card_fraud_detector
```

### 🌐 **MLflow UI:**
```bash
mlflow ui
# Access at: http://localhost:5000
```

### 📈 **Advanced Features:**
- Automatic model versioning and staging
- Performance history tracking
- Experiment cleanup and management
- Comprehensive metadata logging

## API Server Integration

### Start API Server
```bash
python scripts/run_pipeline.py api --host 0.0.0.0 --port 8000
```

### API Documentation
Visit: http://localhost:8000/docs

## Troubleshooting

### Common Issues

1. **Model not found in registry:**
   ```bash
   # Check available models
   python -c "import mlflow; client = mlflow.tracking.MlflowClient(); print([m.name for m in client.list_registered_models()])"
   ```

2. **Data file not found:**
   - Ensure `data/raw/creditcard.csv` exists
   - Download from: https://www.kaggle.com/mlg-ulb/creditcardfraud

3. **MLflow server not running:**
   ```bash
   mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
   ```

### Performance Tips

1. **Quick training mode:** Use `--quick` flag for faster iterations
2. **Batch predictions:** Use `/batch_predict` endpoint for multiple samples
3. **Model caching:** Models are cached in memory for faster serving

## Example Workflows

### Development Workflow
```bash
# 1. Train and register model
python scripts/train_with_mlflow.py

# 2. Compare with existing models
python run_components.py compare --models credit_card_fraud_detector baseline_model

# 3. Run A/B test
python run_components.py ab-test --model-a baseline_model --model-b credit_card_fraud_detector

# 4. Deploy best model
python run_components.py serve --model_name credit_card_fraud_detector
```

### Production Workflow
```bash
# 1. Monitor model performance
python scripts/run_pipeline.py monitor --reference-data ref.csv --current-data curr.csv --predictions pred.csv --baseline-f1 0.85

# 2. Trigger retraining if needed
python run_components.py retrain --current-f1 0.82 --baseline-f1 0.85 --drift-detected

# 3. Compare new model with production
python run_components.py compare --models production_model retrained_model

# 4. A/B test before full deployment
python run_components.py ab-test --model-a production_model --model-b retrained_model --traffic-split 0.1
```

## Next Steps

1. **Set up CI/CD:** Use `.github/workflows/` for automated testing and deployment
2. **Configure monitoring:** Set up automated monitoring with `monitoring/` components
3. **Scale deployment:** Use Docker configurations in `docker/` directory
4. **Add custom features:** Extend `src/features/` for domain-specific feature engineering

For more details, see the individual component documentation in the `docs/` directory.