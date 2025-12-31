# Quick Start Guide

## 🚀 Run Everything at Once

```bash
# Complete demo of all components
python demo_pipeline.py
```

## 🎯 Individual Components

### 1. Model Serving
```bash
# Start REST API server
python run_components.py serve

# Test the API
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": {"V1": -1.36, "V2": -0.07, "Amount": 149.62}}'
```

### 2. A/B Testing
```bash
# Compare two models
python run_components.py ab-test --model-a model_v1 --model-b model_v2
```

### 3. Model Comparison
```bash
# Compare multiple models with visualizations
python run_components.py compare --models model_v1 model_v2 model_v3
```

### 4. Automated Retraining
```bash
# Trigger retraining based on performance drop
python run_components.py retrain --current-f1 0.82 --baseline-f1 0.85

# Trigger retraining based on data drift
python run_components.py retrain --current-f1 0.85 --baseline-f1 0.85 --drift-detected
```

## 📊 View Results

```bash
# MLflow UI for experiment tracking
mlflow ui

# API documentation
python scripts/run_pipeline.py api
# Visit: http://localhost:8000/docs
```

## 🔧 Prerequisites

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download data:**
   - Place `creditcard.csv` in `data/raw/`
   - Get from: https://www.kaggle.com/mlg-ulb/creditcardfraud

3. **Train initial model:**
   ```bash
   python scripts/train_with_mlflow.py
   ```

## 📁 Key Files

- `run_components.py` - Main runner for all components
- `demo_pipeline.py` - Complete pipeline demo
- `USAGE_GUIDE.md` - Detailed documentation
- `scripts/serve_model.py` - Model serving
- `scripts/run_ab_test.py` - A/B testing
- `src/utils/model_comparison.py` - Model comparison utilities
- `src/pipelines/retraining_pipeline.py` - Automated retraining

## 🎯 Common Use Cases

**Development:**
```bash
python demo_pipeline.py  # See everything in action
```

**Production Monitoring:**
```bash
python run_components.py retrain --current-f1 0.82 --baseline-f1 0.85
```

**Model Deployment:**
```bash
python run_components.py serve --model_name best_model --port 8000
```

**Performance Analysis:**
```bash
python run_components.py compare --models prod_model new_model
```