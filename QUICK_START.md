# Quick Start Guide

## 🚀 Production Workflow

### 1. Train Model on Real Data
```bash
# Train Random Forest on 284K real transactions
python scripts/train_with_mlflow.py --model random_forest

# Expected: F1=0.84, Precision=89%, Recall=80%
# Training time: ~4-5 minutes
```

### 2. View Training Results
```bash
# Start MLflow UI
mlflow ui

# Visit: http://localhost:5000
# Check experiment: credit_card_fraud_detection
```

### 3. Serve Model via API
```bash
# Start production API server
python scripts/serve_model.py

# API available at: http://localhost:5000
```

### 4. Make Predictions
```bash
# Test fraud detection
curl -X POST "http://localhost:5000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": {"V1": -1.36, "V2": -0.07, "Amount": 149.62}}'
```

---

## 🎯 Advanced Features

### 1. A/B Testing
```bash
# Compare two models
python run_components.py ab-test --model-a model_v1 --model-b model_v2
```

### 2. Model Comparison
```bash
# Compare multiple models with visualizations
python run_components.py compare --models model_v1 model_v2 model_v3
```

### 3. Automated Retraining
```bash
# Trigger retraining based on performance drop
python run_components.py retrain --current-f1 0.82 --baseline-f1 0.85

# Trigger retraining based on data drift
python run_components.py retrain --current-f1 0.85 --baseline-f1 0.85 --drift-detected
```

---

## 📊 Performance Metrics

**Current Model (Random Forest on Real Data):**
- Dataset: 284,807 transactions
- Fraud F1 Score: **0.8432**
- Precision: 89.66% (low false positives)
- Recall: 79.59% (catches 80% of fraud)
- Training Time: ~4.5 minutes

See `BASELINE_METRICS.md` for detailed analysis.

---

## 🔧 Setup Instructions

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download data:**
   - Place `creditcard.csv` in `data/raw/`
   - Get from: https://www.kaggle.com/mlg-ulb/creditcardfraud

3. **Train production model:**
   ```bash
   # Random Forest (recommended)
   python scripts/train_with_mlflow.py --model random_forest
   
   # Other options: logistic_regression, xgboost, svm
   python scripts/train_with_mlflow.py --model xgboost
   ```

---

## 📁 Key Files

**Documentation:**
- `BASELINE_METRICS.md` - **Production performance metrics**
- `TRAINING_COMPLETED.md` - Training results summary
- `USAGE_GUIDE.md` - Detailed usage guide
- `README.md` - Project overview

**Scripts:**
- `scripts/train_with_mlflow.py` - Train models on real data
- `scripts/serve_model.py` - Production API server
- `scripts/run_ab_test.py` - A/B testing
- `scripts/mlflow_dashboard.py` - MLflow utilities

**Source Code:**
- `src/models/train.py` - Training logic
- `src/data/loader.py` - Data loading
- `src/api/main.py` - FastAPI application
- `src/monitoring/` - Monitoring and drift detection

---

## 🎯 Common Workflows

**Train Multiple Models:**
```bash
python scripts/train_multiple_models.py
# Trains: Random Forest, Logistic Regression, XGBoost
```

**Compare Model Performance:**
```bash
# View in MLflow UI
mlflow ui

# Or use comparison script
python run_components.py compare --models model_v1 model_v2
```

**Production Deployment:**
```bash
# 1. Train best model
python scripts/train_with_mlflow.py --model random_forest

# 2. Start API server
python scripts/serve_model.py

# 3. Monitor performance
python scripts/run_monitoring.py
```

**Automated Retraining:**
```bash
# Trigger when performance drops
python run_components.py retrain --current-f1 0.82 --baseline-f1 0.85
```

---

## 📈 Next Steps

1. **Optimize Performance**: Implement custom inference for <50ms latency
2. **Add Monitoring**: Set up drift detection and alerting
3. **Deploy to Production**: Containerize with Docker
4. **Scale**: Add load balancing and auto-scaling

See `BASELINE_METRICS.md` for optimization roadmap.
