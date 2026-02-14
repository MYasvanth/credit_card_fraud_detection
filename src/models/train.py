from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd
import numpy as np
import mlflow

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

from ..features.scaler import FeatureScaler
from ..features.smote import SMOTETransformer
from ..utils.constants import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI
from ..utils.logger import logger

def train_model(X_train, y_train, model_type="random_forest", use_smote=True, **kwargs):
    """Train a model with SMOTE and class balancing."""
    
    # Apply SMOTE if requested
    if use_smote and len(np.unique(y_train)) > 1:
        smote = SMOTETransformer(method="smote", random_state=42)
        X_train, y_train = smote.fit_transform(X_train, y_train)
        logger.info(f"Applied SMOTE: {dict(pd.Series(y_train).value_counts())}")
    
    # Get model with optimized parameters
    if model_type == "random_forest":
        default_params = {
            "n_estimators": 200,
            "max_depth": 15,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "random_state": 42
        }
        default_params.update(kwargs)
        model = RandomForestClassifier(**default_params)
        
    elif model_type == "logistic_regression":
        default_params = {
            "C": 0.1,
            "penalty": "l2",
            "solver": "liblinear",
            "class_weight": "balanced",
            "max_iter": 2000,
            "random_state": 42
        }
        default_params.update(kwargs)
        model = LogisticRegression(**default_params)
        
    elif model_type == "svm":
        default_params = {
            "C": 1.0,
            "kernel": "rbf",
            "probability": True,
            "class_weight": "balanced",
            "random_state": 42
        }
        default_params.update(kwargs)
        model = SVC(**default_params)
        
    elif model_type == "xgboost":
        if XGBClassifier is None:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")
        
        # Calculate scale_pos_weight for class imbalance
        fraud_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        default_params = {
            "n_estimators": 200,
            "max_depth": 8,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": fraud_ratio,
            "random_state": 42
        }
        default_params.update(kwargs)
        model = XGBClassifier(**default_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model.fit(X_train, y_train)
    logger.info(f"Trained {model_type} model with {len(X_train)} samples")
    return model

def train_and_evaluate_model(data, model_type="random_forest", test_size=0.2, use_smote=True, log_mlflow=True):
    """Complete training pipeline with evaluation and MLflow logging."""
    
    if log_mlflow:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    # Prepare data
    X = data.drop("Class", axis=1)
    y = data["Class"]
    
    logger.info(f"Training {model_type} on {len(data)} samples, {y.sum()} fraud cases ({y.mean()*100:.1f}%)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Feature scaling
    scaler = FeatureScaler(method="standard")
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if log_mlflow:
        with mlflow.start_run(run_name=f"fraud_detection_{model_type}"):
            # Train model
            model = train_model(X_train_scaled, y_train, model_type, use_smote=use_smote)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_proba)
            
            # Extract fraud metrics
            fraud_precision = report.get('1', {}).get('precision', 0.0)
            fraud_recall = report.get('1', {}).get('recall', 0.0)
            fraud_f1 = report.get('1', {}).get('f1-score', 0.0)
            
            # Count fraud detection
            fraud_actual = (y_test == 1).sum()
            fraud_correct = ((y_test == 1) & (y_pred == 1)).sum()
            
            # Log parameters
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("n_samples", len(data))
            mlflow.log_param("n_features", X.shape[1])
            mlflow.log_param("fraud_ratio", y.mean())
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("use_smote", use_smote)
            mlflow.log_param("scaling_method", "standard")
            
            # Log metrics
            mlflow.log_metric("accuracy", report['accuracy'])
            mlflow.log_metric("fraud_f1", fraud_f1)
            mlflow.log_metric("fraud_precision", fraud_precision)
            mlflow.log_metric("fraud_recall", fraud_recall)
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric("fraud_cases_actual", fraud_actual)
            mlflow.log_metric("fraud_cases_correct", fraud_correct)
            
            # Log model
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=f"fraud_detection_{model_type}"
            )
            
            logger.info(f"{model_type.upper()} Results: Accuracy={report['accuracy']:.4f}, F1={fraud_f1:.4f}, AUC={roc_auc:.4f}")
            
            return model, scaler, {
                "accuracy": report['accuracy'],
                "fraud_f1": fraud_f1,
                "fraud_precision": fraud_precision,
                "fraud_recall": fraud_recall,
                "roc_auc": roc_auc,
                "fraud_detected": f"{fraud_correct}/{fraud_actual}"
            }
    else:
        # Train without MLflow
        model = train_model(X_train_scaled, y_train, model_type, use_smote=use_smote)
        return model, scaler
