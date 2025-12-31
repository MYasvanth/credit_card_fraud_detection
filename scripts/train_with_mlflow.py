"""Training script with MLflow integration."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # Added parent directory for src imports

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import mlflow
import mlflow.sklearn

from src.features.scaler import FeatureScaler
from src.utils.logger import logger
from src.utils.constants import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI

def main():
    # Set MLflow tracking
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    with mlflow.start_run():
        logger.info("Starting MLflow training run...")
        
        # Create synthetic data
        np.random.seed(42)
        n_samples = 2000
        
        # Generate realistic fraud detection data
        X_normal = np.random.randn(int(n_samples * 0.95), 29) * 0.5
        X_fraud = np.random.randn(int(n_samples * 0.05), 29) * 2.0 + 3.0
        
        X = np.vstack([X_normal, X_fraud])
        y = np.hstack([np.zeros(len(X_normal)), np.ones(len(X_fraud))])
        
        # Shuffle
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        feature_names = [f"V{i}" for i in range(1, 29)] + ["Amount"]
        df = pd.DataFrame(X, columns=feature_names)
        df["Class"] = y
        
        # Prepare data
        X = df.drop("Class", axis=1)
        y = df["Class"]
        
        # Log data info
        mlflow.log_param("n_samples", n_samples)
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("fraud_ratio", y.mean())
        
        logger.info(f"Data shape: {X.shape}")
        logger.info(f"Class distribution: {y.value_counts().to_dict()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        mlflow.log_param("test_size", 0.2)
        
        # Feature scaling
        logger.info("Applying feature scaling...")
        scaler = FeatureScaler(method="standard")
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        mlflow.log_param("scaling_method", "standard")
        
        # Train model (configurable type)
        model_type = getattr(__main__, 'model_type', 'random_forest')
        mlflow.log_param("model_type", model_type)
        logger.info(f"Training {model_type} model...")
        
        if model_type == "random_forest":
            model_params = {
                "n_estimators": 200,
                "max_depth": 15,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "random_state": 42,
                "class_weight": "balanced_subsample",
                "bootstrap": True,
                "oob_score": True
            }
            model = RandomForestClassifier(**model_params)
        elif model_type == "xgboost":
            try:
                from xgboost import XGBClassifier
                model_params = {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "random_state": 42,
                    "scale_pos_weight": 1
                }
                model = XGBClassifier(**model_params)
            except ImportError:
                logger.warning("XGBoost not available, falling back to RandomForest")
                model_type = "random_forest"
                model_params = {"n_estimators": 100, "random_state": 42, "class_weight": "balanced"}
                model = RandomForestClassifier(**model_params)
        elif model_type == "logistic_regression":
            model_params = {
                "C": 1.0,
                "penalty": "l2",
                "solver": "liblinear",
                "random_state": 42,
                "class_weight": "balanced",
                "max_iter": 1000
            }
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(**model_params)
        elif model_type == "svm":
            model_params = {
                "C": 1.0,
                "kernel": "rbf",
                "probability": True,
                "random_state": 42,
                "class_weight": "balanced"
            }
            from sklearn.svm import SVC
            model = SVC(**model_params)
        
        # Log model parameters
        mlflow.log_params(model_params)
        
        model.fit(X_train_scaled, y_train)
        
        # Log OOB score if available
        if hasattr(model, 'oob_score_'):
            mlflow.log_metric("oob_score", model.oob_score_)
        
        # Make predictions
        logger.info("Making predictions...")
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)
        
        # Evaluate model
        logger.info("Evaluating model...")
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Calculate metrics for logging
        fraud_actual = sum(y_test)
        fraud_predicted = sum(y_pred)
        fraud_correct = sum((y_test == 1) & (y_pred == 1))
        
        # Log metrics
        mlflow.log_metric("accuracy", report['accuracy'])
        
        if '1' in report:
            mlflow.log_metric("fraud_f1", report['1']['f1-score'])
            mlflow.log_metric("fraud_precision", report['1']['precision'])
            mlflow.log_metric("fraud_recall", report['1']['recall'])
        else:
            mlflow.log_metric("fraud_f1", 0.0)
            mlflow.log_metric("fraud_precision", 0.0)
            mlflow.log_metric("fraud_recall", 0.0)
        
        # Additional metrics
        mlflow.log_metric("fraud_cases_actual", fraud_actual)
        mlflow.log_metric("fraud_cases_predicted", fraud_predicted)
        mlflow.log_metric("fraud_cases_correct", fraud_correct)
        
        # Log model
        model_uri = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="fraud_detection_model"
        ).model_uri
        
        # Save artifacts locally
        Path("models").mkdir(exist_ok=True)
        joblib.dump(model, "models/mlflow_fraud_model.pkl")
        scaler.save("mlflow_fraud_scaler")
        
        # Feature importance (only for tree-based models)
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            feature_importance.to_csv("models/mlflow_feature_importance.csv", index=False)
            mlflow.log_artifact("models/mlflow_feature_importance.csv")
        elif hasattr(model, 'coef_'):
            # For linear models, use coefficients
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'coefficient': model.coef_[0]
            }).sort_values('coefficient', key=abs, ascending=False)
            
            feature_importance.to_csv("models/mlflow_feature_importance.csv", index=False)
            mlflow.log_artifact("models/mlflow_feature_importance.csv")
        
        print("\n" + "="*60)
        print("MLFLOW TRAINING COMPLETED")
        print("="*60)
        print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")
        print(f"Dataset Size: {n_samples} samples")
        print(f"Test Accuracy: {report['accuracy']:.4f}")
        
        if '1' in report:
            print(f"Fraud F1 Score: {report['1']['f1-score']:.4f}")
            print(f"Fraud Precision: {report['1']['precision']:.4f}")
            print(f"Fraud Recall: {report['1']['recall']:.4f}")
        
        print(f"Fraud Cases - Actual: {fraud_actual}, Predicted: {fraud_predicted}, Correct: {fraud_correct}")
        print("="*60)
        
        logger.info("MLflow training completed successfully")
        
        return model, scaler, report

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train fraud detection models')
    parser.add_argument('--model', default='random_forest', 
                       choices=['random_forest', 'logistic_regression', 'xgboost', 'svm'],
                       help='Model type to train')
    args = parser.parse_args()
    
    # Update model_type globally
    import __main__
    __main__.model_type = args.model
    
    try:
        model, scaler, metrics = main()
        print("\nSUCCESS: MLflow training completed!")
        print("View results: mlflow ui")
    except Exception as e:
        logger.error(f"MLflow training failed: {e}")
        import traceback
        traceback.print_exc()