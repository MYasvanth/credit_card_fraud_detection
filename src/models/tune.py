from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from typing import Dict, Any, Tuple
import pandas as pd
from sklearn.base import BaseEstimator
import optuna


class HyperparameterTuner:
    """Hyperparameter tuning class supporting GridSearch and Optuna."""

    def __init__(self, model_type: str = "random_forest", param_grid: Dict[str, Any] = None,
                 cv_folds: int = 5, scoring: str = "f1", n_jobs: int = -1, random_state: int = 42):
        """
        Initialize the tuner.

        Args:
            model_type: Type of model to tune
            param_grid: Parameter grid for GridSearch
            cv_folds: Number of CV folds
            scoring: Scoring metric
            n_jobs: Number of parallel jobs
            random_state: Random state
        """
        self.model_type = model_type
        self.param_grid = param_grid or self._get_default_param_grid()
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _get_default_param_grid(self) -> Dict[str, Any]:
        """Get default parameter grid based on model type."""
        if self.model_type == "random_forest":
            return {
                'n_estimators': [100, 200],
                'max_depth': [10, 20],
                'min_samples_split': [2, 5]
            }
        elif self.model_type == "logistic_regression":
            return {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
        elif self.model_type == "svm":
            return {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        return {}

    def _get_base_model(self) -> BaseEstimator:
        """Get base model instance."""
        if self.model_type == "random_forest":
            return RandomForestClassifier(random_state=self.random_state)
        elif self.model_type == "logistic_regression":
            return LogisticRegression(random_state=self.random_state)
        elif self.model_type == "svm":
            return SVC(random_state=self.random_state, probability=True)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def tune(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[BaseEstimator, Dict[str, Any]]:
        """
        Perform hyperparameter tuning using GridSearchCV.

        Args:
            X_train: Training features
            y_train: Training target

        Returns:
            Best model and tuning results
        """
        import numpy as np
        
        # Convert to numpy arrays to avoid MaskedArray issues
        X_array = np.asarray(X_train)
        y_array = np.asarray(y_train)
        
        model = self._get_base_model()
        grid_search = GridSearchCV(
            model, self.param_grid, cv=self.cv_folds,
            scoring=self.scoring, n_jobs=self.n_jobs
        )
        grid_search.fit(X_array, y_array)

        # Clean results to avoid serialization issues
        tuning_results = {
            'best_params': dict(grid_search.best_params_),
            'best_score': float(grid_search.best_score_),
            'n_splits': self.cv_folds,
            'scoring': self.scoring
        }

        return grid_search.best_estimator_, tuning_results

    def evaluate_model(self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Evaluate model on given data.

        Args:
            model: Trained model
            X: Features
            y: Target

        Returns:
            F1 score
        """
        import numpy as np
        
        # Convert to numpy arrays
        X_array = np.asarray(X)
        y_array = np.asarray(y)
        
        y_pred = model.predict(X_array)
        score = f1_score(y_array, y_pred, average='weighted')
        return float(score)

    def tune_with_optuna(self, X_train: pd.DataFrame, y_train: pd.Series, n_trials: int = 50) -> Tuple[BaseEstimator, Dict[str, Any]]:
        """
        Perform hyperparameter tuning using Optuna.

        Args:
            X_train: Training features
            y_train: Training target
            n_trials: Number of optimization trials

        Returns:
            Best model and tuning results
        """
        def objective(trial):
            if self.model_type == "random_forest":
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 10, 50),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4)
                }
                model = RandomForestClassifier(**params, random_state=self.random_state)
            elif self.model_type == "logistic_regression":
                params = {
                    'C': trial.suggest_float('C', 0.01, 100.0, log=True),
                    'penalty': trial.suggest_categorical('penalty', ['l1', 'l2'])
                }
                model = LogisticRegression(**params, solver='liblinear', random_state=self.random_state)
            else:
                raise ValueError(f"Optuna tuning not supported for {self.model_type}")

            scores = cross_val_score(model, X_train, y_train, cv=self.cv_folds, scoring=self.scoring)
            return scores.mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        # Train best model
        best_params = study.best_params
        if self.model_type == "random_forest":
            best_model = RandomForestClassifier(**best_params, random_state=self.random_state)
        elif self.model_type == "logistic_regression":
            best_model = LogisticRegression(**best_params, solver='liblinear', random_state=self.random_state)

        best_model.fit(X_train, y_train)

        tuning_results = {
            'best_params': best_params,
            'best_score': study.best_value,
            'n_trials': n_trials
        }

        return best_model, tuning_results


def tune_model(model, X_train, y_train, param_grid):
    """Legacy function for backward compatibility."""
    tuner = HyperparameterTuner()
    tuner.param_grid = param_grid
    return tuner.tune(X_train, y_train)[0]
