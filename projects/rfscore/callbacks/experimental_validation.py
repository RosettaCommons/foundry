import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder

from modelhub.callbacks.base import BaseCallback
from modelhub.utils.ddp import RankedLogger
from modelhub.utils.logging import print_df_as_table
from beartype.typing import Any
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

ranked_logger = RankedLogger(__name__, rank_zero_only=True)


def clean_and_encode(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, np.ndarray, LabelEncoder]:
    """Preprocess experimental data and encode categorical variables

    Args:
        X: Feature matrix
        y: Target vector

    Returns:
        Processed features, encoded target, and fitted label encoder
    """
    # Convert all columns to numeric where possible
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Check for missing values
    missing_count = X.isnull().sum().sum()
    if missing_count > 0:
        ranked_logger.warning(f"Found {missing_count} missing values in feature matrix. Dropping rows.")
        X = X.dropna()
        y = y.loc[X.index]

    # Encode target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X, y_encoded, label_encoder


def train_model(
    X_train: pd.DataFrame, 
    y_train: np.ndarray, 
    model_params: dict | None = None
) -> RandomForestClassifier:
    """Train Random Forest classifier

    Args:
        X_train: Training features
        y_train: Training target
        model_params: Parameters for RandomForestClassifier

    Returns:
        Trained Random Forest classifier
    """
    # Default parameters
    default_params = {
        "n_estimators": 20,
        "max_depth": 2,
        "random_state": 42,
        "class_weight": "balanced",
    }
    
    # Use provided parameters or defaults
    params = {**default_params, **(model_params or {})}
    
    # Initialize and train model
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    return model

def evaluate(
    model: RandomForestClassifier,
    X: pd.DataFrame,
    y: np.ndarray,
    label_encoder: LabelEncoder,
    split_name: str,
) -> dict[str, Any]:
    """Evaluate model on a dataset split and generate metrics from predictions
    
    Args:
        model: Trained classifier
        X: Feature matrix
        y: Target vector
        label_encoder: Fitted label encoder
        split_name: Dataset split (train, validation, test)
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Generate predictions and probabilities
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    
    # Get class index for "ACTIVE" (assuming binary classification)
    if "ACTIVE" not in label_encoder.classes_:
        ranked_logger.warning(f"Class 'ACTIVE' not found in label encoder. Cannot compute experimental metrics.")
        return {}

    active_class_index = list(label_encoder.classes_).index("ACTIVE")
    
    # Calculate metrics (binary classification)
    accuracy = accuracy_score(y, y_pred)
    auc = roc_auc_score(y == active_class_index, y_proba[:, active_class_index])
    conf_matrix = confusion_matrix(y, y_pred)
    
    # Create metrics summary DataFrame
    metrics_summary = pd.DataFrame({
        'Metric': ['Accuracy', 'ROC AUC'],
        'Value': [accuracy, auc]
    })
    
    # Print metrics summary
    print_df_as_table(metrics_summary, title=f"{split_name.upper()} SET METRICS")
    
    # Print confusion matrix
    conf_df = pd.DataFrame(
        conf_matrix, 
        index=label_encoder.classes_, 
        columns=label_encoder.classes_
    )
    conf_df.index.name = 'True'
    conf_df.columns.name = 'Predicted'
    print_df_as_table(conf_df, title=f"{split_name.upper()} CONFUSION MATRIX")
    
    # Return metrics
    return {
        "accuracy": accuracy,
        "auc": auc,
        "confusion_matrix": conf_matrix,
    }


def evaluate_model_on_all_splits(
    model: RandomForestClassifier,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    label_encoder: LabelEncoder,
    X_test: pd.DataFrame | None = None,
    y_test: np.ndarray | None = None,
) -> dict[str, dict[str, Any]]:
    """Evaluate model on train, validation, and (possibly) test splits
    
    Args:
        model: Trained classifier
        X_train: Training features
        y_train: Training target
        X_valid: Validation features
        y_valid: Validation target
        label_encoder: Original label encoder
        X_test: Test features (optional)
        y_test: Test target (optional)
        
    Returns:
        Dictionary of evaluation metrics for all datasets
    """
    # Define datasets to evaluate (train, validation are required)
    datasets = [
        ("train", X_train, y_train),
        ("validation", X_valid, y_valid)
    ]
    if X_test is not None and y_test is not None:
        datasets.append(("test", X_test, y_test))
    
    # Evaluate and store metrics for all splits (train, validation, test)
    all_metrics = {}
    for name, X, y in datasets:
        all_metrics[name] = evaluate(model, X, y, label_encoder, name)
    
    return all_metrics


def fit_and_evaluate(
    df: pd.DataFrame,
    feature_metrics: list[str],
    model_params: dict | None = None,
    labels_to_include: list[str] | None = ["ACTIVE", "INACTIVE"],
    datasets: list[str] | None = None,
) -> dict:
    """Train models for each dataset and evaluate classification performance across train, validation, and test splits
    
    Args:
        df: DataFrame containing features and targets, grouped by dataset (distinct from "split")
        feature_metrics: List of metric prefixes to use as features
        model_params: Parameters for RandomForestClassifier
        labels_to_include: List of labels to include in the target variable
        datasets: List of datasets to process (if None, process all datasets in df)
        
    Returns:
        Dictionary of evaluation metrics for all datasets and all models
    """

    # Fit a model for each dataset
    results_by_dataset = {}
    datasets_to_fit = datasets or df["dataset"].unique()
    for dataset in datasets_to_fit:
        ranked_logger.info(f"Processing dataset: {dataset}")
        
        # ... subset data for the current dataset
        dataset_df = df[df["dataset"] == dataset].copy()
        assert len(dataset_df) > 0, f"No data found for dataset {dataset}!"
        
        # ... filter out labels not in "labels_to_include"
        if labels_to_include:
            before_count = len(dataset_df)
            dataset_df = dataset_df[dataset_df["extra_info.activity_bin"].isin(labels_to_include)]
            after_count = len(dataset_df)
            if before_count > after_count:
                ranked_logger.info(f"Filtered out {before_count - after_count} samples not in {labels_to_include}. Remaining: {after_count}")
        
        # Extract target
        if "extra_info.activity_bin" not in dataset_df.columns:
            ranked_logger.warning(f"Target column 'extra_info.activity_bin' not found in dataset {dataset}. Skipping.")
            continue
            
        y = dataset_df["extra_info.activity_bin"]
        
        # Extract features
        feature_cols = dataset_df.columns[dataset_df.columns.str.startswith(tuple(feature_metrics))]
        if len(feature_cols) == 0:
            ranked_logger.warning(f"No feature columns found for dataset {dataset}. Skipping.")
            continue
            
        X = dataset_df[feature_cols]
        
        # Preprocess data
        X, y, label_encoder = clean_and_encode(X, y)
        
        train_mask = dataset_df["extra_info.set"] == "train"
        valid_mask = dataset_df["extra_info.set"] == "valid"
        test_mask = dataset_df["extra_info.set"] == "test"
        
        X_train, y_train = X[train_mask.values], y[train_mask.values]
        X_valid, y_valid = X[valid_mask.values], y[valid_mask.values]
        X_test, y_test = (X[test_mask.values], y[test_mask.values]) if test_mask.any() else (None, None)
        
        # Train model
        model = train_model(X_train, y_train, model_params)
        
        # Evaluate model
        metrics = evaluate_model_on_all_splits(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            X_test=X_test,
            y_test=y_test,
            label_encoder=label_encoder,
        )
        
        # Store results
        results_by_dataset[dataset] = {
            "model": model,
            "label_encoder": label_encoder,
            "metrics": metrics
        }
    
    return results_by_dataset


class FitAndEvaluateOnExperimentalDataCallback(BaseCallback):
    def __init__(
        self, 
        feature_metrics: list[str],
        model_params: dict | None = None,
        datasets: list[str] | None = None,
    ):
        """Callback to fit and evaluate models on experimental data
        
        Args:
            feature_metrics: List of metric prefixes to use as input features
            model_params: Parameters for RandomForestClassifier
            datasets: List of datasets to process (if None, process all datasets in df)
        """
        super().__init__()
        self.feature_metrics = feature_metrics
        self.model_params = model_params
        self.datasets = datasets

    def on_validation_epoch_end(self, trainer: Any):
        # Only fit and evaluate on experimental data for the global zero rank
        if not trainer.fabric.is_global_zero:
            return

        # Check if validation results are available
        assert hasattr(trainer, "validation_results_path"), "Results path not found! Ensure that StoreValidationMetricsInDFCallback is called first."
        
        # Load validation results
        df = pd.read_csv(trainer.validation_results_path)

        # Subset to current epoch
        current_epoch = trainer.state["current_epoch"]
        df = df[df["epoch"] == current_epoch]
        
        # Fit and evaluate models
        try:
            results = fit_and_evaluate(
                df=df,
                feature_metrics=self.feature_metrics,
                model_params=self.model_params,
                datasets=self.datasets,
            )

            # Log to Fabric
            for dataset, result in results.items():
                for split, metrics in result["metrics"].items():
                    for metric in ["accuracy", "auc"]:
                        trainer.fabric.log_dict(
                            {
                                f"val/exp/{dataset}/{split}/{metric}": metrics[metric]
                            },
                            step=trainer.state["current_epoch"]
                        )
        except ValueError as e:
            ranked_logger.error(f"Error during experimental model fitting/evaluation: {e}")
