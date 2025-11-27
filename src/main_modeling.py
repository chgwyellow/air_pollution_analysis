"""
Modeling Pipeline
=================

This script runs the end-to-end machine learning pipeline:

    1. Load cleaned data
    2. Apply feature engineering (clip → rolling → IQR → log transform → scaling)
    3. Select features (X) and target (y)
    4. Split into train/test sets
    5. Train model through Model Registry:
        - "linear" → Baseline Linear Regression
        - "rf" → Random Forest (default params)
        - "rf_tuned" → Random Forest with tuning (Randomized + Grid Search)
    6. Evaluate & save results (model + metrics)

You can run the whole pipeline simply by:

    run_model_pipeline("Taiwan", model_type="rf_tuned")

This makes the modeling process fully modular, reusable, and maintainable.
"""

import joblib
import pandas as pd

from src.config import MODEL_DIR, PROCESSED_DIR
from src.features.feature_engineering import (
    add_rolling_features,
    add_time_features,
    clip_pollutants,
    handle_outliers_iqr,
    log_transform_features,
    scale_features,
)
from src.modeling.model_registry import MODEL_REGISTRY
from src.modeling.train_baseline import build_features, split_train_test
from src.utils.emoji_log import done, error
from src.utils.IO_file import convert_csv_to_parquet


def run_model_pipeline(filename: str, model_type: str = "linear", sample_frac=0.5):
    """
    Run the complete model training pipeline.

    Parameters
    ----------
    filename : str
        Name of the processed CSV file (e.g., "Taiwan").
    model_type : str
        Choose the model registered in MODEL_REGISTRY:
            - "linear"   → Baseline Linear Regression
            - "rf"       → Default Random Forest
            - "rf_best"  → Best Random Forest
            - "rf_tuned" → Tuned Random Forest (RandomSearch + GridSearch)

    Returns
    -------
    model : trained model
    metrics : dict
        Evaluation results including MAE, RMSE, R².
    """

    # === 1. Smart Data Loading (CSV -> Parquet) ===
    # Define paths
    csv_path = PROCESSED_DIR / f"{filename}.csv"
    parquet_path = PROCESSED_DIR / "full_data.parquet"

    # If LightGBM, force full data (because it's fast and memory-friendly)
    if model_type == "lgbm":
        sample_frac = 1.0

    # Automatically check and convert
    if not parquet_path.exists():
        convert_csv_to_parquet(csv_path, parquet_path)

    # Read Parquet
    df = pd.read_parquet(parquet_path)

    # Sample (if needed)
    if sample_frac < 1:
        df = df.sample(frac=sample_frac, random_state=42)

    # === 2. Feature Engineering Pipeline ===
    # Step 1: Clip pollutants to reasonable range
    df = clip_pollutants(df)

    # Step 2: Reduce extreme outliers using IQR clipping
    df = handle_outliers_iqr(df)

    # Step 3: Add rolling mean features (3-day, 7-day)
    df = add_rolling_features(df)

    # Step 4: Log-transform skewed pollutants
    df = log_transform_features(df)

    # Step 5: Add temporal feature
    df = add_time_features(df)

    # === 3. Feature selection & scaling ===
    X, y = build_features(df)

    # Save the training dataset used for visualization
    df_training_clean = pd.concat([X, y], axis=1)
    df_training_clean.to_parquet(PROCESSED_DIR / "training_data_cleaned.parquet")
    joblib.dump(list(X.columns), MODEL_DIR / "feature_names.pkl")

    # Split into train/test sets
    data_split = split_train_test(X, y)
    X_train, X_test = data_split["X_train"], data_split["X_test"]

    # fit train and transform test features (except excluded)
    X_train_scaled, scaler = scale_features(X_train, mode="train", save_=True)
    X_test_scaled = scale_features(X_test, scaler=scaler, mode="test")

    data_split["X_train"] = X_train_scaled
    data_split["X_test"] = X_test_scaled

    # === 4. Model selection via Model Registry ===
    if model_type not in MODEL_REGISTRY:
        raise ValueError(error(f"Unsupported model type: {model_type}"))

    model_info = MODEL_REGISTRY[model_type]

    train_func = model_info["train_func"]
    model_name = model_info["model_name"]
    model_path = model_info["save_dir"]

    # === 5. Train + Save model & metrics ===
    model, metrics = train_func(data_split, model_name, model_path)

    done(f"Pipeline completed successfully! ({model_type})")

    return model, metrics


if __name__ == "__main__":
    run_model_pipeline("Taiwan", model_type="lgbm")
