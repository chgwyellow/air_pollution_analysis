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

from src.features.feature_engineering import (
    add_rolling_features,
    clip_pollutants,
    handle_outliers_iqr,
    log_transform_features,
    scale_features,
)
from src.modeling.model_registry import MODEL_REGISTRY
from src.modeling.train_baseline import (
    build_features,
    load_cleaned_data,
    split_train_test,
)
from src.utils.emoji_log import done, error


def run_model_pipeline(filename: str, model_type: str = "linear"):
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
            - "rf_tuned" → Tuned Random Forest (RandomSearch + GridSearch)

    Returns
    -------
    model : trained model
    metrics : dict
        Evaluation results including MAE, RMSE, R².
    """

    # === 1. Load cleaned data ===
    df = load_cleaned_data(filename)

    # === 2. Feature Engineering Pipeline ===
    # Step 1: Clip pollutants to reasonable range
    df_clip = clip_pollutants(df.copy())

    # Step 2: Add rolling mean features (3-day, 7-day)
    df_rolling = add_rolling_features(df_clip)

    # Step 3: Reduce extreme outliers using IQR clipping
    df_iqr = handle_outliers_iqr(df_rolling)

    # Step 4: Log-transform skewed pollutants
    df_log = log_transform_features(df_iqr)

    # === 3. Feature selection & scaling ===
    X, y = build_features(df_log)

    # Scale all features (except excluded)
    X_scaled = scale_features(X, save_=False)

    # Split into train/test sets
    data_split = split_train_test(X_scaled, y)

    # === 4. Model selection via Model Registry ===
    if model_type not in MODEL_REGISTRY:
        raise ValueError(error(f"Unsupported model type: {model_type}"))

    train_func = MODEL_REGISTRY[model_type]

    # === 5. Train + Save model & metrics ===
    model, metrics = train_func(data_split)

    done(f"Pipeline completed successfully! ({model_type})")

    return model, metrics


if __name__ == "__main__":
    run_model_pipeline("Taiwan")
