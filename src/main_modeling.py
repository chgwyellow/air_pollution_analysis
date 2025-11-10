from src.features.feature_engineering import (
    add_rolling_features,
    clip_pollutants,
    handle_outliers_iqr,
    log_transform_features,
    scale_features,
)
from src.modeling.train_baseline import (
    build_features,
    load_cleaned_data,
    split_train_test,
)
from src.utils.emoji_log import error


def run_model_pipeline(filename: str, model_type: str = "linear"):
    df = load_cleaned_data(filename)

    # === Feature Engineering Pipeline ===
    df_clip = clip_pollutants(df.copy())
    df_rolling = add_rolling_features(df_clip)
    df_iqr = handle_outliers_iqr(df_rolling)
    df_log = log_transform_features(df_iqr)

    # === Feature Selection & Scaling ===
    X, y = build_features(df_log)
    X_scaled = scale_features(X)
    data_split = split_train_test(X_scaled, y)

    # === Model Training ===
    if model_type == "linear":
        from src.modeling.train_baseline import train_linear_model

        model, metrics = train_linear_model(data_split)
    elif model_type == "rf":
        from src.modeling.train_rf import train_random_forest

        model, metrics = train_random_forest(data_split)
    else:
        raise ValueError(error(f"Unsupported model type: {model_type}"))


if __name__ == "__main__":
    run_model_pipeline("Taiwan")
