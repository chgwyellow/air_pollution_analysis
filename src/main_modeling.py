import pandas as pd

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


def run_model_pipeline(filename: str, model_type: str = "linear"):
    df = load_cleaned_data(filename)

    df_clip = clip_pollutants(df.copy())

    df_rolling = add_rolling_features(df_clip)

    df_iqr = handle_outliers_iqr(df_rolling)

    df_log = log_transform_features(df_iqr)

    X, y = build_features(df_log)

    X_scaled = scale_features(X)

    data_split = split_train_test(X_scaled, y)

    if model_type == "linear":
        from src.modeling.train_baseline import train_linear_model

        train_linear_model(data_split)
    elif model_type == "rf":
        from src.modeling.train_rf import train_random_forest

        train_random_forest(data_split)


if __name__ == "__main__":
    run_model_pipeline("Taiwan")
