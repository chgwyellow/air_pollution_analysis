"""
The functions that features the elements for model prediction.
It aims to establish the featrues of the statistics based on the cleaned data.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import MODEL_DIR
from src.utils.emoji_log import error, save, success, warn


# -----------------------------
# 1. Limit the pollutant values
# -----------------------------
def clip_pollutants(
    df: pd.DataFrame, lower: float = 0, upper: float = 1000
) -> pd.DataFrame:
    """
    Limit pollutant values to a reasonable range (default 0–1000).
    Used for histogram normalization before modeling.
    """
    pollutant_cols = ["so2", "co", "o3", "pm10", "pm2.5", "no2", "nox", "no"]
    for col in pollutant_cols:
        if col in df.columns:
            df[col] = df[col].clip(lower=lower, upper=upper)

    success("The pollutants limit has been set.")

    return df


# -----------------------------
# 2. establish IQR
# -----------------------------
def handle_outliers_iqr(df: pd.DataFrame, columns: list[str] = None) -> pd.DataFrame:
    """
    Apply IQR-based clipping to reduce the influence of extreme outliers.
    """
    if columns is None:
        columns = ["pm2.5", "pm10", "so2", "co", "o3", "no", "no2", "nox"]

    for col in columns:
        if col in df.columns:
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower, upper)

    success("IQR has been set.")

    return df


# -----------------------------
# 3. add pollutants rolling features to the df
# -----------------------------
def add_rolling_features(
    df: pd.DataFrame, pollutants: list[str] = None, windows: list[int] = [3, 7]
) -> pd.DataFrame:
    """
    Add rolling mean features for multiple raw pollutants.

    Args:
        df (pd.DataFrame): Input DataFrame containing pollutant columns.
        pollutants (list[str]): List of pollutant column names.
        windows (list[int]): Rolling window sizes in days.

    Returns:
        pd.DataFrame: DataFrame with new rolling features.
    """

    if pollutants is None:
        pollutants = ["so2", "co", "o3", "pm10", "pm2.5", "no2", "no", "nox"]

    df = df.sort_values("date", ascending=False).copy()

    for pollutant in pollutants:
        # Skip already smoothed columns
        if any(col in pollutant.lower() for col in ["_8hr", "_avg"]):
            warn(f"{pollutant} has been skipped to roll.")
            continue

        # Skip nox if no and no2 exist
        if pollutant.lower() == "nox" and all(
            col in df.columns for col in ["no", "no2"]
        ):
            df.drop(columns=["nox"], inplace=True)
            warn(f"Skipping {pollutant} due to NO + NO2 already exist.")
            continue

        for w in windows:
            new_col = f"{pollutant}_rolling_{w}d"
            df[new_col] = df.groupby("sitename")[pollutant].transform(
                lambda x: x.rolling(w, min_periods=1).mean()
            )
    success("Rolling features added.")

    return df


# -----------------------------
# 4. smooth the pollutants skewes
# -----------------------------
def log_transform_features(df, cols=None):
    """Apply log1p transform to skewed pollutant features for modeling."""
    target_col = "aqi"

    if cols is None:
        cols = [
            col
            for col in df.columns
            if any(k in col for k in ["pm", "so2", "o3", "co", "no"])
            and col != target_col
        ]

    # only select numerical column
    numeric_cols = df[cols].select_dtypes(include=["int64", "float64"]).columns
    for c in numeric_cols:
        if c in df.columns:
            df[c] = df[c].clip(lower=0)
            df[c] = np.log1p(df[c])  # log(1+x) to avoid log(0)

    df[numeric_cols] = df[numeric_cols].astype("float32")

    success("Pollutants skewes has been smoothed.")
    return df


# -----------------------------
# 5. scale features
# -----------------------------
def scale_features(
    df: pd.DataFrame,
    scaler=None,
    exclude: list[str] | None = None,
    mode="train",
    save_: bool = True,
):
    """
    Scaling helper:
    mode="train": fit + transform, return df_scaled + new scaler
    mode="test":  transform only, return df_scaled (must pass scaler)
    """

    df_scaled = df.copy()

    exclude = exclude or []
    cols_to_scale = [col for col in df.columns if col not in exclude]

    if df.index.names is not None:
        df_scaled.reset_index(drop=True, inplace=True)

    # --- train mode ---
    if mode == "train":
        scaler = StandardScaler()
        df_scaled[cols_to_scale] = scaler.fit_transform(df_scaled[cols_to_scale])

        if save_:
            joblib.dump(scaler, MODEL_DIR / "standard_scaler.pkl")
            save("Scaler has been saved!")
        return df_scaled, scaler

    # --- test mode ---
    elif mode == "test":
        if scaler is None:
            raise ValueError("Scaler must be provided in test mode.")
        df_scaled[cols_to_scale] = scaler.transform(df_scaled[cols_to_scale])
        return df_scaled
