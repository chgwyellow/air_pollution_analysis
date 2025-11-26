import joblib
import pandas as pd
import streamlit as st

from src.config import MODEL_DIR, MODEL_LGBM_DIR, PROCESSED_DIR
from src.features.feature_engineering import (
    add_rolling_features,
    clip_pollutants,
    handle_outliers_iqr,
    log_transform_features,
    scale_features,
)


@st.cache_data
def load_data():
    """
    Load the full dataset (Parquet) and cache it for performance.
    """

    parquet_path = PROCESSED_DIR / "full_data.parquet"

    if not parquet_path.exists():
        st.error(f"Data not found: {parquet_path}")
        return pd.DataFrame()

    df = pd.read_parquet(parquet_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def get_station_data(df, sitename, target_date, lookback_days=7):
    """
    Filter data for a specific station and a date range (target_date - lookback).
    """

    start_date = target_date - pd.Timedelta(days=lookback_days)
    end_date = pd.to_datetime(target_date)

    # Filter by station
    station_df = df[df["sitename"] == sitename].copy()

    # Filter by date range
    period = (station_df["date"] >= start_date) & (station_df["date"] <= end_date)
    filter_df = station_df[period].copy()

    return filter_df


def process_and_predict(input_df, model):
    """
    Apply feature engineering pipeline and predict.
    """

    # Feature Engineering
    df = input_df.copy()
    df = clip_pollutants(df)
    df = handle_outliers_iqr(df)
    df = log_transform_features(df)
    df = add_rolling_features(df)

    # Scaling
    scaler = joblib.load(MODEL_DIR / "standard_scaler.pkl")
    df = scale_features(df, scaler, mode="test")

    # Select features
    feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")

    if df.empty:
        return None

    # fill missing values
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    target_row = df.iloc[[-1]][feature_names]

    prediction = model.predict(target_row)[0]
    return prediction


@st.cache_resource
def load_model():
    """Load the trained LightGBM model."""
    model_path = MODEL_LGBM_DIR / "latest.pkl"

    if not model_path.exists():
        st.error(f"Model not found: {model_path}")
        return None

    return joblib.load(model_path)
