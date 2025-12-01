import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model as load_keras_model

from src.config import MODEL_DIR, MODEL_LGBM_DIR, MODEL_LSTM_DIR, PROCESSED_DIR
from src.features.feature_engineering import (
    add_rolling_features,
    add_time_features,
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

    # Ensure `target_date` is a pandas Timestamp before arithmetic
    end_date = pd.to_datetime(target_date)
    start_date = end_date - pd.Timedelta(days=lookback_days)

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
    df = add_rolling_features(df)
    df = log_transform_features(df)
    df = add_time_features(df)
    df = df.drop(columns=["date", "county", "sitename", "status", "aqi", "season"])

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


# === LSTM ===


@st.cache_resource
def load_lstm_model(sitename):
    """Load the trained LSTM model for a specific station."""
    model_path = MODEL_LSTM_DIR / f"lstm_{sitename}.keras"

    if not model_path.exists():
        st.error(f"Model not found: {model_path}")
        return None

    return load_keras_model(model_path)


@st.cache_resource
def load_lstm_scaler(sitename):
    """Load the MinMaxScaler for a specific station."""
    scaler_path = MODEL_LSTM_DIR / f"lstm_scaler_{sitename}.pkl"

    if not scaler_path.exists():
        st.error(f"Scaler not found: {scaler_path}")
        return None

    return joblib.load(scaler_path)


def prepare_lstm_input(df, sitename, target_date, scaler, look_back=7):
    """
    Prepare input data for LSTM prediction (Sliding Window).
    Returns: (1, 7, 10) numpy array
    """

    # 1. Date range
    target_dt = pd.to_datetime(target_date)
    buffer_days = 30  # for rolling features
    start_dt = target_dt - pd.Timedelta(days=buffer_days)
    end_dt = target_dt - pd.Timedelta(days=1)

    # 2. Filter sitename and date
    station_df = df[df["sitename"] == sitename].sort_values("date").copy()
    mask = (station_df["date"] >= start_dt) & (station_df["date"] <= end_dt)
    input_df = station_df[mask].copy()

    if len(input_df) < look_back + 7:
        return None

    # 3. Apply Feature Engineering Pipeline
    input_df = clip_pollutants(input_df)
    input_df = handle_outliers_iqr(input_df)
    input_df = add_rolling_features(input_df)
    input_df = log_transform_features(input_df)
    input_df = add_time_features(input_df)

    # 4. Slice the last 7 days
    lstm_window_df = input_df.iloc[-look_back:].copy()

    if len(lstm_window_df) != look_back:
        return None

    # 5. Select features
    features = [
        "pm2.5",
        "pm10",
        "o3",
        "co",
        "windspeed",
        "pm2.5_rolling_3d",
        "pm2.5_rolling_7d",
        "month",
        "season_encoded",
        "aqi",
    ]

    for col in features:
        if col not in lstm_window_df.columns:
            lstm_window_df[col] = 0

    data = lstm_window_df[features].values

    # 6. Scaling
    data_scaled = scaler.transform(data)

    # 7. Reshaping
    input_seq = data_scaled.reshape(1, look_back, len(features))

    return input_seq


def predict_lstm(model, input_seq, scaler):
    """
    Predict AQI using LSTM and inverse transform the result.
    """
    # 1. Predict
    pred_scaled = model.predict(input_seq)

    # 2. Inverse transform
    n_features = scaler.n_features_in_
    dummy = np.zeros((1, n_features))

    dummy[:, -1] = pred_scaled[:, 0]

    pred_inv = scaler.inverse_transform(dummy)

    return pred_inv[0, -1]
