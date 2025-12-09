import json
import math

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential

from src.config import (
    FIGURE_DIR,
    FONT_SIZE,
    MODEL_LSTM_DIR,
    PROCESSED_DIR,
    RESULT_DIR,
    TITLE_FONT_SIZE,
)
from src.features.feature_engineering import (
    add_rolling_features,
    add_time_features,
    clip_pollutants,
    handle_outliers_iqr,
    log_transform_features,
)
from src.utils.emoji_log import done, error, info, task, warn


def prepare_lstm_data(
    df: pd.DataFrame, sitename: str, look_back: int = 7, train_split: float = 0.8
):
    """
    Filter data for a station, scale it, and create sliding window sequences.
    """
    info(f"Preparing data for {sitename}")

    # 1. Filter & sort
    df_station = df[df["sitename"] == sitename].sort_values("date").copy()
    if len(df_station) == 0:
        raise ValueError(error(f"No data found for station: {sitename}"))

    # 2. Select features
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

    data = df_station[features].values

    # 3. Split Train/Test (chronological)
    train_size = int(len(data) * train_split)
    train_data, test_data = data[:train_size], data[train_size:]

    # 4. scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    # 5. Sliding window
    def create_dataset(dataset):
        X, y = [], []
        for i in range(len(dataset) - look_back):
            X.append(dataset[i : i + look_back])
            y.append(dataset[i + look_back, -1])
        return np.array(X), np.array(y)

    X_train, y_train = create_dataset(train_scaled)
    X_test, y_test = create_dataset(test_scaled)

    # Save scaler for inference
    scaler_path = MODEL_LSTM_DIR / f"lstm_scaler_{sitename}.pkl"
    joblib.dump(scaler, scaler_path)
    info(f"Scaler saved to {scaler_path}")

    return X_train, y_train, X_test, y_test, scaler


def prepare_unified_lstm_data(
    df: pd.DataFrame, look_back: int = 14, train_split: float = 0.8
):
    """
    Prepare unified LSTM data
    """
    df = df.sort_values(["sitename", "date"], ascending=True).copy()

    # Convert the sitename to numbers
    le = LabelEncoder()
    df["sitename_encoded"] = le.fit_transform(df["sitename"])

    features = [
        "pm2.5",
        "pm10",
        "o3",
        "co",
        "so2",
        "no2",
        "windspeed",
        "winddirec",
        "pm2.5_rolling_3d",
        "pm2.5_rolling_7d",
        "month",
        "weekday",
        "season_encoded",
        "sitename_encoded",
        "aqi",
    ]
    print(df.isna().sum())

    available_features = [f for f in features if f in df.columns]
    data = df[available_features].values

    nan_count = np.isnan(data).sum()
    if nan_count > 0:
        warn(f"Found {nan_count} NaN values, filling with 0")
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    # 3. Split Train/Test (chronological)
    train_size = int(len(data) * train_split)
    train_data, test_data = data[:train_size], data[train_size:]

    # 4. scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    # 5. Sliding window
    def create_dataset(dataset):
        X, y = [], []
        for i in range(len(dataset) - look_back):
            X.append(dataset[i : i + look_back])
            y.append(dataset[i + look_back, -1])
        return np.array(X), np.array(y)

    X_train, y_train = create_dataset(train_scaled)
    X_test, y_test = create_dataset(test_scaled)

    # Save scaler for inference
    scaler_path = MODEL_LSTM_DIR / "lstm_scaler_unified.pkl"
    joblib.dump(scaler, scaler_path)
    joblib.dump(le, MODEL_LSTM_DIR / "lstm_label_encoder.pkl")
    info(f"Scaler saved to {scaler_path}")
    info(f"LE_encoder saved to {MODEL_LSTM_DIR / "lstm_label_encoder.pkl"}")

    return X_train, y_train, X_test, y_test, scaler


def build_lstm_model(input_shape: tuple, units: int = 50, dropout_rate: float = 0.2):
    """
    Build and compile the LSTM model.
    """
    task("Start building LSTM model...")
    model = Sequential(
        [
            Input(input_shape),
            LSTM(units=units, return_sequences=True),
            Dropout(dropout_rate),
            LSTM(units=units, return_sequences=False),
            Dropout(dropout_rate),
            Dense(units=1),
        ]
    )

    model.compile(optimizer="adam", loss="mean_squared_error")
    done("LSTM model done.")
    return model


def train_lstm_model(
    model,
    X_train,
    y_train,
    sitename: str,
    epochs: int = 20,
    batch_size: int = 32,
):
    """
    Train the LSTM model with EarlyStopping and ModelCheckpoint.
    """
    task(f"Start training LSTM model for {sitename}...")

    model_path = MODEL_LSTM_DIR / f"lstm_{sitename}.keras"

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(
            filepath=str(model_path), monitor="val_loss", save_best_only=True
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1,
    )

    done(f"Training finished. Best model saved to {model_path}")
    return history


def evaluate_lstm_model(model, X_test, y_test, scaler, sitename: str):
    """
    Evaluate the LSTM model: Predict, Inverse Transform, Calculate Metrics, and Plot.
    """
    task(f"Evalutating LSTM model for {sitename}")

    # 1. Predict
    y_pred_scaled = model.predict(X_test)

    # 2. Inverse Transform
    n_features = scaler.n_features_in_  # catch the features count automatically

    def inverse_transform_aqi(pred_aray):
        dummy = np.zeros((len(pred_aray), n_features))
        dummy[:, -1] = pred_aray[:, 0]
        inversed = scaler.inverse_transform(dummy)
        return inversed[:, -1]

    y_test_inv = inverse_transform_aqi(y_test.reshape(-1, 1))
    y_pred_inv = inverse_transform_aqi(y_pred_scaled)

    rmse = math.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    r2 = r2_score(y_test_inv, y_pred_inv)

    metrics = {"RMSE": round(rmse, 2), "MAE": round(mae, 2), "R2": round(r2, 4)}

    info(f"Test Metrics for {sitename}: {metrics}")

    # 4. Plotting
    plt.figure(figsize=(15, 6))
    subset_n = 300  # Get the last 300 hours

    plt.plot(y_test_inv[-subset_n:], label="Actual AQI", color="blue", alpha=0.6)
    plt.plot(y_pred_inv[-subset_n:], label="Predicted AQI", color="red", alpha=0.8)

    plt.title(f"LSTM Prediction (Last {subset_n} Hours) - Station: {sitename}")
    plt.xlabel("Time Steps")
    plt.ylabel("AQI")
    plt.legend()

    # Save plot
    plot_path = FIGURE_DIR / "lstm" / f"lstm_prediction_{sitename}.png"
    plt.savefig(plot_path)
    plt.close()

    done(f"Evaluation plot saved to {plot_path}")

    return metrics


def train_unified_lstm(data_split: dict, model_name: str, model_path):
    """
    Training unified LSTM
    """
    task(f"🚀 Training Unified LSTM: {model_name}")

    df = pd.read_parquet(PROCESSED_DIR / "full_data.parquet")

    # Apply feature engineering (same as main_modeling.py)
    df = clip_pollutants(df)
    df = handle_outliers_iqr(df)
    df = add_rolling_features(df)
    df = log_transform_features(df)
    df = add_time_features(df)

    X_train, y_train, X_test, y_test, scaler = prepare_unified_lstm_data(df)

    # Build Model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape, units=64, dropout_rate=0.3)

    # Train
    model_path = model_path / "latest.keras"

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ModelCheckpoint(
            filepath=str(model_path), monitor="val_loss", save_best_only=True
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=64,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1,
    )

    done(f"Training finished. Best model saved to {model_path}")

    # Evaluate
    y_pred_scaled = model.predict(X_test)

    # Inverse transform
    n_features = scaler.n_features_in_

    def inverse_transform_aqi(pred_aray):
        dummy = np.zeros((len(pred_aray), n_features))
        dummy[:, -1] = pred_aray[:, 0]
        inversed = scaler.inverse_transform(dummy)
        return inversed[:, -1]

    y_test_inv = inverse_transform_aqi(y_test.reshape(-1, 1))
    y_pred_inv = inverse_transform_aqi(y_pred_scaled)

    rmse = math.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    r2 = r2_score(y_test_inv, y_pred_inv)

    metrics = {
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "R2": round(r2, 4),
    }

    info(f"Unified LSTM Metrics: {metrics}")

    metrics_path = RESULT_DIR / f"{model_name}_metrics_latest.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    done(f"Metrics saved to {metrics_path}")

    # Plotting
    plt.figure(figsize=(15, 6))
    subset_n = 300  # Get the last 500 days

    plt.plot(y_test_inv[-subset_n:], label="Actual AQI", color="blue", alpha=0.6)
    plt.plot(y_pred_inv[-subset_n:], label="Predicted AQI", color="red", alpha=0.8)

    plt.title(f"Unified LSTM Prediction (Last {subset_n} Days)")
    plt.xlabel("Time Steps (Days)")
    plt.ylabel("AQI")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save plot
    plot_path = FIGURE_DIR / "lstm" / "lstm_prediction_unified.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    done(f"Evaluation plot saved to {plot_path}")

    # Visualize training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], label="Training Loss", linewidth=2)
    plt.plot(history.history["val_loss"], label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch", fontsize=FONT_SIZE)
    plt.ylabel("Loss (MSE)", fontsize=FONT_SIZE)
    plt.title("LSTM Training History", fontsize=TITLE_FONT_SIZE)
    plt.legend()
    plt.grid(True, alpha=0.3)
    history_plot_path = FIGURE_DIR / "lstm" / "lstm_training_history_unified.png"
    plt.savefig(history_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    done(f"Training history plot saved to {history_plot_path}")

    return model, metrics
