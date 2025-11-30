import math

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential

from src.config import FIGURE_DIR, MODEL_LSTM_DIR
from src.utils.emoji_log import done, error, info, task


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

    metrics = {"RMSE": round(rmse, 2), "MAE": round(mae, 2)}
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
