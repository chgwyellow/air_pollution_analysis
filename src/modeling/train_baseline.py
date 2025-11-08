from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from config import MODEL_DIR, PROCESSED_DIR
from src.utils.emoji_log import error, info, success, warn
from src.utils.IO_file import name_check, open_csv


# 1️⃣ Load data
def load_cleaned_data(filename: str) -> pd.DataFrame:
    """Load processed air quality data from CSV."""
    path = name_check(filename)
    df = open_csv(path)
    # Transfer the date column to datetime type
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d %H:%M:%S", errors="coerce")

    return df


# 2️⃣ Choose features and targets
def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select features (X) and target (y) for modeling."""
    # The target is aqi
    df.dropna(subset=["aqi"], inplace=True)
    y = df["aqi"]

    # Features columns
    features_cols = [
        "so2",
        "co",
        "o3",
        "pm10",
        "pm2.5",
        "no2",
        "no",
        "nox",
        "windspeed",
    ] + [col for col in df.columns if "rolling" in col]

    # Filter the actual existing columns
    features_cols = [col for col in features_cols if col in df.columns]

    # Establish X
    X = df[features_cols].copy()

    # Fill the na with 0
    X = X.fillna(0)

    # Check the X and y shape
    info(f"Features shape: {X.shape}, Target shape: {y.shape}")

    return (X, y)


# 3️⃣ 資料分割
def split_data(X, y, test_size=0.2, random_state=42):
    """Split dataset into training and testing sets."""
    ...


# 4️⃣ 模型訓練
def train_model(X_train, y_train):
    """Train baseline Linear Regression model."""
    ...


# 5️⃣ 模型評估
def evaluate_model(model, X_test, y_test):
    """Calculate MAE, RMSE, and R² metrics."""
    ...


# 6️⃣ 模型儲存（可選）
def save_model(model, model_path: Path):
    """Save trained model to /models directory."""
    ...
