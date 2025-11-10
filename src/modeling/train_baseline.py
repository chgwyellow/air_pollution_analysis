from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.config import MODEL_DIR, PROCESSED_DIR
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

    # Establish X
    X = df.drop(columns=["date", "county", "sitename", "status", "aqi", "season"])

    # Fill the na with 0
    X = X.fillna(0)

    # Check the X and y shape
    info(f"Features shape: {X.shape}, Target shape: {y.shape}")

    return (X, y)


# 3️⃣ 資料分割
def split_train_test(
    X, y, test_size: float = 0.2, random_state: int = 42, show_info: bool = True
) -> dict[str, any]:
    """
    Split dataset into training and testing sets for model evaluation.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed used by the random number generator.
        show_info (bool): Whether to print dataset size info.

    Returns:
        dict: A dictionary containing X_train, X_test, y_train, y_test.
    """

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size, random_state
        )

        if show_info:
            success(
                f"Data successfully split! "
                f"Train: {X_train.shape}, Test: {X_test.shape}"
            )

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }
    except Exception as e:
        error(f"Train-test split failed: {e}")
        return {}


# 4️⃣ 模型訓練
def train_baseline_model(data_split: dict) -> tuple[LinearRegression, dict]:
    """
    Train and evaluate a simple Linear Regression model.
    """


# 5️⃣ 模型評估
def evaluate_model(model, X_test, y_test):
    """Calculate MAE, RMSE, and R² metrics."""
    ...


# 6️⃣ 模型儲存（可選）
def save_model(model, model_path: Path):
    """Save trained model to /models directory."""
    ...
