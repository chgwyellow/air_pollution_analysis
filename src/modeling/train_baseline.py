import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from src.modeling.evaluate_model import evaluate_and_save
from src.utils.emoji_log import error, info, success
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
    df = df.copy()
    df.dropna(subset=["aqi"], inplace=True)
    y = df["aqi"]

    # Establish X
    X = df.drop(columns=["date", "county", "sitename", "status", "aqi", "season"])

    # Fill the na with 0
    X = X.fillna(0)

    # Check the X and y shape
    info(f"Features shape: {X.shape}, Target shape: {y.shape}")

    return (X, y)


# 3️⃣ split data
def split_train_test(
    X, y, test_size: float = 0.2, random_state: int = 42, show_info: bool = True
) -> dict[str, any]:
    """
    Split scaled dataset into training and testing sets for model evaluation.

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
            X, y, test_size=test_size, random_state=random_state
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


# 4️⃣ train model
def train_linear_model(
    data_split: dict, model_name: str, model_path: str
) -> tuple[LinearRegression, dict]:
    """
    Train Linear Regression model and evaluate performance.
    """
    X_train, X_test = data_split["X_train"], data_split["X_test"]
    y_train, y_test = data_split["y_train"], data_split["y_test"]

    model = LinearRegression()
    model.fit(X=X_train, y=y_train)

    # Evaluate
    metrics = evaluate_and_save(model, X_test, y_test, model_name, model_path)

    return model, metrics
