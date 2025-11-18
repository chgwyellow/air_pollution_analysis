import json

from sklearn.ensemble import RandomForestRegressor

from src.config import RESULT_DIR
from src.modeling.evaluate_model import evaluate_and_save
from src.utils.emoji_log import done


def train_random_forest(data_split: dict, model_name: str, model_path: str, **kwargs):
    """
    Train Random Forest model and evaluate performance.
    Args:
        data_split (dict): Train/test split dictionary.
        **kwargs: Parameters to pass into RandomForestRegressor.
    """

    X_train, X_test = data_split["X_train"], data_split["X_test"]
    y_train, y_test = data_split["y_train"], data_split["y_test"]

    rf_params = {
        "n_estimators": 80,
        "max_depth": 20,
        "random_state": 42,
        "n_jobs": -1,
        "oob_score": True,
        "bootstrap": True,
    }
    rf_params.update(kwargs)

    rf_model = RandomForestRegressor(**rf_params)
    rf_model.fit(X=X_train, y=y_train)
    done("Random Forest trained successfully!")

    metrics = evaluate_and_save(rf_model, X_test, y_test, model_name, model_path)

    return rf_model, metrics


def train_random_forest_best(data_split: dict, model_name: str, model_path: str):
    """Train RF using pre-saved best hyperparameters."""

    # 讀取最佳參數
    with open(RESULT_DIR / "best_params.json", "r") as f:
        best_params = json.load(f)

    X_train, X_test = data_split["X_train"], data_split["X_test"]
    y_train, y_test = data_split["y_train"], data_split["y_test"]

    model = RandomForestRegressor(**best_params)
    model.fit(X_train, y_train)

    metrics = evaluate_and_save(model, X_test, y_test, model_name, model_path)

    return model, metrics
