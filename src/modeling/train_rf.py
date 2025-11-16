from sklearn.ensemble import RandomForestRegressor

from src.modeling.evaluate_model import evaluate_and_save
from src.utils.emoji_log import done


def train_random_forest(data_split: dict, **kwargs):
    """
    Train Random Forest model and evaluate performance.
    Args:
        data_split (dict): Train/test split dictionary.
        **kwargs: Parameters to pass into RandomForestRegressor.
    """

    X_train, X_test = data_split["X_train"], data_split["X_test"]
    y_train, y_test = data_split["y_train"], data_split["y_test"]

    rf_params = {
        "n_estimators": 200,
        "max_depth": None,
        "random_state": 42,
        "n_jobs": -1,
        "oob_score": True,
        "bootstrap": True,
    }
    rf_params.update(kwargs)

    rf_model = RandomForestRegressor(**rf_params)
    rf_model.fit(X=X_train, y=y_train)
    done("Random Forest trained successfully!")

    metrics = evaluate_and_save(rf_model, X_test, y_test, "RandomForest")

    return rf_model, metrics
