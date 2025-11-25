import time

import lightgbm as lgb

from src.modeling.evaluate_model import evaluate_and_save
from src.utils.emoji_log import done, task


def train_lightgbm(
    data_split: dict,
    model_name: str,
    model_path: str,
):
    """
    Train LightGBM Regressor with Early Stopping.
    """

    X_train, X_test = data_split["X_train"], data_split["X_test"]
    y_train, y_test = data_split["y_train"], data_split["y_test"]

    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 5000,  # Maximum number of trees (will be truncated by early_stopping)
        "learning_rate": 0.05,  # Learning rate
        "num_leaves": 63,  # Number of leaves (controls model complexity)
        "max_depth": -1,  # Tree depth (-1 means unlimited)
        "subsample": 0.8,  # Random sampling ratio (prevents overfitting)
        "colsample_bytree": 0.8,  # Feature sampling ratio
        "random_state": 42,
        "n_jobs": -1,  # Use all CPU cores
        "device": "cpu",  # If you have GPU, change to "gpu"
    }

    task("Training LightGBM model...")

    start = time.time()

    # Build model
    model = lgb.LGBMRegressor(**params)

    # Train model (with Early Stopping)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="rmse",
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=50
            ),  # means if the validation set error does not decrease for 50 consecutive times, stop training
            lgb.log_evaluation(period=100),  # means print log every 100 iterations
        ],
    )

    elapsed = time.time() - start
    done(f"Training finished in {elapsed:.2f} seconds!")

    # Evaluate model
    metrics = evaluate_and_save(model, X_test, y_test, model_name, model_path)

    return model, metrics
