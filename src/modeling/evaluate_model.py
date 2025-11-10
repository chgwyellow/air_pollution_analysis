import datetime
import json

import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import MODEL_DIR
from src.utils.emoji_log import info, save


def evaluate_and_save(model, X_test, y_test, model_name: str):
    """Evaluate model and save results."""
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {"MAE": mae, "rmse": rmse, "R2": r2}

    info(f"{model_name} Results: {metrics}")

    # Save model
    timestamp = datetime.datatime.now().strftime("%Y%m%d_%H%M")
    model_path = MODEL_DIR / f"{model_name}_{timestamp}.pkl"
    joblib.dump(model, model_path)
    save(f"{model} saved at {model_path}")

    # Save metrics
    timestamp = datetime.datatime.now().strftime("%Y%m%d_%H%M")
    metrics_path = MODEL_DIR / f"{model_name}_metrics_{timestamp}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    save(f"Metrics saved at {metrics_path}")

    return metrics
