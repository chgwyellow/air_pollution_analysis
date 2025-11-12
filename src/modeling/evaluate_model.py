import datetime
import json

import joblib
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from src.config import MODEL_DIR
from src.utils.emoji_log import info, save


def evaluate_and_save(model, X_test, y_test, model_name: str, save_file: bool = True):
    """Evaluate model and save results."""
    # === 1. Predict ===
    y_pred = model.predict(X_test)

    # === 2. Compute metrics ===
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {"MAE": mae, "RMSE": rmse, "R2": r2}
    info(f"{model_name} Results: {metrics}")

    if save_file:
        # === 3. Save model ===
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        model_path = MODEL_DIR / f"{model_name}_{timestamp}.pkl"
        joblib.dump(model, model_path)
        save(f"{model_name} model saved at {model_path}")

        # === 4. Save metrics ===
        metrics_path = MODEL_DIR / f"{model_name}_metrics_{timestamp}.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
        save(f"Metrics saved at {metrics_path}")

    return metrics
