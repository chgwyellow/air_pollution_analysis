from src.modeling.evaluate_model import evaluate_and_save
from src.modeling.model_search import run_grid_search, run_random_search
from src.utils.emoji_log import done, info


def tune_random_forest(data_split: dict):
    """Full tuning pipeline for Random Forest."""
    X_train, X_test = data_split["X_train"], data_split["X_test"]
    y_train, y_test = data_split["y_train"], data_split["y_test"]

    info("Running RandomizedSearchCV...")
    best_random_model, random_params = run_random_search(X_train, y_train)

    info(f"Best parameters (RandomizedSearch): {random_params}")

    info("Running GridSearchCV (fine tuning)...")
    best_grid_model, grid_params = run_grid_search(X_train, y_train, random_params)

    info(f"Best parameters (GridSearch): {grid_params}")

    done("Tuning complete! Evaluating final model...")

    final_metrics = evaluate_and_save(
        best_grid_model, X_test, y_test, "RandomForest_Tuned"
    )

    return best_grid_model, final_metrics
