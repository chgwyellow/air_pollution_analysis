from sklearn.ensemble import RandomForestRegressor

from src.modeling.evaluate_model import evaluate_and_save
from src.modeling.model_search import run_grid_search, run_random_search
from src.utils.emoji_log import done, info


def tune_random_forest(
    data_split: dict,
    grid_sample_ratio=0.3,
    use_dynamic_grid=True,
):
    """
    Full tuning pipeline for Random Forest.

    grid_sample_ratio: ratio of training data for GridSearch (speed boost)
    use_dynamic_grid: shrink grid around random best params
    """

    X_train, X_test = data_split["X_train"], data_split["X_test"]
    y_train, y_test = data_split["y_train"], data_split["y_test"]

    # ---------------------------------------------------------
    # 1) RandomSearch (full data)
    # ---------------------------------------------------------
    info("Running RandomizedSearchCV...")
    best_random_model, random_params, random_cv = run_random_search(X_train, y_train)
    info(f"Best parameters (RandomizedSearch): {random_params}")

    # ---------------------------------------------------------
    # 2) GridSearch (subsample + dynamic grid)
    # ---------------------------------------------------------
    info("Running GridSearchCV (fine tuning)...")
    best_grid_model, grid_params, grid_cv = run_grid_search(
        X_train=X_train,
        y_train=y_train,
        base_params=random_params,
        sample_ratio=grid_sample_ratio,
        dynamic_grid=use_dynamic_grid,
    )

    info(f"Best parameters (GridSearch): {grid_params}")

    # ---------------------------------------------------------
    # 3) Final model: train with full training data
    # ---------------------------------------------------------
    final_model = RandomForestRegressor(**grid_params)
    final_model.fit(X_train, y_train)

    # ---------------------------------------------------------
    # 4) Final evaluation
    # ---------------------------------------------------------
    done("Tuning complete! Evaluating final model...")

    final_metrics = evaluate_and_save(final_model, X_test, y_test, "RandomForest_Tuned")

    return final_model, final_metrics
