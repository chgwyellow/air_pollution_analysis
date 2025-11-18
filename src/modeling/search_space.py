"""
Manage the hyperparameters search space for Random Forest
"""


def get_rf_random_params():
    """Hyperparameter space for RandomizedSearchCV."""
    return {
        "n_estimators": [100, 200, 300, 500, 700, 900],
        "max_depth": [None, 10, 20, 30, 40],
        "min_samples_split": [2, 5, 10, 15],
        "min_samples_leaf": [1, 2, 4, 6],
    }


def get_rf_grid_params():
    """Static hyperparameter grid for GridSearchCV."""
    return {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["log2"],
    }


def get_rf_grid_params_from_random(best_random_params: dict):
    """
    Dynamically shrink grid search space based on random search best params.
    Example:
        If random search found n_estimators=300,
        then grid search will explore [250, 300, 350]
    """
    return {
        "n_estimators": [
            max(50, best_random_params["n_estimators"] - 50),
            best_random_params["n_estimators"],
            best_random_params["n_estimators"] + 50,
        ],
        "max_depth": [
            best_random_params["max_depth"],
            None,
        ],
        "min_samples_split": [
            best_random_params["min_samples_split"],
            max(2, best_random_params["min_samples_split"] - 1),
        ],
        "min_samples_leaf": [
            best_random_params["min_samples_leaf"],
            best_random_params["min_samples_leaf"] + 1,
        ],
        "max_features": [best_random_params["max_features"]],
    }
