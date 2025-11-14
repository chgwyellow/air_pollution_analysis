"""
Manage the hyperparameters search space for Random Forest
"""


def get_rf_random_params():
    """Hyperparameter space for RandomizedSearchCV."""
    return {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }


def get_rf_grid_params():
    """Hyperparameter grid for GridSearchCV (refined search)."""
    return {
        "n_estimators": [200, 300, 400],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["log2"],
    }
