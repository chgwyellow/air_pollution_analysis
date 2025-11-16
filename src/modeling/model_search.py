"""
The process of adjusting the RandomizedSearchCV and GridSearchCV hyperparameter
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src.modeling.search_space import (
    get_rf_grid_params,
    get_rf_grid_params_from_random,
    get_rf_random_params,
)
from src.utils.emoji_log import info, success


def run_random_search(X_train, y_train, cv=3, n_iter=20, random_state=42):
    """Run RandomizedSearch for Random Forest."""
    rf = RandomForestRegressor(n_jobs=-1, random_state=random_state)

    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=get_rf_random_params(),
        n_iter=n_iter,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=1,
        random_state=random_state,
    )

    search.fit(X_train, y_train)

    success(f"[Random Search] Best params: {search.best_params_}")
    return (search.best_estimator_, search.best_params_, search.cv_results_)


def run_grid_search(
    X_train,
    y_train,
    base_params=None,
    cv=3,
    sample_ratio=0.3,
    dynamic_grid=True,
):
    """
    Run GridSearch for fine-tuning.

    sample_ratio: reduce training time by sampling training data
    dynamic_grid: shrink grid search space around random-search best params
    """
    # 1) Subsample training data
    if sample_ratio < 1.0:
        n_samples = int(len(X_train) * sample_ratio)
        idx = np.random.choice(len(X_train), n_samples, replace=False)

        X_train_small = X_train.iloc[idx]
        y_train_small = y_train.iloc[idx]

        info(
            f"[Grid Search] Using subsample: {n_samples} rows ({sample_ratio*100:.1f}%)"
        )
    else:
        X_train_small = X_train
        y_train_small = y_train

    # 2) Shrink grid search space based on random search best params
    if dynamic_grid:
        grid = get_rf_grid_params_from_random(base_params)
        info("[Grid Search] Using dynamic grid based on RandomSearch best params.")
    else:
        grid = get_rf_grid_params()

    rf = RandomForestRegressor(n_jobs=-1, **base_params)

    search = GridSearchCV(
        estimator=rf,
        param_grid=grid,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_train_small, y_train_small)

    success(f"[Grid Search] Best params: {search.best_params_}")
    return (search.best_estimator_, search.best_params_, search.cv_results_)
