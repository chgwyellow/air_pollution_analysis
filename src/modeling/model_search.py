"""
The process of adjusting the RandomizedSearchCV and GridSearchCV hyperparameter
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src.modeling.search_space import get_rf_grid_params, get_rf_random_params
from src.utils.emoji_log import success


def run_random_search(X_train, y_train, cv=3, n_iter=20, random_state=42):
    """Run RandomizedSearch for Random Forest."""
    rf = RandomForestRegressor(random_state=random_state)

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
    success(
        f"Best estimator: {search.best_estimator_}, Best params: {search.best_params}"
    )
    return search.best_estimator_, search.best_params_


def run_grid_search(X_train, y_train, base_params=None, cv=3):
    """Run GridSearch for Random Forest."""
    base_params = base_params or {}
    rf = RandomForestRegressor(**base_params)

    search = GridSearchCV(
        estimator=rf,
        param_grid=get_rf_grid_params(),
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_train, y_train)
    success(
        f"Best estimator: {search.best_estimator_}, Best params: {search.best_params}"
    )
    return search.best_estimator_, search.best_params_
