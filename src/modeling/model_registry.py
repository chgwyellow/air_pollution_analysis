from src.config import (
    MODEL_LGBM_DIR,
    MODEL_LINEAR_DIR,
    MODEL_RF_BEST_DIR,
    MODEL_RF_DIR,
    MODEL_RF_TUNED_DIR,
)
from src.modeling.train_baseline import train_linear_model
from src.modeling.train_lightgbm import train_lightgbm
from src.modeling.train_rf import train_random_forest, train_random_forest_best
from src.modeling.tune_rf import tune_random_forest

MODEL_REGISTRY = {
    "linear": {
        "model_name": "Baseline_linear",
        "save_dir": MODEL_LINEAR_DIR,
        "train_func": train_linear_model,
    },
    "rf": {
        "model_name": "RandomForest_Safe",
        "save_dir": MODEL_RF_DIR,
        "train_func": train_random_forest,
    },
    "rf_best": {
        "model_name": "RandomForest_Best",
        "save_dir": MODEL_RF_BEST_DIR,
        "train_func": train_random_forest_best,
    },
    "rf_tuned": {
        "model_name": "RandomForest_Tuned",
        "save_dir": MODEL_RF_TUNED_DIR,
        "train_func": tune_random_forest,
    },
    "lgbm": {
        "model_name": "LightGBM",
        "save_dir": MODEL_LGBM_DIR,
        "train_func": train_lightgbm,
    },
}
