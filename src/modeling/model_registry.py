from src.modeling.train_baseline import train_linear_model
from src.modeling.train_rf import train_random_forest
from src.modeling.tune_rf import tune_random_forest

MODEL_REGISTRY = {
    "linear": train_linear_model,
    "rf": train_random_forest,
    "rf_tuned": tune_random_forest,
}
