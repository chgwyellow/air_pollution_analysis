from pathlib import Path

# === Paths ===
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"
RESULT_DIR = BASE_DIR / "result"

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

FIGURE_DIR = OUTPUT_DIR / "figures"
PREDICTION_DIR = OUTPUT_DIR / "predictions"

MODEL_LINEAR_DIR = MODEL_DIR / "linear"
MODEL_RF_DIR = MODEL_DIR / "rf"
MODEL_RF_BEST_DIR = MODEL_DIR / "rf_best"
MODEL_RF_TUNED_DIR = MODEL_DIR / "rf_tuned"
MODEL_LGBM_DIR = MODEL_DIR / "lgbm"
MODEL_LSTM_DIR = MODEL_DIR / "lstm"

TITLE_FONT_SIZE = 18
FONT_SIZE = 14
