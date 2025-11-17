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

# === Visualization Settings ===
FIG_SIZE = (12, 8)
FONT_SIZE = 14
TITLE_FONT_SIZE = 18
COLOR_PALETTE = ["#2E86AB", "#F6C85F", "#6FB07F", "#9B6A6C"]

# === Other constants ===
SEED = 42
