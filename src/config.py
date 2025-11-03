from pathlib import Path

# === Paths ===
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MAT_DIR = BASE_DIR / "output" / "figures" / "matplotlib"
PLOTLY_DIR = BASE_DIR / "output" / "figures" / "plotly"
PDF_DIR = BASE_DIR / "output" / "figures" / "pdf"

# 自動建立資料夾
for path in [RAW_DIR, PROCESSED_DIR, MAT_DIR, PLOTLY_DIR, PDF_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# === Visualization Settings ===
FIG_SIZE = (12, 8)
FONT_SIZE = 14
COLOR_PALETTE = ["#2E86AB", "#F6C85F", "#6FB07F", "#9B6A6C"]

# === Other constants ===
SEED = 42
