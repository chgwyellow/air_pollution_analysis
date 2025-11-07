"""
An utils lib containing data I/O functions.
"""

import pandas as pd
from colorama import Fore
from pathlib import Path
from src.config import PROCESSED_DIR
from src.utils.emoji_log import error, success, save


# === 1. 通用 CSV 讀取 ===
def open_csv(path: str) -> pd.DataFrame:
    """Retrieve the csv file.

    Args:
        path (_type_): The path of csv file is.

    Returns:
        pd.DataFrame: A csv file will be transformed to DataFrame type.
    """
    try:
        print(Fore.YELLOW + f"📂 Loading CSV: {path}")
        df = pd.read_csv(path, low_memory=False)
        success("Read the csv file successfully.")
        return df
    except FileNotFoundError:
        error(f"File not found: {path}")
        return pd.DataFrame()
    except Exception as e:
        error(f"Failed to read CSV: {e}")
        return pd.DataFrame()


# === 2. 單一 CSV 儲存 ===
def save_csv_no_index(df: pd.DataFrame, filename: str) -> None:
    """Save the DataFrame object to csv file without the index"""

    # 替檔名加上suffix
    filename = Path(filename)
    if filename.suffix != ".csv":
        filename = filename.with_suffix(".csv")

    save_path = PROCESSED_DIR / filename.name

    try:
        df.to_csv(save_path, index=False)
        save(f"💾 Saved file → {save_path}")
    except Exception as e:
        error(f"Failed to save file: {e}")
