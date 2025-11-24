"""
An utils lib containing data I/O functions.
"""

import gc
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from colorama import Fore

from src.config import PROCESSED_DIR
from src.utils.emoji_log import done, error, info, save, success, task


# === 1. 通用 CSV 讀取 ===
def open_csv(path: str, parse_dates: list[str] | None = None) -> pd.DataFrame:
    """Retrieve the csv file.

    Args:
        path (_type_): The path of csv file is.

    Returns:
        pd.DataFrame: A csv file will be transformed to DataFrame type.
    """
    if not isinstance(path, Path):
        path = Path(path)

    try:
        print(Fore.YELLOW + f"📂 Loading CSV: {path}")
        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        success(f"Read CSV successfully! Shape: {df.shape}")
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
        save(f"Saved file → {save_path}")
    except Exception as e:
        error(f"Failed to save file: {e}")


# === 3. 輸入檔案名稱判斷 ===
def name_check(filename: str) -> Path:
    path = Path(filename)

    if path.suffix == "":
        path = Path(filename + ".csv")
    elif path.suffix != ".csv":
        error(f"Invalid file extension: {filename}")
        return None

    full_path = PROCESSED_DIR / path

    return full_path


# === 4. csv轉換為parquet ===
def convert_csv_to_parquet(csv_path, parquet_path, chunksize=100000):
    """
    Convert CSV to Parquet in chunks to save memory.
    """
    if parquet_path.exists():
        success(f"File already exists, skipping conversion: {parquet_path}")
        return

    task(f"Start converting: {csv_path.name} -> {parquet_path.name}...")
    info(f"Chunk Size: {chunksize}")
    start_time = time.time()

    # use chunksize to read csv
    chunk_iter = pd.read_csv(csv_path, chunksize=chunksize)
    writer = None
    total_rows = 0

    try:
        for i, chunk in enumerate(chunk_iter):
            # turn pandas chunk to pyarrow table
            table = pa.Table.from_pandas(chunk)

            # initialize writer (use the schema of the first chunk)
            if writer is None:
                writer = pq.ParquetWriter(parquet_path, table.schema)

            writer.write_table(table)
            total_rows += len(chunk)

            # show progress every 10 chunks (about 10 million rows)
            if (i + 1) % 10 == 0:
                info(f"Processed {total_rows} rows...")

            # force release memory
            del chunk, table
            gc.collect()

    except Exception as e:
        error(f"Failed to convert CSV to Parquet: {e}")
        if writer:
            writer.close()
        # if failed, delete the possibly damaged file
        if parquet_path.exists():
            parquet_path.unlink()
        return

    if writer:
        writer.close()

    elapsed = time.time() - start_time
    done(f"Done! Total rows: {total_rows}")
    info(f"Time elapsed: {elapsed:.2f} seconds")
    info(f"New file size: {parquet_path.stat().st_size / 1024 / 1024:.2f} MB")
