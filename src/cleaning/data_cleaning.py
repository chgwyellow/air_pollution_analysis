import pandas as pd
from colorama import Fore
from pathlib import Path
from src.utils.IO_file import save_csv_no_index


def clean_air_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw data and save it."""
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Transfer the air element columns' type to float
    numeric_col = [
        "so2",
        "co",
        "o3",
        "o3_8hr",
        "pm10",
        "pm2.5",
        "no2",
        "nox",
        "no",
        "windspeed",
        "winddirec",
        "co_8hr",
        "pm2.5_avg",
        "pm10_avg",
        "so2_avg",
    ]
    existing_col = [col for col in numeric_col if col in df.columns]
    df[existing_col] = df[existing_col].apply(pd.to_numeric, errors="coerce")

    # Column unit has 5,882,208 Nan so we can drop it off
    df = df.drop(columns=["unit"], errors="ignore")

    # replace the space between county name with _
    df["county"] = df["county"].str.replace(" ", "_")

    print(Fore.GREEN + "✅ Raw data has been cleaned.")

    save_csv_no_index(df, output_dir / "all_cleaned_data.csv")

    return df
