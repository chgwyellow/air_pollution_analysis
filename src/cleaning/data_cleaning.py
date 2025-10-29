import pandas as pd
from colorama import Fore


def clean_air_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw data

    Args:
        df (pd.DataFrame): DataFrame object of raw data

    Returns:
        pd.DataFrame: Cleaned DataFrame object
    """
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

    df[numeric_col] = df[numeric_col].apply(pd.to_numeric, errors="coerce")

    # Column unit has 5,882,208 Nan so we can drop it off
    df = df.drop(columns=["unit"])

    # replace the space between county name with _
    df["county"] = df["county"].str.replace(" ", "_")

    print(Fore.GREEN + "✅ Raw data has been cleaned.")
    return df
