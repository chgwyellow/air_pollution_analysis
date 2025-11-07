import pandas as pd
from colorama import Fore
from src.utils.IO_file import save_csv_no_index
from src.utils.time_utils import add_time_features


def clean_air_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess nationwide air quality data.

    This function performs the following steps:
    1. Standardizes column names and ensures consistent formatting.
    2. Converts date columns to datetime objects and removes invalid rows.
    3. Handles missing values by filling with the mean value within each monitoring site.
    4. Removes duplicate records and filters out extreme or invalid AQI values.

    Args:
        df (pd.DataFrame): Raw air quality dataset.

    Returns:
        pd.DataFrame: Cleaned and standardized dataset ready for analysis or modeling.
    """

    df = df.copy()

    # Standardize column names
    df.columns = df.columns.str.lower().str.strip()

    # Convert date column to datetime and drop invalid dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

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

    # Handle missing values for major pollutants (fill by site mean)
    pollutant_cols = ["aqi", "pm2.5", "pm10", "so2", "no2", "co", "o3"]
    for col in pollutant_cols:
        if col in df.columns:
            df[col] = df.groupby("sitename")[col].transform(
                lambda x: x.fillna(x.mean())
            )

    # Column unit has 5,882,208 Nan so we can drop it off
    df = df.drop(columns=["unit"], errors="ignore")

    # replace the space between county name with _
    df["county"] = df["county"].str.replace(" ", "_")

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove invalid or extreme AQI values
    df = df[(df["aqi"] >= 0) & (df["aqi"] <= 500)]

    # Add time columns
    df = add_time_features(df)

    print(Fore.GREEN + "✅ Raw data has been cleaned.")

    save_csv_no_index(df, "Taiwan.csv")

    return df
