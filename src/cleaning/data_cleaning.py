import numpy as np
import pandas as pd

from src.utils.emoji_log import info, success, warn
from src.utils.IO_file import save_csv_no_index
from src.utils.time_utils import add_season_feature, add_time_features


def clean_air_quality(df: pd.DataFrame, drop_high_corr: bool = True) -> pd.DataFrame:
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
    pollutant_cols = ["so2", "co", "o3", "pm10", "pm2.5", "no2", "nox", "no"]
    for col in pollutant_cols:
        if col in df.columns:
            # if the pollutants less than 0, turn it to missing value
            df.loc[df[col] < 0, col] = np.nan
            # Fill na with the mean in every sitename
            df[col] = df.groupby("sitename")[col].transform(
                lambda x: x.fillna(x.mean())
            )

    # Column unit has 5,882,208 Nan so we can drop it off
    df = df.drop(columns=["unit"], errors="ignore")

    # replace the space between county name with _
    df["county"] = df["county"].str.replace(" ", "_")

    # Remove invalid or extreme AQI values
    df = df[(df["aqi"] >= 0) & (df["aqi"] <= 500)]

    # Drop highly redundant pollutant average columns
    if drop_high_corr:
        drop_columns = ["pm2.5_avg", "pm10_avg"]
        df.drop(columns=drop_columns, errors="ignore", inplace=True)

    # With too high missing value percentage, we drop it.
    drop_cols = ["pollutant", "siteid", "so2_avg"]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Fill the missing value in windspeed and winddirec with mean
    df["windspeed"] = df["windspeed"].fillna(df["windspeed"].mean())
    df["winddirec"] = df["winddirec"].fillna(df["winddirec"].mean())
    warn("Filling missing windspeed/winddirec with global mean.")

    # Fill the pollutants with mean
    pollutant_cols = ["no", "nox", "o3_8hr", "co_8hr"]
    for col in pollutant_cols:
        if col in df.columns:
            df[col] = df.groupby("sitename")[col].transform(
                lambda x: x.fillna(x.mean())
            )

    # Fill the missing value in longtitude and latitude with mean
    geo_cols = ["latitude", "longitude"]
    for col in geo_cols:
        if col in df.columns:
            df[col] = df.groupby("county")[col].transform(lambda x: x.fillna(x.mean()))

    # Add time columns
    df = add_time_features(df)

    # Add seasom colums
    df = add_season_feature(df)

    # Remove duplicate rows
    df = df.drop_duplicates()

    success("Raw data has been cleaned.")

    info(f"Shape: {df.shape}")

    save_csv_no_index(df, "Taiwan.csv")

    return df
