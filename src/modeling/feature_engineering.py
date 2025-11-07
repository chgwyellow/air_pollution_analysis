"""
The functions that features the elements for model prediction.
"""

import pandas as pd
from src.utils.emoji_log import success, warn


def add_rolling_features(
    df: pd.DataFrame, pollutants: list[str] = None, windows: list[int] = [3, 7]
) -> pd.DataFrame:
    """
    Add rolling mean features for multiple raw pollutants.

    Args:
        df (pd.DataFrame): Input DataFrame containing pollutant columns.
        pollutants (list[str]): List of pollutant column names.
        windows (list[int]): Rolling window sizes in days.

    Returns:
        pd.DataFrame: DataFrame with new rolling features.
    """

    if pollutants is None:
        pollutants = ["so2", "co", "o3", "pm10", "pm2.5", "no2", "no", "nox"]

    df = df.sort_values("date").copy()

    for pollutant in pollutants:
        # Skip already smoothed columns
        if any(col in pollutant.lower() for col in ["_8hr", "_avg"]):
            warn(f"{pollutant} has been skipped to roll.")
            continue

        # Skip nox if no and no2 exist
        if pollutant.lower() == "nox" and all(
            col in df.columns for col in ["no", "no2"]
        ):
            warn(f"Skipping {pollutant} deu to NO + NO2 already exist.")
            continue

        for w in windows:
            new_col = f"{pollutant}_rolling_{w}d"
            df[new_col] = df.groupby("sitename")[pollutant].transform(
                lambda x: x.rolling(w, min_periods=1).mean()
            )
    success("Rolling features added.")
