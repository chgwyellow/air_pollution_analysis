import pandas as pd

from src.utils.emoji_log import error, success


# -----------------------------
# 1. add year, month, day, weekday, and hour in df
# -----------------------------
def add_time_features(df: pd.DataFrame, data_col: str = "date") -> pd.DataFrame:
    """
    Add basic time-related features to the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing a datetime column.
        date_col (str): Name of the date column.

    Returns:
        pd.DataFrame: DataFrame with new columns: year, month, day, weekday, hour.
    """
    df = df.copy()

    if data_col not in df.columns:
        error(f"No column named {data_col} found.")
        return df

    df[data_col] = pd.to_datetime(df[data_col], errors="coerce")

    df["year"] = df[data_col].dt.year
    df["month"] = df[data_col].dt.month
    df["day"] = df[data_col].dt.day
    df["weekday"] = df[data_col].dt.weekday
    df["hour"] = df[data_col].dt.hour

    success("Time features (year, month, day, weekday, hour) added successfully.")

    return df


# -----------------------------
# 2. Add season column
# -----------------------------
def add_season_feature(df: pd.DataFrame, month_col: str = "month") -> pd.DataFrame:
    """
    Add a 'season' column based on the month number.

    Args:
        df (pd.DataFrame): DataFrame containing a 'month' column.
        month_col (str): The column used to determine season.

    Returns:
        pd.DataFrame: DataFrame with a new 'season' column.
    """
    if month_col not in df.columns:
        error(f"No column named {month_col} found.")
        return df

    def _get_season(month: int) -> str:
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Autumn"

    df = df.copy()

    df["season"] = df[month_col].apply(_get_season)
    success("Season feature added successfully.")

    return df
