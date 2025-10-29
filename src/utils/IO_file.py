import pandas as pd
import os
from colorama import Fore


def open_csv(path: str) -> pd.DataFrame:
    """Retrieve the csv file.

    Args:
        path (_type_): The path of csv file is.

    Returns:
        pd.DataFrame: A csv file will be transformed to DataFrame type.
    """
    try:
        df = pd.read_csv(path, low_memory=False)
        print(Fore.GREEN + "✅ Read the csv file successfully.")
        return df
    except FileNotFoundError:
        print(Fore.RED + "❌ Can not find the file.")
        return pd.DataFrame


def save_csv_no_index(df: pd.DataFrame, path: str) -> None:
    """Save the DataFrame object to csv file without the index

    Args:
        df (pd.DataFrame): The DataFrame which is gonna be stored.
        path (str): The final path to store csv file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(Fore.GREEN + "✅ The csv file has been saved.")


def save_multi_csv_no_index(county_list: list, df: pd.DataFrame) -> None:
    """Separate the DataFrame based on county and save them.

    Args:
        county_list (list): list with county name
        df (pd.DataFrame): cleaned DataFrame object
    """
    try:
        for county in county_list:
            temp = df[df["county"] == county]
            path = f"data/processed/{county}.csv"
            save_csv_no_index(df=temp, path=path)
        print(Fore.GREEN + "✅ Saved all separated csv files.")
    except Exception:
        print(Fore.RED + "❌ There's something wrong here.")
