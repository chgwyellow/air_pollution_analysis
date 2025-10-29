import pandas as pd
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
        print("Didn't find the file.")
        return pd.DataFrame
