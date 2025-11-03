from src.cleaning.data_cleaning import clean_air_quality
from src.utils.IO_file import open_csv, save_multi_csv_no_index
from colorama import Fore
from pathlib import Path

city = "Kaohsiung_City/"
RAW_PATH = Path("data/raw")
TARGET_PATH = Path("data/processed") / f"{city}.csv"


if __name__ == "__main__":
    if not TARGET_PATH.exists():
        df = open_csv(RAW_PATH / "air_quality.csv")
        df_cleaned = clean_air_quality(df)

        # Save the data via county
        county_list = df_cleaned["county"].unique()
        save_multi_csv_no_index(county_list, df_cleaned)
    else:
        df = open_csv(TARGET_PATH)
        print(Fore.GREEN + "✅ Loaded the cleaned data.")
