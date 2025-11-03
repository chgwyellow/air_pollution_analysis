from src.cleaning.data_cleaning import clean_air_quality
from src.utils.IO_file import open_csv, save_multi_csv_no_index
from colorama import Fore
from config import RAW_DIR, PROCESSED_DIR

city = "Kaohsiung_City/"
TARGET_PATH = PROCESSED_DIR / f"{city}.csv"


if __name__ == "__main__":
    if not TARGET_PATH.exists():
        try:
            df = open_csv(RAW_DIR / "air_quality.csv")
        except FileNotFoundError:
            print(Fore.RED + "❌ Raw data not found. Please check RAW_DIR.")
            exit(1)

        df_cleaned = clean_air_quality(df)

        # Save the data via county
        county_list = df_cleaned["county"].unique()
        save_multi_csv_no_index(county_list, df_cleaned)
    else:
        df = open_csv(TARGET_PATH)
        print(Fore.GREEN + "✅ Loaded the cleaned data.")
