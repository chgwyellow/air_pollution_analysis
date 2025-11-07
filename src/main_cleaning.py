"""
Clean the row data and separate the files based on the county or city.
"""

from src.cleaning.data_cleaning import clean_air_quality
from src.utils.IO_file import open_csv, save_csv_no_index
from src.utils.emoji_log import error, done, save, warn, success
from config import RAW_DIR, PROCESSED_DIR
from colorama import Fore
import sys
import time


def main():
    raw_file = RAW_DIR / "air_quality.csv"

    # Check if the file is existing.
    if not raw_file.exists():
        error(f"File not found: {raw_file}")
        sys.exit(1)

    # Loading csv file
    start = time.perf_counter()
    df = open_csv(raw_file)

    # Clean data
    print(Fore.BLUE + "🧹 Cleaning data...")
    df_cleaned = clean_air_quality(df)
    end = time.perf_counter()
    done(f"Cleaning data costs {round((end - start), 2)} seconds.")

    # Separate by county and city
    save("Saving cleaned data by county...")
    county_list = df_cleaned["county"].unique()
    try:
        for county in county_list:
            start = time.perf_counter()
            temp = df_cleaned[df_cleaned["county"] == county]
            if temp.empty:
                warn(f"No data found for {county}, skipping...")
                continue
            save_csv_no_index(df=temp, filename=county)
            end = time.perf_counter()
            done(f"It costs {round((end - start), 2)} seconds.")
        success(f"Cleaning complete! Files saved in {PROCESSED_DIR}")
    except Exception as e:
        error(f"Error while saving multiple CSVs: {e}")


if __name__ == "__main__":
    main()
