"""
Clean the row data and separate the files based on the county or city.
"""

from src.cleaning.data_cleaning import clean_air_quality
from src.utils.IO_file import open_csv, save_csv_no_index
from colorama import Fore
from config import RAW_DIR, PROCESSED_DIR
import sys


def main():
    raw_file = RAW_DIR / "air_quality.csv"

    # 檢查檔案是否存在
    if not raw_file.exists():
        print(Fore.RED + f"❌ File not found: {raw_file}")
        sys.exit(1)

    print(Fore.YELLOW + f"📂 Loading raw data from: {raw_file}")
    df = open_csv(raw_file)

    # 清理資料
    print(Fore.BLUE + "🧹 Cleaning data...")
    df_cleaned = clean_air_quality(df)

    # 分縣市輸出
    print(Fore.CYAN + "💾 Saving cleaned data by county...")
    county_list = df_cleaned["county"].unique()
    try:
        for county in county_list:
            temp = df[df["county"] == county]
            save_csv_no_index(df=temp, filename=county)
        print(Fore.GREEN + f"✅ Cleaning complete! Files saved in {PROCESSED_DIR}")
    except Exception as e:
        print(Fore.RED + f"❌ Error while saving multiple CSVs: {e}")


if __name__ == "__main__":
    main()
