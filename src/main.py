from src.cleaning.data_cleaning import clean_air_quality
from src.utils.IO_file import open_csv, save_multi_csv_no_index
from src.analysis.trend_analysis import pollutant_trends_city, pollutant_trends_district
from colorama import Fore
from config import RAW_DIR, PROCESSED_DIR

city = "Taoyuan_City"
TARGET_PATH = PROCESSED_DIR / f"{city}.csv"


if __name__ == "__main__":
    # 如果檔案不存在，讀取raw data然後開始清理，else則開啟既有檔案
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

    # 執行資料分析
    df, pollutant, method, s_date, e_date = pollutant_trends_city(
        df,
        pollutant_sign=["aqi", "pm2.5"],
        aggregation="mean",
        time_granularity="month",
        hour_range=(20),
    )
    print(df)
