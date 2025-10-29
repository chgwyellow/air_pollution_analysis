from src.cleaning.data_cleaning import clean_air_quality
from src.utils.IO_file import open_csv, save_multi_csv_no_index

RAW_PATH = r"data/raw/air_quality.csv"

if __name__ == "__main__":
    df = open_csv(RAW_PATH)
    df_cleaned = clean_air_quality(df)

    # Save the data via county
    county_list = df_cleaned["county"].unique()
    save_multi_csv_no_index(county_list, df_cleaned)
