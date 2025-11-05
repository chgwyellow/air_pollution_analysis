"""
Define functions related to the trend analysis
"""

import pandas as pd
from colorama import Fore
from src.utils.time_utils import add_time_columns, filter_by_hour


# -----------------------------
# 1. 以縣市做groupby
# -----------------------------
def pollutant_trends_city(
    df: pd.DataFrame,
    pollutant_sign: str | list[str] = "pm2.5",
    aggregation: str = "mean",
    start_date: str = "2016-11-25",
    end_date: str = "2024-08-31",
    as_long: bool = False,
    time_granularity: str | None = None,  # 年/月/日/時
    hour_range: tuple[int, int] | None = None,  # 篩選一天中的時段，例如 (20, 23)
) -> tuple[pd.DataFrame, list[str], str, str, str]:
    """
    以選定的汙染物質、聚合方式及起止日期，計算各縣市的趨勢。
    支援多污染物、多時間層級與特定時段篩選。

    參數：
        df: 原始 DataFrame，需包含 'county'、'date' 欄位
        pollutant_sign: 汙染物欄位名稱或 list
        aggregation: 聚合方式，例如 'mean'、'median'、'max'
        start_date, end_date: 日期篩選範圍
        as_long: 若為 True，輸出長格式 (方便繪圖)
        time_granularity: 可選 'year'、'month'、'day'、'hour'
        hour_range: 篩選特定時段，例如 (20, 23) 表示每天 20:00~23:59

    回傳：
        tuple(df, pollutant_signs, aggregation, start_date, end_date)
    """

    # --- 時間轉換與篩選 ---
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[
        (df["date"] >= pd.to_datetime(start_date))
        & (df["date"] <= pd.to_datetime(end_date))
    ]

    # --- 時段篩選 ---
    df = filter_by_hour(df, hour_range)

    # --- 若 pollutant_sign 是單一字串，就轉成 list ---
    if isinstance(pollutant_sign, str):
        pollutant_sign = [pollutant_sign]

    # --- 根據 time_granularity 新增對應欄位 ---
    df, group_cols = add_time_columns(df, time_granularity, "county")

    # --- 聚合 ---
    grouped = df.groupby(group_cols)[pollutant_sign].agg(aggregation).reset_index()

    # --- 若要輸出長格式 (方便畫圖) ---
    if as_long:
        grouped = grouped.melt(
            id_vars=group_cols,
            value_vars=pollutant_sign,
            var_name="pollutant",
            value_name=f"{aggregation}_value",
        )

    return (
        grouped,
        pollutant_sign,
        aggregation,
        str(pd.to_datetime(start_date).date()),
        str(pd.to_datetime(end_date).date()),
    )


# -----------------------------
# 2. 以鄉鎮區做groupby
# -----------------------------
def pollutant_trends_district(
    df: pd.DataFrame,
    city: str | None = None,
    site_name: str | None = None,
    pollutant_sign: str | list[str] = "pm2.5",
    aggregation: str = "mean",
    start_date: str = "2016-11-25",
    end_date: str = "2024-08-31",
    as_long: bool = False,
    time_granularity: str | None = None,  # 年/月/日/時
    hour_range: tuple[int, int] | None = None,  # 篩選一天中的時段，例如 (20, 23)
) -> tuple[pd.DataFrame, list[str], str, str, str]:
    """
    以選定的汙染物質、聚合方式及起止日期，計算鄉鎮市區的趨勢。
    支援多個汙染物、多時間層級及特定時段篩選。

    參數：
        df: 原始 DataFrame，需包含 'city'、'sitename'、'date' 欄位
        city: 選定縣市名稱（若為 None，使用全部資料）
        site_name: 選定測站名稱（若為 None，包含該縣市全部測站）
        pollutant_sign: 汙染物欄位名稱或 list
        aggregation: 聚合方式，例如 'mean'、'median'、'max'
        start_date, end_date: 日期篩選範圍
        as_long: 若為 True，輸出長格式 (方便繪圖)
        time_granularity: 可選 'year'、'month'、'day'、'hour'
        hour_range: 篩選特定時段，例如 (20, 23) 表示每天 20:00~23:59

    回傳：
        tuple(df, pollutant_signs, aggregation, start_date, end_date)
    """

    # --- 時間轉換與篩選 ---
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[
        (df["date"] >= pd.to_datetime(start_date))
        & (df["date"] <= pd.to_datetime(end_date))
    ]

    # --- 時段篩選 ---
    df = filter_by_hour(df, hour_range)

    # --- city 篩選 ---
    if city is not None:
        if city in df["city"].unique():
            df = df[df["city"] == city]
        else:
            print(Fore.YELLOW + f"⚠️ City '{city}' not found in dataset.")
            return pd.DataFrame(columns=df.columns), [], aggregation, "", ""

    # --- sitename 篩選 ---
    if site_name is not None:
        if site_name in df["sitename"].unique():
            df = df[df["sitename"] == site_name]
        else:
            print(
                Fore.YELLOW + f"⚠️ Site '{site_name}' not found in {city or 'dataset'}."
            )
            return pd.DataFrame(columns=df.columns), [], aggregation, "", ""

    # --- 若 pollutant_sign 是單一字串，就轉成 list ---
    if isinstance(pollutant_sign, str):
        pollutant_sign = [pollutant_sign]

    # --- 根據 time_granularity 新增對應欄位 ---
    df, group_cols = add_time_columns(df, time_granularity, ["county", "sitename"])

    # --- 聚合 ---
    grouped = df.groupby(group_cols)[pollutant_sign].agg(aggregation).reset_index()

    # --- 若要輸出長格式 (方便畫圖) ---
    if as_long:
        grouped = grouped.melt(
            id_vars=group_cols,
            value_vars=pollutant_sign,
            var_name="pollutant",
            value_name=f"{aggregation}_value",
        )

    return (
        grouped,
        pollutant_sign,
        aggregation,
        str(pd.to_datetime(start_date).date()),
        str(pd.to_datetime(end_date).date()),
    )
