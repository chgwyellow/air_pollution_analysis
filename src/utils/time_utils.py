import pandas as pd


# -----------------------------
# 1. 根據時間顆粒度新增時間欄位
# -----------------------------
def add_time_columns(
    df: pd.DataFrame,
    time_granularity: str | None = None,
    base_cols: list[str] | None = None,
):
    """
    根據 time_granularity 新增時間欄位並回傳 groupby 需要的欄位。

    參數：
        df: DataFrame，需包含 'date' 欄位 (datetime 格式)
        time_granularity: 可選 'year'、'month'、'day'、'hour'
        base_cols: groupby 時的基礎欄位，例如 ["county"] 或 ["city", "sitename"]

    回傳：
        df, group_cols
    """

    if base_cols is None:
        base_cols = []
    # 如果傳入是str，轉成list
    elif isinstance(base_cols, str):
        base_cols = [base_cols]

    if time_granularity is None:
        return df, base_cols

    valid_levels = ["year", "month", "day", "hour"]
    if time_granularity not in valid_levels:
        raise ValueError(f"time_granularity 必須是 {valid_levels} 之一")

    # 依 granularity 增加時間欄位
    df["year"] = df["date"].dt.year
    group_cols = base_cols + ["year"]

    if time_granularity in ["month", "day", "hour"]:
        df["month"] = df["date"].dt.month
        group_cols.append("month")

    if time_granularity in ["day", "hour"]:
        df["day"] = df["date"].dt.day
        group_cols.append("day")

    if time_granularity == "hour":
        df["hour"] = df["date"].dt.hour
        group_cols.append("hour")

    return df, group_cols


# -----------------------------
# 2. 篩選一天中的時間段，最小可以是一小時
# -----------------------------
def filter_by_hour(df: pd.DataFrame, hour_range: tuple[int, int] | int | None = None):
    """
    篩選一天中指定小時或時段。

    hour_range:
        - None: 不篩選
        - int: 單一小時，例如 20
        - tuple(start_hour, end_hour): 範圍，例如 (20, 23)
    """
    if hour_range is None:
        return df

    if isinstance(hour_range, int):  # 只有一個時間
        return df[df["date"].dt.hour == hour_range]
    elif isinstance(hour_range, tuple):  # 一個時間區間
        start, end = hour_range
        return df[df["date"].dt.hour.between(start, end)]
    else:
        raise ValueError("hour_range 必須是 None, int, 或 tuple(int,int)")
