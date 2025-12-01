import folium
import pandas as pd
from folium.plugins import MarkerCluster


def get_aqi_color(aqi):
    """
    Retrieve colors corresponding to AQI which follows Taiwan Air Quality Index Standard
    """
    # No aqi
    if pd.isna(aqi):
        return "gray"

    if aqi <= 50:
        return "green"
    elif aqi <= 100:
        return "orange"
    elif aqi <= 150:
        return "red"
    elif aqi <= 200:
        return "purple"
    elif aqi <= 300:
        return "darkpurple"
    else:
        return "black"


def plot_station_map(df: pd.DataFrame, target_date: pd.Timestamp) -> folium.map:
    """
    Create a Folium map covered with Taiwan measurement station,
    Color changed follows the AQI on target_date.
    """

    # Ensure the correct date format
    df["date"] = pd.to_datetime(df["date"])

    # Find the closest data with target_date in hour unit
    target_hour = target_date.floor("h")
    daily_data = df[df["date"] == target_hour].copy()

    # If the specific hour is out of data, trying to find the last data at the same day
    if daily_data.empty:
        target_day = target_date.floor("D")
        daily_data = df[df["date"].dt.date == target_day].copy()

        daily_data = daily_data.sort_values("date").drop_duplicates(
            "sitename", keep="last"
        )  # Preserve the last data

    # Create the basic plot: Central point in Taiwan (23.7, 121.0)
    m = folium.Map(location=[23.6978, 120.9605], zoom_start=8, tiles="CartoDB positron")

    # Mark on the station
    for _, row in daily_data.iterrows():
        if pd.isna(row.get("latitude")) or pd.isna(row.get("longitude")):
            continue

        aqi_val = row.get(key="aqi", default=0)
        color = get_aqi_color(aqi_val)

        # Set popup content
        popup_html = f"""
        <b>測站:</b> {row["sitename"]}<br>
        <b>時間:</b> {row["date"]}<br>
        <b>AQI:</b> {int(aqi_val) if not pd.isna(aqi_val) else "N/A"}<br>
        <b>PM2.5:</b> {row.get("pm2.5", "N/A")}<br>
        """

        # Marker
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=folium.Popup(popup_html, max_width=200),
            tooltip=f"{row["sitename"]} (AQI: {aqi_val})",  # For hover
            icon=folium.Icon(color=color, icon="cloud", prefix="fa"),
        ).add_to(m)

    return m
