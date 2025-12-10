import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from src.app.backend import get_station_data, load_data, load_model, process_and_predict
from src.visualization.map_plot import plot_station_map

# === 1. Page Config ===
st.set_page_config(
    page_title="Taiwan Air Quality Time Machine", page_icon="🍃", layout="wide"
)

# === 2. Load Data ===
df = load_data()

if df.empty:
    st.error("Failed to load data. Please check if 'full_data.parquet' exists.")
    st.stop()

# === 3. sidebar ===
st.sidebar.title("🍃 Air Quality Time Machine")
st.sidebar.markdown("Predict historical AQI using LightGBM model.")

# === 3.0  App Mode Selection ===
app_mode = st.sidebar.radio(
    "Select Mode",
    ["🔮 Prediction (Time Machine)", "🗺️ Geo-spatial Map"],
    help="Choose single station AQI prediction or Taiwan Map Browse.",
)

# Prediction mode
if app_mode == "🔮 Prediction (Time Machine)":

    # === 3.1 Station Selection ===
    stations = sorted(df["sitename"].unique())

    default_index = 0
    if "Taoyuan" in stations:
        default_index = stations.index("Taoyuan")

    selected_station = st.sidebar.selectbox(
        "Select Station", stations, index=default_index
    )

    # === 3.2 Date Selection ===
    min_date = df["date"].min().date()
    max_date = df["date"].max().date()

    selected_date = st.sidebar.date_input(
        "Select Date",
        value=pd.to_datetime("2020-01-01").date(),
        min_value=min_date,
        max_value=max_date,
    )

    # === 4. Predict Button ===
    if st.sidebar.button("🔮 Predict AQI", type="primary"):

        # 4.1 Main Panel Header
        st.title(f"📍 {selected_station} Station Analysis")
        st.markdown(f"**Target Date**: {selected_date}")

        # 4.2 Get Data Window
        input_data = get_station_data(df, selected_station, selected_date)

        if input_data.empty or len(input_data) < 7:
            st.warning(
                "⚠️ Not enough historical data for this date to calculate rolling features."
            )
        else:
            # 4.3 Load Model & Predict
            prediction = None
            model = load_model()

            if model:
                prediction = process_and_predict(input_data, model)

            # === 5. Display Result ===
            if prediction is not None:
                col1, col2 = st.columns(2)

                # Show the prediction on the left side
                with col1:
                    st.metric(
                        label="Predicted AQI",
                        value=f"{prediction:.2f}",
                        delta="LightGBM Model",
                    )

                # Show the actual value if we have, comparing the accuracy
                with col2:
                    actual_row = input_data[input_data["date"].dt.date == selected_date]
                    if not actual_row.empty and "aqi" in actual_row.columns:
                        actual_aqi = actual_row["aqi"].iloc[0]
                        # Calculate the error
                        error = prediction - actual_aqi
                        st.metric(
                            label="Actual AQI",
                            value=f"{actual_aqi:.2f}",
                            delta=f"{error:.2f} Error",
                            delta_color="inverse",  # lower better
                        )
                    else:
                        st.info("Actual AQI not available for this date.")

        # === 6. Historical Trend Charts ===
        st.subheader("📉 Past 7 Days Trend (AQI)")
        chart_data = input_data.set_index("date")["aqi"]
        st.line_chart(chart_data)

    else:
        st.info("👈 Select a station and date from the sidebar to start prediction!")


# Geo-spatial mode
elif app_mode == "🗺️ Geo-spatial Map":
    # Date Selection
    min_date = df["date"].min().date()
    max_date = df["date"].max().date()

    selected_date = st.sidebar.date_input(
        "Select Date",
        value=pd.to_datetime("2020-01-01").date(),
        min_value=min_date,
        max_value=max_date,
    )

    # Combine Date and Hour
    target_datetime = pd.to_datetime(f"{selected_date}")

    # Display Map
    st.title("🗺️ Taiwan Air Quality Map")
    st.markdown(f"### 📅 Showing Data for: **{selected_date}**")

    m = plot_station_map(df, target_datetime)
    st_folium(m, width=700, height=900)

    # Legend
    st.markdown(
        """
    **AQI Color Legend:**
    - 🟢 **Green (0-50)**: Good
    - 🟡 **Orange (51-100)**: Moderate
    - 🔴 **Red (101-150)**: Unhealthy for Sensitive Groups
    - 🟣 **Purple (151-200)**: Unhealthy
    - ⚫ **Black (201+)**: Hazardous
    """
    )
