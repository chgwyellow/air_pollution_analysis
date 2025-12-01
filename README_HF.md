---
title: Taiwan Air Quality Prediction
emoji: 🍃
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: "1.51.0"
app_file: app.py
pinned: false
---

# 🍃 Taiwan Air Quality Time Machine

An interactive web application for predicting air quality (AQI) across Taiwan using machine learning.

## Features

- 🗺️ **Geo-spatial Map**: Visualize real-time AQI across all Taiwan monitoring stations
- 🔮 **LightGBM Prediction**: Fast and accurate tabular model predictions
- 🧠 **LSTM Time Series**: Deep learning model for temporal patterns
- 📊 **Historical Trends**: View past 7 days AQI trends

## Tech Stack

- **Frontend**: Streamlit, Folium
- **ML Models**: LightGBM, LSTM (TensorFlow/Keras)
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn, Plotly

## Dataset

Taiwan EPA Air Quality Monitoring Data (2016-2024)

- 5.8M+ records
- 80+ monitoring stations
- Hourly measurements

## GitHub Repository

[View Source Code](https://github.com/chgwyellow/air_pollution_analysis)
