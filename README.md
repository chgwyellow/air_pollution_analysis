# Taiwan Air Pollution Analysis

- Data cleaning and visualization on the Taiwan air quality dataset (Kaggle, 2016–2024).

## Data source
- Dataset: **"Taiwan Air Quality Data (2016-2024)"** from Kaggle.  
- Place the downloaded CSV at: `data/raw/air_quality.csv`.

## Project layout
- data/
  - raw/          # original CSV
  - processed/    # cleaned/derived files
- notebooks/      # checks and ETL (e.g., data_check.ipynb)
- README.md

## Field summary (selected)
- date: date and time  
- sitename: station name  
- county: county/city  
- aqi: air quality index  
- pollutant: main pollutant  
- status: status description  
- so2, co, o3, o3_8hr, pm10, pm2.5, no2, nox, no: pollutant concentrations  
- windspeed, winddirec, longitude, latitude, siteid
