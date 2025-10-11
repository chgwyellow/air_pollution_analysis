# Taiwan Air Pollution Analysis

One-line: Data cleaning, EDA and visualization on the Taiwan air quality dataset (Kaggle, 2016–2024).

## Data source
- Dataset: "Taiwan Air Quality Data (2016-2024)" from Kaggle.  
- Place the downloaded CSV at: `data/raw/air_quality.csv`.

## Project layout
- data/
  - raw/          # original CSV
  - processed/    # cleaned/derived files
- notebooks/      # checks (e.g., data_check.ipynb)
- src/            # data processing and scripts
- README.md

## Field summary (selected)
- date: date and time  
- sitename: station name  
- county: county / city  
- aqi: air quality index  
- pollutant: main pollutant  
- status: status description  
- so2, co, o3, o3_8hr, pm10, pm2.5, no2, nox, no: pollutant concentrations  
- windspeed, winddirec, longitude, latitude, siteid

## Data quality & preprocessing recommendations
- Convert pollutant columns to numeric, coercing errors to NaN:
  ```
  ```python
  cols = ["so2","co","o3","o3_8hr","pm10","pm2.5","no2"]
  df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
  ```
  ```
- Check missing values by column:
  ```
  ```python
  df.isna().sum()
  ```
  ```
- Detect special non-numeric flags (e.g. "-") before/after conversion:
  ```
  ```python
  mask = df['so2'].astype(str).str.strip() == '-'
  df[mask]
  ```
  ```
- Get unique non-null values of a column (exclude empty/"-"):
  ```
  ```python
  unique_vals = df['county'].dropna().astype(str).str.strip().unique().tolist()
  unique_vals = [v for v in unique_vals if v not in ("", "-")]
  ```
  ```