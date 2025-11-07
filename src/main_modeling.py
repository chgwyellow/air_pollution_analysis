import pandas as pd

from src.modeling.feature_engineering import add_rolling_features
from src.modeling.train_baseline import load_cleaned_data

filename = "Taiwan"

# Load the target file
df = load_cleaned_data(filename)

# Add the rolling columns
df_rolling = add_rolling_features(df)
