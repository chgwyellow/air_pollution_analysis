import pandas as pd

from src.features.feature_engineering import (
    add_rolling_features,
    clip_pollutants,
    handle_outliers_iqr,
    log_transform_features,
)
from src.modeling.train_baseline import load_cleaned_data

filename = "Taiwan"

# Load the target file
df = load_cleaned_data(filename)

# clip the pollutants
df_clip = clip_pollutants(df.copy())

# Add the rolling columns
df_rolling = add_rolling_features(df_clip)

# Set iqr
df_iqr = handle_outliers_iqr(df_rolling)

# smooth the skewes
df_log = log_transform_features(df_iqr)
