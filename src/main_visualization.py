"""
Main Visualization Script
=========================

This script loads:
    - processed feature data
    - trained model (baseline / RF / tuned RF)
    - SHAP values (or recompute)
And generates key plots:
    - feature/EDA plots
    - model performance plots
    - SHAP explainability plots

All output figures will be saved to: output/figures/
"""

import joblib
import pandas as pd
import shap

from src.config import MODEL_DIR, PROCESSED_DIR
from src.utils.emoji_log import info, save, success
from src.visualization.feature_plot import (
    plot_correlation_heatmap,
    plot_feature_distributions,
)
from src.visualization.model_plots import plot_residuals, plot_y_true_vs_pred
from src.visualization.shap_plots import plot_shap_dependence, plot_shap_summary


def load_data(filename="Taiwan.csv"):
    """Load processed data for visualization."""
    return pd.read_csv(PROCESSED_DIR / filename)


def load_model(model_name="rf_final_model_20251115_1215.pkl"):
    """Load trained model (pkl file)."""
    return joblib.load(MODEL_DIR / model_name)


def visualize_features(df, feature, model_type):
    """Generate feature-related plots."""
    plot_correlation_heatmap(df, f"{model_type}/corr_heatmap")
    plot_feature_distributions(df, feature, f"{model_type}/{feature}_dist")
    save("Heatmap and feature distributions saved.")


def visualize_model(df, model, model_type, feature_names):
    """Generate model evaluation plots."""
    scaler = joblib.load(MODEL_DIR / "standard_scaler.pkl")
    X_scaled = scaler.transform(df[feature_names])
    X_scaled = pd.DataFrame(X_scaled, columns=feature_names)
    y_pred = model.predict(X_scaled)
    y_true = df["aqi"]

    plot_y_true_vs_pred(y_true, y_pred, f"{model_type}/true_vs_pred")
    plot_residuals(y_true, y_pred, f"{model_type}/residuals")
    save("True vs predicted and residuals saved.")


def visualize_shap(df, model, model_type, feature, feature_names):
    """Generate SHAP explainability plots."""
    X = df[feature_names]

    # Detect if model is linear
    if model_type == "linear":
        info("Skipping SHAP for Linear Regression")
        return

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    plot_shap_summary(shap_values, X, f"{model_type}/shap_summary")
    plot_shap_dependence(shap_values, X, feature, f"{model_type}/shap_{feature}")
    save("SHAP summary and dependence saved.")


def run_all_visualizations(model_type="linear"):
    # Load exact training dataset (rolling/log-transformed)
    df = pd.read_parquet(PROCESSED_DIR / "training_data_cleaned.parquet")

    # Load model's feature names
    feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")

    # Load model
    latest_path = MODEL_DIR / f"{model_type}/latest.pkl"
    model = joblib.load(latest_path)

    visualize_features(df, "pm2.5", model_type)
    visualize_model(df, model, model_type, feature_names)
    visualize_shap(df, model, model_type, "pm2.5", feature_names)

    success("All visualization completed!")


if __name__ == "__main__":
    run_all_visualizations("rf")
