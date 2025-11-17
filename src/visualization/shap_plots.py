import matplotlib.pyplot as plt
import shap

from src.config import FIGURE_DIR


def plot_shap_summary(shap_value, X, filename=None):
    """
    Generate a SHAP summary plot (global feature importance).

    The summary plot visualizes:
        - Mean absolute SHAP value per feature (importance ranking)
        - Distribution of SHAP contributions across samples
        - Relationship between feature values and SHAP magnitude

    Args:
        shap_values (array-like or SHAP object):
            SHAP values computed from a fitted TreeExplainer.
        X (pd.DataFrame):
            Feature matrix used to compute SHAP values.
        filename (str | None, optional):
            If provided, saves the plot to this path.

    Returns:
        None

    Example:
        >>> shap_values = explainer.shap_values(X_train)
        >>> plot_shap_summary(shap_values, X_train, "shap_summary.png")
    """

    shap.summary_plot(shap_value, X, show=False)

    if filename:
        save_path = FIGURE_DIR / f"{filename}.png"
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_shap_dependence(shap_value, X, feature, filename=None):
    """
    Generate a SHAP dependence plot for a single feature.

    This plot illustrates:
        - How feature values influence SHAP contributions
        - Non-linear relationships between the feature and the model output
        - Potential interaction effects (highlighted by color coding)

    Args:
        shap_values (array-like):
            SHAP values from TreeExplainer.
        X (pd.DataFrame):
            Feature dataset aligned with shap_values.
        feature (str):
            Name of the feature to visualize.
        filename (str | None, optional):
            Optional save path for the output figure.

    Returns:
        None

    Example:
        >>> plot_shap_dependence(shap_values, X, "pm25", "pm25_shap.png")
    """

    shap.dependence_plot(feature, shap_value, X, show=False)

    if filename:
        save_path = FIGURE_DIR / f"{filename}.png"
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
