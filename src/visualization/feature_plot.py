import matplotlib.pyplot as plt
import seaborn as sns

from src.config import FIGURE_DIR, TITLE_FONT_SIZE


def plot_correlation_heatmap(df, filename=None):
    """
    Generate and optionally save a correlation heatmap for numerical features.

    This function computes `df.corr()` internally and visualizes the correlation
    matrix using a heatmap (Seaborn). It is commonly used in early EDA and
    feature engineering inspection.

    Args:
        df (pd.DataFrame):
            Input DataFrame containing numerical columns to visualize.
        filename (str | None, optional):
            If provided, the heatmap will be saved to this path.
            If None, the plot will only be displayed or closed.

    Returns:
        None

    Example:
        >>> from src.visualization.feature_plot import plot_correlation_heatmap
        >>> plot_correlation_heatmap(df, filename="corr.png")
    """

    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
    plt.title("Correlation Heatmap", fontsize=TITLE_FONT_SIZE)

    if filename:
        save_path = FIGURE_DIR / f"{filename}.png"
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_feature_distributions(df, feature, filename=None):
    """
    Plot the distribution (histogram + KDE) of a given feature.

    This visualization helps evaluate skewness, outliers, and whether
    transformations (e.g., log1p) are needed.

    Args:
        df (pd.DataFrame):
            Input DataFrame containing the feature.
        feature (str):
            Column name of the feature to visualize.
        filename (str | None, optional):
            Optional path to save the generated plot.

    Returns:
        None

    Example:
        >>> plot_feature_distribution(df, "pm25", "output/pm25_dist.png")
    """

    plt.figure(figsize=(8, 5))
    sns.histplot(df[feature], kde=True)
    plt.title(f"Distribution of {feature}", fontsize=TITLE_FONT_SIZE)

    if filename:
        save_path = FIGURE_DIR / f"{filename}.png"
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
