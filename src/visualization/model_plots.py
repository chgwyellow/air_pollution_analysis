import matplotlib.pyplot as plt
import seaborn as sns

from src.config import FIGURE_DIR, FONT_SIZE, TITLE_FONT_SIZE


def plot_y_true_vs_pred(y_true, y_pred, filename=None):
    """
    Create a scatter plot comparing true vs predicted values.

    This plot indicates model fit quality. Ideally, points should be aligned
    along the 45-degree diagonal line. Useful for regression evaluation.

    Args:
        y_true (array-like):
            Ground truth target values.
        y_pred (array-like):
            Model predicted values (same length as y_true).
        filename (str | None, optional):
            Destination path to save the plot. If None, the plot is not saved.

    Returns:
        None

    Example:
        >>> plot_y_true_vs_pred(y_test, y_pred, "ytrue_ypred.png")
    """

    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5)
    plt.xlabel("True AQI", fontsize=FONT_SIZE)
    plt.ylabel("Predicted AQI", fontsize=FONT_SIZE)
    plt.title("True vs Predicted AQI", fontsize=TITLE_FONT_SIZE)

    if filename:
        save_path = FIGURE_DIR / f"{filename}.png"
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_residuals(y_true, y_pred, filename=None):
    """
    Plot the distribution of residuals (y_true - y_pred).

    Residual analysis helps identify:
        - Bias (mean residual drift)
        - Skewness or heavy tails
        - Model underfitting or overfitting
    A well-performing regression model should show a roughly symmetric,
    centered residual distribution.

    Args:
        y_true (array-like):
            Ground truth values.
        y_pred (array-like):
            Predicted values.
        filename (str | None, optional):
            Optional path to save the residual plot.

    Returns:
        None

    Example:
        >>> plot_residuals(y_test, y_pred, "residuals.png")
    """

    residuals = y_true - y_pred
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, kde=True)
    plt.title("Residual Distribution", fontsize=TITLE_FONT_SIZE)

    if filename:
        save_path = FIGURE_DIR / f"{filename}.png"
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
