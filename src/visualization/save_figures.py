import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4


# -----------------------------
# 1. 通用檔名加時間戳記
# -----------------------------
def add_timestamp(filename: str, timestamp: bool = True) -> str:
    if timestamp:
        now = datetime.now().strftime("_%Y-%m-%d_")
        stem, ext = Path(filename).stem, Path(filename).suffix
        filename = f"{stem}_{now}{ext}"
    return filename


# -----------------------------
# 2. 儲存 matplotlib 圖
# -----------------------------
def save_plot(
    fig: plt.Figure,
    filename: str,
    show: bool = False,
    theme: str = "seaborn-v0_8-darkgrid",
    timestamp: bool = True,
):
    # -----------------------------
    # 套用主題
    # -----------------------------
    plt.style.use(theme)

    # 中文字型與負號設定
    plt.rcParams["font.family"] = "Microsoft JhengHei"  # 支援繁體中文
    plt.rcParams["axes.unicode_minus"] = False  # 支援負號

    # -----------------------------
    # 加上時間戳記
    # -----------------------------
    filename = add_timestamp(filename, timestamp)

    # -----------------------------
    # 儲存圖片
    # -----------------------------
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename)

    # 是否立即顯示
    if show:
        plt.show()

    print(f"Saved: {filename}")
    return filename


# -----------------------------
# 3. 儲存 Plotly 互動圖
# -----------------------------
def save_plotly_fig(fig: go.Figure, filename: str, timestamp: bool = True):
    filename = add_timestamp(filename, timestamp)
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(filename)
    print(f"Saved interactive Plotly figure: {filename}")
    return filename


# -----------------------------
# 4. 多圖整合成 PDF
# -----------------------------
def multi_chart_report(
    image_files: list[str], output_pdf: str = "report.pdf", timestamp: bool = True
):
    output_pdf = add_timestamp(output_pdf, timestamp)
    c = canvas.Canvas(output_pdf, pagesize=A4)
    width, height = A4

    for img_path in image_files:
        c.drawImage(str(img_path), 50, 150, width=width - 100, height=height - 300)
        c.showPage()  # 換頁

    c.save()
    print(f"Saved PDF report: {output_pdf}")
    return output_pdf
