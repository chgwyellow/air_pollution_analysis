import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from src.config import MAT_DIR, PLOTLY_DIR, PDF_DIR
from PIL import Image


# -----------------------------
# 1. 通用檔名加時間戳記
# -----------------------------
def add_timestamp(
    filename: str, timestamp: bool = True, default_ext: str = ".png"
) -> str:
    path = Path(filename)
    stem = path.stem()  # 主檔名
    ext = path.suffix() if path.suffix() else default_ext  # 副檔名，沒有的話補預設

    if timestamp:
        now = datetime.now().strftime("%Y-%m-%d")
        filename = f"{stem}_{now}{ext}"
    else:
        filename = f"{stem}{ext}"

    return filename


# -----------------------------
# 2. 儲存 matplotlib 圖
# -----------------------------
def save_plot(
    fig: plt.Figure,
    filename: str,
    show: bool = True,
    theme: str = "seaborn-v0_8-darkgrid",
    timestamp: bool = True,
    figure_dir: Path = MAT_DIR,  # 使用 config 裡的 FIGURE_DIR
):
    """
    儲存 Matplotlib 圖為 png 檔案，支援時間戳記與指定資料夾
    """
    # -----------------------------
    # 套用主題
    # -----------------------------
    plt.style.use(theme)
    plt.rcParams["font.family"] = "Microsoft JhengHei"
    plt.rcParams["axes.unicode_minus"] = False

    # -----------------------------
    # 加上時間戳記
    # -----------------------------
    filename = add_timestamp(filename, timestamp)

    # -----------------------------
    # 指定儲存路徑
    # -----------------------------
    save_path = figure_dir / filename

    # 檢查資料夾是否存在
    if not save_path.parent.exists():
        print(f"❌ Folder does'nt exist: {save_path.parent}. Please create it.")
        raise Exception(save_path.parent)

    # -----------------------------
    # 儲存圖片
    # -----------------------------
    fig.savefig(save_path)

    # 是否立即顯示
    if show:
        plt.show()

    print(f"💾 Saved: {save_path}")
    return save_path


# -----------------------------
# 3. 儲存 Plotly 互動圖
# -----------------------------
def save_plotly_fig(
    fig: go.Figure,
    filename: str,
    timestamp: bool = True,
    figure_dir: Path = PLOTLY_DIR,
) -> Path:
    """
    儲存 Plotly 互動圖為 HTML 檔案，支援時間戳記與指定資料夾
    """
    # 加上時間戳記並補副檔名
    filename = add_timestamp(filename, timestamp, default_ext=".html")
    save_path = figure_dir / filename

    # 檢查資料夾是否存在
    if not save_path.parent.exists():
        raise FileNotFoundError(
            f"❌ Folder doesn't exist: {save_path.parent}. Please create it."
        )

    # 儲存 HTML
    fig.write_html(save_path)
    print(f"💾 Saved interactive Plotly figure: {save_path}")

    return save_path


# -----------------------------
# 4. 多圖整合成 PDF
# -----------------------------
def multi_chart_report(
    image_files: list[str],
    output_pdf: str = "report.pdf",
    timestamp: bool = True,
    pdf_dir: Path = PDF_DIR,
) -> Path:
    """
    將多張圖片整合成 PDF，每張圖片一頁。
    """
    # 加上 timestamp
    output_pdf = add_timestamp(output_pdf, timestamp)
    save_path = pdf_dir / output_pdf

    # 檢查資料夾
    if not save_path.parent.exists():
        raise FileNotFoundError(
            f"❌ Folder doesn't exist: {save_path.parent}. Please create it."
        )

    # 建立 PDF Canvas
    c = canvas.Canvas(save_path, pagesize=A4)
    page_width, page_height = A4

    for img_path in image_files:
        img_path = Path(img_path)
        if not img_path.exists():
            print(f"⚠️ Image not found, skipped: {img_path}")
            continue

        # 取得圖片尺寸
        with Image.open(img_path) as img:
            img_width, img_height = img.size

        # 計算縮放比例，保持比例不變形
        max_width = page_width - 100
        max_height = page_height - 100
        scale = min(max_width / img_width, max_height / img_height)

        display_width = img_width * scale
        display_height = img_height * scale

        # 將圖片置中
        x = (page_width - display_width) / 2
        y = (page_height - display_height) / 2

        c.drawImage(str(img_path), x, y, width=display_width, height=display_height)
        c.showPage()  # 換頁

    c.save()
    print(f"💾 Saved PDF report: {save_path}")
    return save_path
