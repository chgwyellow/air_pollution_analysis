# Taiwan Air Pollution Analysis

Data cleaning, feature engineering, modeling and visualization on the Taiwan air quality dataset (Kaggle, 2016–2024).  
針對 **台灣 2016–2024 空氣品質資料** 進行資料清理、特徵工程、建模與視覺化分析。

---

## 📂 Data Source 資料來源

- Dataset: **"Taiwan Air Quality Data (2016–2024)"** from Kaggle  
- Download and place the CSV at: `data/raw/air_quality.csv`  

---

## 📁 Project Layout 專案結構

```text
.
├─ data/
│  ├─ raw/               # 原始資料（air_quality.csv）
│  │  └─ .gitkeep
│  └─ processed/         # 清理後資料
│
├─ models/               # 訓練後模型與物件
│  ├─ *.pkl
│  ├─ *.json
│
├─ notebook/
│  ├─ 01_data_cleaning_check.ipynb
│  ├─ 02_feature_check.ipynb
│  ├─ 03_baseline_modeling.ipynb
│  ├─ 04_nonlinear_modeling.ipynb
│  └─ 05_model_optimization.ipynb
│
├─ output/
│  ├─ figures/            # 圖表輸出
│  ├─ predictions/        # 預測結果
|
├─ result/                # 參數與cv結果
│
├─ src/
│  ├─ cleaning/           # 資料清理
│  ├─ features/           # 特徵工程
│  ├─ modeling/           # 建模流程
│  ├─ utils/              # 工具函式（log、paths、IO 等）
│  ├─ visualization/      # 視覺化繪圖
│  ├─ __init__.py
│  ├─ config.py
│  ├─ main_cleaning.py
│  ├─ main_modeling.py
│  └─ main_visualization.py
│
├─ emoji.txt
├─ poetry.lock
├─ pyproject.toml
├─ .gitignore
└─ README.md
```

---

## 📘 Notebook / Chapter Overview  

各章節說明與 Notebook 對應關係

---

### **Chapter 01 — Data Cleaning Quality Check**  

**Notebook:** `01_data_cleaning_check.ipynb`

- 檢查原始資料品質  
- 缺值分佈 / 重複列檢查  
- 數值範圍是否合理  
- 日期連續性與格式檢查  
- 初步相關分析（sanity check）  
- 產出：clean 版本 processed 檔案＆清理報告

---

### **Chapter 02 — Feature Engineering Quality Check**  

**Notebook:** `02_feature_check.ipynb`

- 驗證特徵工程結果  
- 時間特徵（year/month/day/hour）檢查  
- 滾動統計（rolling mean / std）檢查  
- 特徵分佈、離群值、dtype  
- 特徵與目標變數的關係圖  
- 產出：確認後的特徵列表＋分析圖表

---

### **Chapter 03 — Baseline Modeling**  

**Notebook:** `03_baseline_modeling.ipynb`

- 定義 `X` / `y`  
- 訓練 Linear Regression baseline  
- 評估 MAE / RMSE / R²  
- 儲存 baseline 模型與 metrics  
- 與 Chapter 04 做效能比較基準

---

### **Chapter 04 — Nonlinear Modeling (Random Forest)**  

**Notebook:** `04_nonlinear_modeling.ipynb`

- 建立 Random Forest 回歸模型  
- 與 baseline 做效能比較  
- 初步 `feature_importances_` 分析  
- 視覺化殘差、預測 vs 實際  
- 釐清 RF 模型的初版效能瓶頸  
- 為 Chapter 05 的調參做準備

---

### **Chapter 05 — Model Optimization & Tuning**  

**Notebook:** `05_model_optimization.ipynb`

- 設計 RF 搜尋空間（n_estimators / max_depth / min_samples_split…）  
- RandomizedSearchCV：廣域快速搜尋  
- GridSearchCV：局部精細搜尋  
- 比較：初版 RF vs RandomSearch RF vs GridSearch RF  
- 儲存最佳模型（Final Refit）與 metrics  
- Summary：模型已接近最佳化上限，效能增益有限但穩定

---

### **(Planned) Chapter 06 — Model Explainability (Feature Importance & SHAP)**  

**Notebook（預計）:** `06_model_explainability.ipynb`

- 使用最佳 RF 模型做特徵重要性分析  
- 建立 SHAP TreeExplainer  
- SHAP Summary Plot（Global）  
- SHAP Dependence Plot（Interactions）  
- 找出影響空氣品質最關鍵的環境因子  
- 把模型結果連結到真實環境情境（風向、溫度、污染物濃度等）

---

## 🧪 Field Summary）

| Field | Description |
|-------|-------------|
| `date` | 日期時間 |
| `sitename` | 測站名稱 |
| `county` | 縣市 |
| `aqi` | 空氣品質指標 |
| `pollutant` | 主要污染物 |
| `so2`, `co`, `o3`, `o3_8hr` | 氣體污染物 |
| `pm10`, `pm2.5` | 懸浮微粒濃度 |
| `windspeed`, `winddirec` | 風速／風向 |
| `longitude`, `latitude` | GPS |
| `siteid` | 測站 ID |

後續會加入更多特徵（時間特徵、滾動統計、交互項等）。

---
