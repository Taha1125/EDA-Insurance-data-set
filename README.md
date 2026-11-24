# Medical Insurance Charges — Exploratory Data Analysis (EDA)

A step-by-step exploratory data analysis and preprocessing notebook on the **Insurance dataset**. The goal is to understand which factors (age, BMI, smoking, etc.) influence individual medical insurance **charges**, and to prepare the data for a regression model.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Requirements](#requirements)
4. [How to Run](#how-to-run)
5. [Workflow (Step by Step)](#workflow-step-by-step)
6. [Key Findings](#key-findings)
7. [Possible Next Steps](#possible-next-steps)

---

## Project Overview

| Item | Details |
|------|---------|
| Notebook | `EDA (1).ipynb` |
| Environment | Google Colab (Python 3) |
| Dataset | `insurance.csv` |
| Rows × Columns | 1338 × 7 (original) → 1338 × 10 (after encoding) |
| Target variable | `charges` |

---

## Dataset

| Column | Type | Description |
|--------|------|-------------|
| `age` | int | Age of the primary beneficiary (18–64) |
| `sex` → `gender` | categorical | `male` / `female` (renamed in the notebook) |
| `bmi` | float | Body Mass Index (15.96–53.13) |
| `children` | int | Number of dependents covered (0–5) |
| `smoker` | categorical | `yes` / `no` |
| `region` | categorical | `northeast`, `northwest`, `southeast`, `southwest` |
| `charges` | float | Individual medical costs billed by insurance (**target**) |

---

## Requirements

- Python 3.8+
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `scipy`

Install with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

---

## How to Run

1. Upload `insurance.csv` to your Google Drive (`MyDrive/insurance.csv`).
2. Open `EDA (1).ipynb` in Google Colab.
3. Run all cells in order (`Runtime → Run all`) and grant Drive access when prompted.

> **Running locally?** Remove the `google.colab` / `drive.mount` cell and change the file path to a local one, e.g. `pd.read_csv('insurance.csv')`.

---

## Workflow (Step by Step)

### Step 1 — Import Libraries
Imports `numpy`, `pandas`, `matplotlib`, `seaborn`, and silences warnings with `warnings.filterwarnings('ignore')`.

### Step 2 — Mount Google Drive
Uses `google.colab.drive.mount('/content/drive')` to access the dataset stored in Google Drive.

### Step 3 — Load the Data
Reads `insurance.csv` into a DataFrame (`df`) and previews it with `df.head()`.

### Step 4 — Check for Missing Values
- `df.isnull()` and `df.isnull().sum()` are used to check for nulls.
- **Result:** no missing values in any column.

### Step 5 — Rename Column
Renames `sex` → `gender` using `df.rename(...)` for clarity. Verified with `df.columns`.

### Step 6 — Explore Categorical Columns
`value_counts()` is used to inspect category balance:
- **gender:** male 676, female 662 (well balanced)
- **region:** southeast 364, southwest 325, northwest 325, northeast 324

### Step 7 — Encode Categorical Variables
| Column | Method | Mapping |
|--------|--------|---------|
| `gender` | Label encoding | male → 1, female → 0 |
| `smoker` | Label encoding | yes → 1, no → 0 |
| `region` | One-hot encoding (`pd.get_dummies`) | creates `region_northeast`, `region_northwest`, `region_southeast`, `region_southwest` |

### Step 8 — Univariate Analysis
- **Histograms with KDE** for `age`, `gender`, `bmi`, `children`, `smoker`, `charges` to view distributions.
- **Count plot** for `children`.
- **Box plots** for each of these columns to spot outliers.

### Step 9 — Correlation Heatmap
A seaborn heatmap of `df.corr()` shows linear relationships between all numeric features.

### Step 10 — Summary Statistics & Data Info
- `df.describe()` gives count, mean, std, min, quartiles, and max for every column.
- `df.info()` confirms 1338 non-null entries and correct data types (all numeric after encoding).

### Step 11 — Feature Scaling
`StandardScaler` (zero mean, unit variance) is applied to the continuous columns:
`age`, `bmi`, `children`.

### Step 12 — Categorical Feature Counts
Count plots for `gender`, `smoker`, `children`, and all four `region_*` columns, to check class balance.

### Step 13 — Bivariate Analysis (vs. Charges)
- **Box plots** of `charges` by `gender`, `smoker`, and `children`.
- **Correlation heatmap** (annotated, `coolwarm` colormap) across all encoded features.

### Step 14 — Statistical Test (Pearson Correlation)
Tests the relationship of each numeric feature with `charges` at **α = 0.05**.

- **H₀:** there is no linear correlation between the feature and `charges`.
- **H₁:** there is a linear correlation.

| Feature | Pearson r | p-value | Decision |
|---------|-----------|---------|----------|
| age | 0.299 | ≈ 0.0 | Reject H₀ (keep) |
| bmi | 0.198 | ≈ 0.0 | Reject H₀ (keep) |
| children | 0.068 | 0.0129 | Reject H₀ (keep) |

---

 Key Findings

- The dataset is **clean**: 1338 rows with **no missing values**.
- Gender and region are **evenly distributed**, so there is no major class imbalance.
- All three numeric features (`age`, `bmi`, `children`) have a **statistically significant** linear relationship with `charges`.
- `age` has the strongest correlation among the three (r ≈ 0.30), followed by `bmi` (r ≈ 0.20); `children` is weak (r ≈ 0.07).
- `smoker` is a binary feature that the notebook visualises against charges via box plots and the correlation heatmap; check these plots for its effect on cost.

 Possible Next Steps

- Handle outliers in `bmi` and `charges` (visible in the box plots).
- Apply a log transform to `charges` to reduce right skew.
- Test the categorical features (`gender`, `smoker`, `region`) against `charges` using a t-test / ANOVA.
- Split the data into train/test sets and build regression models (Linear Regression, Random Forest, XGBoost).
- Evaluate with R², MAE, and RMSE.

- Scaling was applied to the whole dataset before any train/test split. When building a model, fit the scaler on the training set only to avoid data leakage.
- The `region` value-count cell is repeated several times in the notebook; the duplicates can be safely removed.
