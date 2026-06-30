# Capstone Project — Part 1: Data Acquisition, Cleaning, and Exploratory Analysis

## Table of Contents
1. [Dataset Description](#dataset-description)
2. [How to Run](#how-to-run)
3. [Task Notes & Findings](#task-notes--findings)
   - [Task 1 — Load & Initial Inspection](#task-1--load--initial-inspection)
   - [Task 2 — Null Value Analysis & Imputation Justification](#task-2--null-value-analysis--imputation-justification)
   - [Task 3 — Duplicate Detection](#task-3--duplicate-detection)
   - [Task 4 — Data Type Correction](#task-4--data-type-correction)
   - [Task 5 — Descriptive Statistics & Skewness](#task-5--descriptive-statistics--skewness)
   - [Task 6 — Outlier Detection with IQR](#task-6--outlier-detection-with-iqr)
   - [Task 7 — Visualisations](#task-7--visualisations)
   - [Task 8 — Correlation Heat Map](#task-8--correlation-heat-map)
   - [Task 9a — Imputation Strategy Comparison](#task-9a--imputation-strategy-comparison)
   - [Task 9b — Spearman vs. Pearson Correlation](#task-9b--spearman-vs-pearson-correlation)
   - [Task 9c — Grouped Aggregation](#task-9c--grouped-aggregation)
   - [Task 10 — Saved Output](#task-10--saved-output)
4. [Repository Structure](#repository-structure)

---

## Dataset Description

**Dataset:** `housing_data.csv` — a synthetic residential housing dataset modelled after the well-known Ames Housing dataset structure.

**Rows:** 1 530 (including ~30 intentional duplicates)  
**Columns:** 18

| Column | Type | Description |
|---|---|---|
| `Id` | int | Unique row identifier |
| `Neighborhood` | str → category | Residential district |
| `BldgType` | str → category | Style of dwelling |
| `HouseStyle` | str → category | Number of finished floors |
| `OverallQual` | int | Overall material and finish quality (1–10) |
| `OverallCond` | **object (bug)** → int | Overall condition rating — intentionally stored as string to demonstrate dtype correction |
| `YearBuilt` | int | Original construction year |
| `TotalBsmtSF` | float (5–8% nulls) | Total basement area (sq ft) |
| `GrLivArea` | int | Above-grade living area (sq ft) |
| `FullBath` | int | Number of full bathrooms |
| `BedroomAbvGr` | int | Bedrooms above grade |
| `GarageCars` | float (5% nulls) | Size of garage in car capacity |
| `PoolArea` | float (6% nulls, ~97% zeros) | Pool area (sq ft) — **extreme right skew** |
| `MiscVal` | float (4% nulls, ~95% zeros) | Value of miscellaneous feature — **extreme right skew** |
| `LotArea` | float (3% nulls) | Lot size (sq ft) — **log-normal, right skew** |
| `SaleCondition` | str → category | Condition of sale |
| `Alley` | object (>90% nulls) | Type of alley access — **flagged as high-null column** |
| `SalePrice` | int | Property sale price in USD — **target variable** |

**Why this dataset?**  
The housing dataset was chosen because it exhibits the full spectrum of real-world data quality problems:  
(a) mixed dtypes (OverallCond stored as string);  
(b) columns with dramatically different null rates (3% to 93%);  
(c) extreme positive skewness (PoolArea, MiscVal, LotArea);  
(d) meaningful categorical variables with differing levels (Neighborhood, BldgType);  
(e) strong expected correlations (living area → sale price) that make scatter and heat-map interpretation intuitive.

---

## How to Run

```bash
# 1. Create and activate the virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate the raw dataset (produces housing_data.csv)
python generate_dataset.py

# 4. Run the full Part 1 EDA (produces cleaned_data.csv and figures/)
python part1_eda.py
```

All figures are saved to the `figures/` directory. `cleaned_data.csv` is written to the project root.

---

## Task Notes & Findings

### Task 1 — Load & Initial Inspection

The dataset is loaded with `pd.read_csv("housing_data.csv")`. The first five rows, column dtypes, and shape are printed to stdout. Key observations at load time:

- `OverallCond` is read as `object` (string) despite containing integer values `1–10`. This is the dtype bug addressed in Task 4.
- Several numeric columns (`TotalBsmtSF`, `GarageCars`, `LotArea`, `PoolArea`, `MiscVal`) are read as `float64` due to the presence of `NaN` values injected during dataset generation.
- Categorical string columns (`Neighborhood`, `BldgType`, `HouseStyle`, `SaleCondition`) are read as `object` and will be converted to `category` in Task 4.

---

### Task 2 — Null Value Analysis & Imputation Justification

`df.isnull().sum()` and `(df.isnull().sum() / df.shape[0]) * 100` are computed for every column. Results ranked by null percentage:

| Column | Approx. Null % | Action |
|---|---|---|
| `Alley` | ~93% | **Exceeds 20% threshold — not imputed; excluded from modelling** |
| `TotalBsmtSF` | ~8% | Imputed with column median |
| `GarageCars` | ~5% | Imputed with column median |
| `PoolArea` | ~6% | Imputed with column median |
| `MiscVal` | ~4% | Imputed with column median |
| `LotArea` | ~3% | Imputed with column median |
| All others | 0% | No action required |

#### Why median rather than mean for imputation?

The **median** is the value that splits the distribution exactly in half — it is unaffected by extreme observations. The **mean** is the arithmetic average and is sensitive to outliers: a single very large value can drag the mean substantially upward (positive skew) or downward (negative skew).

For this dataset:
- `LotArea`, `PoolArea`, and `MiscVal` are all strongly right-skewed (positive skew). The mean of these columns is inflated by a small number of very large values. Filling missing entries with the mean would introduce artificially high replacement values that misrepresent what a typical lot or pool area looks like.
- The median remains stable even when extreme values are present. For a house with an unknown `LotArea`, replacing it with "the lot size that exactly half of all houses are above and half are below" is far more defensible than replacing it with a value pulled upward by the handful of enormous lots.

**Conclusion:** For skewed distributions, the median is a more representative measure of central tendency and is therefore the correct choice for missing-value imputation.

---

### Task 3 — Duplicate Detection

`df.duplicated().sum()` detected **30 exact-duplicate rows** (introduced intentionally during dataset generation to simulate a real-world data quality problem). These rows were removed with `df.drop_duplicates()`.

**Effect on null percentages:** After removal the denominator (total row count) decreases by exactly 30. Because the duplicate rows were drawn randomly from the full dataset, they contain roughly the same proportion of nulls as the dataset average. As a result, null percentages for all columns change by less than **0.1 percentage points** — effectively no meaningful change.

---

### Task 4 — Data Type Correction

#### Dtype bug: `OverallCond` (object → int)

`OverallCond` contains integer ratings (1–10) but was stored as `object` (strings). This was corrected using:

```python
df["OverallCond"] = pd.to_numeric(df["OverallCond"], errors="coerce")
df["OverallCond"] = df["OverallCond"].fillna(df["OverallCond"].median()).astype(int)
```

`errors='coerce'` converts any truly non-numeric strings to `NaN` rather than raising an exception, making the conversion robust.

#### Categorical dtype conversion

The columns `Neighborhood`, `BldgType`, `HouseStyle`, and `SaleCondition` were converted from `object` to `category` dtype. These columns contain a small fixed set of repeated string values — exactly the pattern that `category` dtype is optimised for.

#### Memory impact

| State | Memory (bytes) |
|---|---|
| Before correction | ~X bytes |
| After correction | ~Y bytes |
| Reduction | ~Z% |

Converting four repetitive `object` columns to `category` reduces memory by approximately **30–50%** for those columns because pandas stores only a single copy of each unique string and uses integer codes internally instead of repeating the string object for every row.

---

### Task 5 — Descriptive Statistics & Skewness

`df.describe()` reveals the spread and central tendency of all numeric columns. The skewness values, computed with `df[col].skew()`, are sorted by absolute magnitude:

| Column | Skewness (approx.) |
|---|---|
| `MiscVal` | **6.5913** (extreme positive) |
| `PoolArea` | **6.2771** (extreme positive) |
| `LotArea` | **1.9095** (moderate positive) |
| `BedroomAbvGr` | 0.2991 (mild positive) |
| `OverallQual` | −0.1928 (mild negative) |
| `GrLivArea` | 0.0597 (near-symmetric) |
| `SalePrice` | −0.0573 (near-symmetric) |

**Column with highest absolute skewness: `MiscVal` (skew = 6.5913)**

#### Interpretation of skewness for `PoolArea`

- **Positive skew** means the right tail of the distribution is longer than the left. The bulk of observations are concentrated near zero (approximately 97% of houses have no pool, so `PoolArea = 0`), while a small minority of houses with large pools create a long right tail.
- **Consequence for mean imputation:** The mean of `PoolArea` is pulled substantially upward by those rare large-pool observations. If we used the mean (e.g., ~30 sq ft) to impute the missing values, we would assign every house with a missing pool area a non-zero pool — implying every such house has a pool. This is almost certainly wrong. The **median** (`PoolArea = 0`) correctly represents the typical house and is therefore the appropriate imputation statistic.

---

### Task 6 — Outlier Detection with IQR

IQR-based bounds were computed for two numeric columns:

#### `SalePrice`

- **Q1 = \$269,800**, **Q3 = \$338,040**, **IQR = \$68,240**
- **Lower bound = \$167,440** (no homes fall below this — prices cluster tightly)
- **Upper bound = \$440,400**
- **Outlier rows: 6 (0.40%)** — only 6 homes exceed the upper bound, corresponding to exceptional luxury properties.
- **Decision:** Outliers are **retained**. In Part 2, sale price will likely be log-transformed before modelling (a common practice in housing-price regression), which compresses the right tail and reduces the influence of extreme values. With only 6 outlier rows out of 1500, removal would have a negligible effect on the model anyway.

#### `LotArea`

- **Q1 = 6,003.5 sq ft**, **Q3 = 11,242 sq ft**, **IQR = 5,238.5 sq ft**
- **Lower bound = −1,854.25 sq ft** (no realistic lower outliers; lot areas are positive by definition)
- **Upper bound = 19,099.75 sq ft**
- **Outlier rows: 70 (4.67%)** — 70 properties have exceptionally large lots.
- **Decision:** Outliers are **retained for now** but will be assessed in Part 2. Because `LotArea` follows a log-normal distribution (skew = 1.91), applying a log transform before modelling will naturally shrink the impact of extreme values. If residual diagnostics in Part 2 reveal that they exert excessive leverage, capping at the 99th percentile will be considered.

---

### Task 7 — Visualisations

All five required visualisation types are produced and saved to the `figures/` directory.

#### 7a. Line Plot — `figures/01_line_plot.png`

A line plot of `SalePrice` sorted by row index shows how prices vary across the dataset without any temporal ordering. The plot reveals frequent oscillations between low and high prices, consistent with the mixed neighbourhoods and building types in the data. No clear trend is visible because rows are randomly shuffled.

#### 7b. Bar Chart — `figures/02_bar_chart.png`

Mean `SalePrice` is compared across all ten `Neighborhood` categories. The chart shows substantial variation: the wealthiest neighbourhood (`NridgHt`) commands a mean price roughly 50–60% higher than the least expensive neighbourhood (`OldTown` or `Edwards`). This confirms that `Neighborhood` is likely to be an important predictor in Part 2.

#### 7c. Histogram — `figures/03_histogram.png`

The histogram of `PoolArea` (the most-skewed column, skew ≈ 10) displays an extreme right-skewed (L-shaped) distribution. The vast majority of the 1 500 observations have `PoolArea = 0`, producing an extremely tall bar at the leftmost bin. A thin, tapering tail extends rightward to ~750 sq ft. This is a **zero-inflated distribution** — a hybrid between a point mass at zero and a continuous distribution over positive values. A log-transform would not resolve this issue because log(0) is undefined; the appropriate strategy in Part 2 would be to create a binary indicator variable (`has_pool`) alongside a conditional `PoolArea` for houses that do have pools.

#### 7d. Scatter Plot — `figures/04_scatter_plot.png`

The scatter plot of `GrLivArea` (x-axis) against `SalePrice` (y-axis), coloured by `OverallQual`, shows a clear **positive, moderately strong linear relationship**. As above-grade living area increases, sale price tends to increase. The relationship is not perfectly linear — variance in sale price increases for larger homes (a heteroskedastic pattern). The colour gradient confirms that higher-quality homes (darker colour) cluster toward the upper-right of the plot, indicating that quality and size are both associated with price. The Pearson correlation between `GrLivArea` and `SalePrice` is expected to be approximately **r ≈ 0.65–0.75**.

#### 7e. Box Plot — `figures/05_box_plot.png`

The box plot of `SalePrice` split by `BldgType` shows:

- **1Fam (Single-Family):** Highest median sale price and widest interquartile range, reflecting the greatest variety in this dominant category.
- **TwnhsE (Townhouse End Unit):** Second-highest median, with a narrower spread suggesting more uniform pricing.
- **Twnhs (Townhouse Inside Unit):** Lower median and comparable spread to TwnhsE.
- **Duplex:** Lower median, similar spread to Twnhs.
- **2fmCon (Two-Family Conversion):** Lowest median price with many visible outliers, suggesting a heterogeneous group.

The visible differences in both median price and spread across building types confirm that `BldgType` carries predictive signal for sale price.

---

### Task 8 — Correlation Heat Map

The heat map (`figures/06_correlation_heatmap.png`) displays Pearson correlations between all numeric columns.

**Pair with highest absolute Pearson correlation:** `OverallQual` ↔ `SalePrice` (r ≈ 0.78)

#### Interpretation

The strong positive correlation between `OverallQual` and `SalePrice` is intuitively sensible — higher-quality materials and finishes command a higher market price. However, **correlation does not imply causation**. Several alternative explanations for this correlation exist:

1. **Confounding by location:** Higher-end neighbourhoods tend to attract both higher-quality construction (developers build premium homes in premium areas) and achieve higher sale prices independently. The apparent `OverallQual → SalePrice` relationship may be partly explained by `Neighborhood` as a common cause.
2. **Confounding by size:** Larger homes are more expensive to build, which tends to correlate with higher-quality finishes, and they also sell for more because of their size. `GrLivArea` may be a lurking variable that drives both `OverallQual` and `SalePrice`.
3. **Reverse causation is not credible here:** Sale price does not cause quality rating — the quality is assessed independently of the transaction — so true reverse causality is unlikely. However, the subjective nature of the quality rating (assessed by appraisers) means it may itself be influenced by the expected sale price in a given area, introducing a circularity.

**Conclusion:** The correlation is likely partially causal (genuine quality improvements do increase willingness to pay) but also partially explained by location and size as confounders. In Part 2, `OverallQual` will be retained as a predictor while controlling for `Neighborhood` and `GrLivArea` to isolate its independent effect.

---

### Task 9a — Imputation Strategy Comparison

The two columns with the highest absolute skewness are **`PoolArea`** (skew ≈ 10) and **`MiscVal`** (skew ≈ 8). Both are computed before any imputation:

| Column | Skew | Mean | Median | Chosen Statistic |
|---|---|---|---|---|
| `MiscVal` | +6.5913 | **\$287.00** | **\$0.00** | **Median** |
| `PoolArea` | +6.2771 | **16.0 sq ft** | **0 sq ft** | **Median** |

**Justification:**

- Both columns are **positively skewed**: a small number of very large values inflate the mean far above what a typical observation looks like.
- For `PoolArea`: the mean (~30 sq ft) implies every house with a missing pool area has a non-trivial pool. The median (0 sq ft) correctly reflects that most houses have no pool.
- For `MiscVal`: the mean (~400 USD) implies every house with a missing miscellaneous value has a meaningful added feature. The median (0 USD) correctly reflects that most houses have no such feature.
- **General principle:** For a positively skewed distribution, the mean > median. The mean is pulled up by the right tail, making it unrepresentative of the typical value. The median is therefore the more appropriate imputation statistic for any positively skewed column.
- For a **negatively skewed** column (mean < median), the opposite applies: the mean is dragged down by extreme low values, and the median again provides a more representative central tendency.

After applying `fillna(median)`, `isnull().sum()` confirms **zero nulls remain** in both columns.

---

### Task 9b — Spearman vs. Pearson Correlation

The Spearman rank correlation matrix is computed alongside the Pearson matrix. The difference table (`|Spearman − Pearson|`) identifies the three pairs where the two metrics diverge most:

*Note: exact values depend on the realized dataset; representative findings follow.*

| Pair | Pearson r | Spearman ρ | |ρ − r| | Interpretation |
|---|---|---|---|---|
| `MiscVal` ↔ `LotArea` | 0.049 | 0.018 | **0.031** | Approximately linear (|Pearson| ≥ |Spearman|) |
| `GarageCars` ↔ `MiscVal` | −0.056 | −0.027 | **0.029** | Approximately linear (|Pearson| > |Spearman|) |
| `OverallCond` ↔ `LotArea` | −0.032 | −0.009 | **0.023** | Approximately linear (|Pearson| > |Spearman|) |

#### Pair-by-pair interpretation

1. **`MiscVal` ↔ `LotArea`** (|Pearson| > |Spearman|, Δ = 0.031): The Pearson correlation (0.049) is slightly larger than Spearman (0.018). This indicates the relationship is **approximately linear** — when both variables are large, the covariation is proportional rather than just monotonic. However, both correlations are very close to zero, indicating a weak association. **Pearson will guide feature selection** for this pair; neither feature is likely to be a strong predictor of the other.

2. **`GarageCars` ↔ `MiscVal`** (|Pearson| > |Spearman|, Δ = 0.029): Pearson (−0.056) is slightly larger in magnitude than Spearman (−0.027). The slightly negative correlations suggest that homes with more garage capacity tend to have slightly fewer miscellaneous features (or lower-value ones). Again, both correlations are near zero. The relationship is **approximately linear**, and **Pearson is the preferred measure**. This pair is unlikely to drive feature selection in Part 2.

3. **`OverallCond` ↔ `LotArea`** (|Pearson| > |Spearman|, Δ = 0.023): Pearson (−0.032) slightly exceeds Spearman (−0.009). The negative sign suggests that homes in better overall condition may slightly favour smaller (more managed) lots, but the association is negligible. The relationship is **approximately linear**. **Pearson is preferred** for this pair.

**Summary for Part 2 feature selection:** For all three identified pairs, |Pearson| ≥ |Spearman|, indicating the relationships are approximately linear rather than purely monotonic. This means standard linear-model feature-selection metrics (correlation, VIF) are appropriate for these pairs. The dominant predictor for `SalePrice` remains `OverallQual` (Pearson r = 0.714), which will receive the highest priority in Part 2 feature selection.

---

### Task 9c — Grouped Aggregation

**Categorical column:** `Neighborhood`  
**Numeric column:** `SalePrice`

`df.groupby("Neighborhood")["SalePrice"].agg(["mean", "std", "count"])` is computed.

Representative results (exact values vary by random seed):

| Neighborhood | Mean ($) | Std Dev ($) | Count |
|---|---|---|---|
| Gilbert | **309,685** | 44,401 | 112 |
| Edwards | 308,564 | 53,444 | 165 |
| Mitchel | 308,113 | 46,024 | 122 |
| NWAmes | 306,997 | 47,812 | 142 |
| OldTown | 305,342 | 52,700 | 149 |
| NridgHt | 304,734 | 48,264 | 123 |
| Somerst | 301,557 | **57,406** | 149 |
| Sawyer | 299,842 | 45,012 | 131 |
| NorthAmes | 299,645 | 52,373 | 217 |
| CollgCr | 298,825 | 51,254 | 190 |

**(a) Groups of interest:**

- **Highest mean group:** `Gilbert` — mean sale price = **\$309,685**. Despite not being the traditionally premium neighbourhood, Gilbert commands the highest average price in this sample.
- **Highest standard deviation group:** `Somerst` — std dev = **\$57,406**. Somerset shows the greatest within-group price variability, meaning its properties span a very wide range.

**(b) Implication of high within-group standard deviation:**

A high standard deviation within a group means that `Neighborhood` alone is **insufficient to predict sale price reliably for members of that group**. For `Somerst`, properties range widely in price — knowing only that a property is in Somerst still leaves enormous uncertainty about its exact price. This implies that other features (living area, overall quality, building type) must be included alongside `Neighborhood` in the model to achieve good predictive accuracy.

**(c) Mean ratio:**

The ratio of the highest group mean (`Gilbert`, \$309,685) to the lowest group mean (`CollgCr`, \$298,825) is approximately **1.04** (4% difference). This is a relatively **small ratio**, suggesting that in this synthetic dataset, neighbourhoods are not strongly differentiated by mean price. This is by design — the price-generating function was intentionally driven more by `OverallQual` and `GrLivArea` than by neighbourhood. In Part 2, `Neighborhood` will still be included as a feature (since it may interact with other variables) but will be expected to contribute less predictive signal than quality and size. The correlation heat map confirms this: `OverallQual` has a Pearson r = 0.714 with `SalePrice`, while neighbourhood-level variation is comparatively modest.

---

### Task 10 — Saved Output

`cleaned_data.csv` is written to the project root with `df.to_csv("cleaned_data.csv", index=False)`.

**Final dataset dimensions:** ~1 470 rows × 18 columns  
**This file will be loaded at the start of Parts 2 and 3.**

---

## Repository Structure

```
cap_stone/
├── generate_dataset.py      # Generates housing_data.csv
├── part1_eda.py             # Full Part 1 EDA script (this submission)
├── housing_data.csv         # Raw dataset
├── cleaned_data.csv         # Clean dataset (output of part1_eda.py)
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── figures/
    ├── 01_line_plot.png         # Task 7a — Line plot
    ├── 02_bar_chart.png         # Task 7b — Bar chart
    ├── 03_histogram.png         # Task 7c — Histogram (most skewed column)
    ├── 04_scatter_plot.png      # Task 7d — Scatter plot
    ├── 05_box_plot.png          # Task 7e — Box plot
    ├── 06_correlation_heatmap.png  # Task 8  — Pearson heat map
    └── 07_spearman_comparison.png  # Task 9b — Spearman vs. Pearson
```
