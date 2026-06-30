"""
part1_eda.py
============
Part 1 — Data Acquisition, Cleaning, and Exploratory Analysis
Capstone Project · Housing Price Dataset

Run: python part1_eda.py  (inside the project venv)
Outputs:
  • cleaned_data.csv
  • figures/  (all 5 visualisation types + correlation heatmap)
"""

# ── Standard imports ──────────────────────────────────────────────────────────
import os, warnings
import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend → saves to file
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
os.makedirs("figures", exist_ok=True)

sns.set_theme(style="darkgrid", palette="muted")
FIGSIZE = (10, 6)

# ═══════════════════════════════════════════════════════════════════════════════
# TASK 0 — Generate dataset (idempotent: skipped if file already exists)
# ═══════════════════════════════════════════════════════════════════════════════
if not os.path.exists("housing_data.csv"):
    exec(open("generate_dataset.py").read())

# ═══════════════════════════════════════════════════════════════════════════════
# TASK 1 — Load & initial inspection
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TASK 1 — LOAD & INITIAL INSPECTION")
print("="*70)

df = pd.read_csv("housing_data.csv")

print("\n── First 5 rows ──")
print(df.head())

print("\n── Column dtypes ──")
print(df.dtypes)

print(f"\n── Shape: {df.shape[0]} rows × {df.shape[1]} columns ──")

# ═══════════════════════════════════════════════════════════════════════════════
# TASK 2 — Null value analysis
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TASK 2 — NULL VALUE ANALYSIS")
print("="*70)

null_count  = df.isnull().sum()
null_pct    = (df.isnull().sum() / df.shape[0]) * 100
null_report = pd.DataFrame({"null_count": null_count, "null_pct": null_pct})
null_report = null_report.sort_values("null_pct", ascending=False)

print("\nNull count & percentage per column:")
print(null_report.to_string())

HIGH_NULL_THRESHOLD = 20.0
high_null_cols = null_report[null_report["null_pct"] > HIGH_NULL_THRESHOLD].index.tolist()
low_null_cols  = null_report[
    (null_report["null_pct"] > 0) & (null_report["null_pct"] <= HIGH_NULL_THRESHOLD)
].index.tolist()

print(f"\nColumns EXCEEDING {HIGH_NULL_THRESHOLD}% null rate → will NOT impute:")
print(high_null_cols)

print(f"\nColumns BELOW {HIGH_NULL_THRESHOLD}% null rate → impute numeric with median:")
print(low_null_cols)

# Impute numeric columns below 20% threshold with median
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in low_null_cols:
    if col in numeric_cols:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"  Filled '{col}' nulls with median = {median_val:.2f}")

# ═══════════════════════════════════════════════════════════════════════════════
# TASK 3 — Duplicate detection and removal
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TASK 3 — DUPLICATE DETECTION & REMOVAL")
print("="*70)

n_before = df.shape[0]
n_dups   = df.duplicated().sum()
print(f"\nDuplicate rows found: {n_dups}")

null_pct_before = (df.isnull().sum() / df.shape[0] * 100).round(4)
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
n_after  = df.shape[0]

null_pct_after = (df.isnull().sum() / df.shape[0] * 100).round(4)
print(f"Rows before: {n_before}  →  after: {n_after}  (removed {n_before - n_after})")

print("\nChange in null percentage after deduplication:")
diff = (null_pct_after - null_pct_before).round(4)
print(diff[diff != 0].to_string() if diff[diff != 0].shape[0] > 0
      else "  No change in null percentages.")

# ═══════════════════════════════════════════════════════════════════════════════
# TASK 4 — Data type correction
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TASK 4 — DATA TYPE CORRECTION")
print("="*70)

mem_before = df.memory_usage(deep=True).sum()
print(f"\nMemory usage BEFORE corrections: {mem_before:,} bytes")

# Bug fix: OverallCond is stored as object but should be int
print("\n  → Converting 'OverallCond' (object → int) via pd.to_numeric(errors='coerce')")
df["OverallCond"] = pd.to_numeric(df["OverallCond"], errors="coerce")
df["OverallCond"] = df["OverallCond"].fillna(df["OverallCond"].median()).astype(int)

# Convert TotalBsmtSF and GarageCars (now imputed) to proper numeric types
df["TotalBsmtSF"] = pd.to_numeric(df["TotalBsmtSF"], errors="coerce")
df["GarageCars"]  = pd.to_numeric(df["GarageCars"],  errors="coerce")

# Convert repetitive string columns to category dtype
cat_cols_to_convert = ["Neighborhood", "BldgType", "HouseStyle", "SaleCondition"]
for col in cat_cols_to_convert:
    df[col] = df[col].astype("category")
    print(f"  → Converted '{col}' to category dtype")

mem_after = df.memory_usage(deep=True).sum()
print(f"\nMemory usage AFTER  corrections: {mem_after:,} bytes")
print(f"Memory saved: {mem_before - mem_after:,} bytes  "
      f"({(1 - mem_after/mem_before)*100:.1f}% reduction)")

print("\nUpdated dtypes:")
print(df.dtypes)

# ═══════════════════════════════════════════════════════════════════════════════
# TASK 5 — Descriptive statistics & skewness
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TASK 5 — DESCRIPTIVE STATISTICS & SKEWNESS")
print("="*70)

num_df = df.select_dtypes(include=[np.number]).drop(columns=["Id"], errors="ignore")

print("\n── df.describe() ──")
print(num_df.describe().round(2).to_string())

skewness = num_df.apply(lambda c: c.skew()).sort_values(key=abs, ascending=False)
print("\n── Skewness per numeric column ──")
print(skewness.round(4).to_string())

most_skewed     = skewness.abs().idxmax()
most_skewed_val = skewness[most_skewed]
print(f"\nColumn with HIGHEST absolute skewness: '{most_skewed}'  (skew = {most_skewed_val:.4f})")

# ═══════════════════════════════════════════════════════════════════════════════
# TASK 6 — Outlier detection with IQR
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TASK 6 — OUTLIER DETECTION (IQR method)")
print("="*70)

iqr_cols = ["SalePrice", "LotArea"]
for col in iqr_cols:
    q1  = df[col].quantile(0.25)
    q3  = df[col].quantile(0.75)
    iqr = q3 - q1
    lb  = q1 - 1.5 * iqr
    ub  = q3 + 1.5 * iqr
    n_out = ((df[col] < lb) | (df[col] > ub)).sum()
    print(f"\n  Column: {col}")
    print(f"    Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}")
    print(f"    Lower bound={lb:.2f}, Upper bound={ub:.2f}")
    print(f"    Outlier rows: {n_out} ({n_out/df.shape[0]*100:.2f}%)")
    print(f"    → Outliers RETAINED (not dropped); see README for strategy.")

# ═══════════════════════════════════════════════════════════════════════════════
# TASK 7 — Five visualisations
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TASK 7 — VISUALISATIONS")
print("="*70)

# ── 7a. Line plot: SalePrice sorted by index ─────────────────────────────────
fig, ax = plt.subplots(figsize=FIGSIZE)
plot_df = df.sort_index()
ax.plot(plot_df.index, plot_df["SalePrice"], color="#4C72B0", linewidth=0.7, alpha=0.8)
ax.set_title("Sale Price by Row Index", fontsize=14, fontweight="bold")
ax.set_xlabel("Row Index")
ax.set_ylabel("Sale Price (USD)")
plt.tight_layout()
plt.savefig("figures/01_line_plot.png", dpi=150)
plt.close()
print("  Saved: figures/01_line_plot.png")

# ── 7b. Bar chart: mean SalePrice by Neighborhood ────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
mean_by_neigh = df.groupby("Neighborhood", observed=True)["SalePrice"].mean().sort_values(ascending=False)
colors = sns.color_palette("muted", len(mean_by_neigh))
ax.bar(mean_by_neigh.index, mean_by_neigh.values, color=colors, edgecolor="white", linewidth=0.5)
ax.set_title("Mean Sale Price by Neighborhood", fontsize=14, fontweight="bold")
ax.set_xlabel("Neighborhood")
ax.set_ylabel("Mean Sale Price (USD)")
ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig("figures/02_bar_chart.png", dpi=150)
plt.close()
print("  Saved: figures/02_bar_chart.png")

# ── 7c. Histogram of most-skewed column ──────────────────────────────────────
fig, ax = plt.subplots(figsize=FIGSIZE)
sns.histplot(df[most_skewed].dropna(), bins=20, kde=True,
             color="#C44E52", edgecolor="white", linewidth=0.5, ax=ax)
ax.set_title(f"Distribution of '{most_skewed}' (skew = {most_skewed_val:.2f})",
             fontsize=14, fontweight="bold")
ax.set_xlabel(most_skewed)
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig("figures/03_histogram.png", dpi=150)
plt.close()
print(f"  Saved: figures/03_histogram.png  (column: {most_skewed})")

# ── 7d. Scatter plot: GrLivArea vs SalePrice ─────────────────────────────────
fig, ax = plt.subplots(figsize=FIGSIZE)
sns.scatterplot(data=df, x="GrLivArea", y="SalePrice",
                hue="OverallQual", palette="viridis",
                alpha=0.5, linewidth=0, ax=ax)
ax.set_title("Above-Grade Living Area vs. Sale Price\n(coloured by Overall Quality)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Above-Grade Living Area (sq ft)")
ax.set_ylabel("Sale Price (USD)")
plt.tight_layout()
plt.savefig("figures/04_scatter_plot.png", dpi=150)
plt.close()
print("  Saved: figures/04_scatter_plot.png")

# ── 7e. Box plot: SalePrice by BldgType ──────────────────────────────────────
fig, ax = plt.subplots(figsize=FIGSIZE)
order = df.groupby("BldgType", observed=True)["SalePrice"].median() \
          .sort_values(ascending=False).index
sns.boxplot(data=df, x="BldgType", y="SalePrice",
            order=order, palette="Set2", ax=ax)
ax.set_title("Sale Price Distribution by Building Type", fontsize=14, fontweight="bold")
ax.set_xlabel("Building Type")
ax.set_ylabel("Sale Price (USD)")
plt.tight_layout()
plt.savefig("figures/05_box_plot.png", dpi=150)
plt.close()
print("  Saved: figures/05_box_plot.png")

# ═══════════════════════════════════════════════════════════════════════════════
# TASK 8 — Pearson correlation heat map
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TASK 8 — PEARSON CORRELATION HEAT MAP")
print("="*70)

pearson_corr = num_df.corr(method="pearson")
print("\nPearson correlation matrix:")
print(pearson_corr.round(3).to_string())

fig, ax = plt.subplots(figsize=(12, 9))
mask = np.triu(np.ones_like(pearson_corr, dtype=bool))
sns.heatmap(pearson_corr, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", center=0, linewidths=0.5,
            linecolor="white", ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Pearson Correlation Heat Map — Numeric Features",
             fontsize=14, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("figures/06_correlation_heatmap.png", dpi=150)
plt.close()
print("  Saved: figures/06_correlation_heatmap.png")

# Find highest absolute correlation (excluding self-correlation)
corr_upper = pearson_corr.where(~mask)
top_corr   = corr_upper.stack().abs().idxmax()
top_val    = corr_upper.loc[top_corr]
print(f"\nHighest absolute Pearson correlation: {top_corr[0]} ↔ {top_corr[1]}  "
      f"(r = {top_val:.3f})")

# ═══════════════════════════════════════════════════════════════════════════════
# TASK 9a — Imputation strategy comparison (mean vs median for top-2 skewed cols)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TASK 9a — IMPUTATION STRATEGY COMPARISON")
print("="*70)

top2_skewed = skewness.abs().nlargest(2).index.tolist()
print(f"\nTwo most-skewed columns: {top2_skewed}")

for col in top2_skewed:
    col_mean   = df[col].mean()
    col_median = df[col].median()
    col_skew   = df[col].skew()
    print(f"\n  Column: {col}  (skew = {col_skew:.4f})")
    print(f"    Mean   = {col_mean:.4f}")
    print(f"    Median = {col_median:.4f}")
    # Apply chosen strategy (median — justified by skewness)
    nulls_remaining = df[col].isnull().sum()
    if nulls_remaining > 0:
        df[col].fillna(col_median, inplace=True)
        print(f"    → Filled {nulls_remaining} remaining nulls with MEDIAN")
    print(f"    Nulls remaining after fillna: {df[col].isnull().sum()}")

# Final null check
print("\nFinal isnull().sum() for all columns:")
print(df.isnull().sum().to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# TASK 9b — Spearman rank correlation
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TASK 9b — SPEARMAN RANK CORRELATION")
print("="*70)

spearman_corr = num_df.corr(method="spearman")
print("\nSpearman correlation matrix:")
print(spearman_corr.round(3).to_string())

# Difference matrix
mask_lower = np.tril(np.ones_like(pearson_corr, dtype=bool), k=-1)
diff_matrix = (spearman_corr - pearson_corr).abs()
diff_upper  = diff_matrix.where(
    np.triu(np.ones_like(diff_matrix, dtype=bool), k=1)
)
diff_stacked = diff_upper.stack().dropna()
top3_diff    = diff_stacked.nlargest(3)

print("\nTop-3 column pairs by |Spearman − Pearson|:")
print(top3_diff.round(4).to_string())

# Visualise difference heat map
fig, axes = plt.subplots(1, 3, figsize=(22, 7))

for ax, (corr_matrix, title) in zip(
    axes,
    [(pearson_corr, "Pearson"), (spearman_corr, "Spearman"),
     (diff_matrix,  "|Spearman − Pearson|")]
):
    cmap = "coolwarm" if "Pearson" in title or "Spearman" in title else "YlOrRd"
    center = 0 if "Pearson" in title or "Spearman" in title else None
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap=cmap,
                center=center, linewidths=0.3, linecolor="white",
                ax=ax, cbar_kws={"shrink": 0.7})
    ax.set_title(title, fontsize=13, fontweight="bold")

plt.suptitle("Pearson vs. Spearman Correlation Comparison", fontsize=15,
             fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("figures/07_spearman_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: figures/07_spearman_comparison.png")

# ═══════════════════════════════════════════════════════════════════════════════
# TASK 9c — Grouped aggregation
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TASK 9c — GROUPED AGGREGATION")
print("="*70)

cat_col = "Neighborhood"
num_col = "SalePrice"

grouped = df.groupby(cat_col, observed=True)[num_col].agg(["mean", "std", "count"])
grouped = grouped.sort_values("mean", ascending=False)
print(f"\nGrouped aggregation: {cat_col} → {num_col}")
print(grouped.round(2).to_string())

highest_mean_group = grouped["mean"].idxmax()
highest_std_group  = grouped["std"].idxmax()
lowest_mean_group  = grouped["mean"].idxmin()
mean_ratio         = grouped["mean"].max() / grouped["mean"].min()

print(f"\n  Highest-mean group  : {highest_mean_group}  "
      f"(mean = {grouped.loc[highest_mean_group, 'mean']:,.0f})")
print(f"  Highest-std  group  : {highest_std_group}  "
      f"(std  = {grouped.loc[highest_std_group,  'std']:,.0f})")
print(f"  Lowest-mean  group  : {lowest_mean_group}   "
      f"(mean = {grouped.loc[lowest_mean_group,  'mean']:,.0f})")
print(f"  Mean ratio (max/min): {mean_ratio:.2f}")

# ═══════════════════════════════════════════════════════════════════════════════
# TASK 10 — Save cleaned dataset
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TASK 10 — SAVE CLEANED DATASET")
print("="*70)

df.to_csv("cleaned_data.csv", index=False)
print(f"\ncleaned_data.csv saved: {df.shape[0]} rows × {df.shape[1]} columns")
print("\n✓ Part 1 complete — all outputs written to disk.")
