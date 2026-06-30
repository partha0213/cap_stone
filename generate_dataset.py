"""
generate_dataset.py
Generates a realistic synthetic housing dataset for the capstone EDA project.
Run this script once to produce housing_data.csv.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
n = 1500

# --- Core numeric features ---
lot_area       = np.random.lognormal(mean=9.0, sigma=0.5, size=n).astype(int)          # sq ft, right-skewed
gr_liv_area    = np.random.normal(1500, 400, size=n).clip(500, 4000).astype(int)       # above-grade living area
year_built     = np.random.randint(1900, 2023, size=n)
overall_qual   = np.random.choice(range(1, 11), size=n,
                                   p=[0.01,0.03,0.05,0.09,0.14,0.20,0.20,0.14,0.09,0.05])
garage_cars    = np.random.choice([0, 1, 2, 3, 4], size=n, p=[0.05,0.15,0.55,0.20,0.05])
total_bsmt_sf  = np.random.normal(1000, 350, size=n).clip(0, 3000).astype(int)
full_bath      = np.random.choice([0, 1, 2, 3], size=n, p=[0.03,0.40,0.50,0.07])
bedrooms       = np.random.choice([1, 2, 3, 4, 5, 6], size=n,
                                   p=[0.03,0.15,0.45,0.28,0.07,0.02])

# Sale price — correlated with quality and size
noise          = np.random.normal(0, 15000, size=n)
sale_price     = (
    50000
    + overall_qual * 18000
    + gr_liv_area  * 75
    + total_bsmt_sf * 30
    + garage_cars  * 8000
    + (2023 - year_built) * -300
    + noise
).clip(50000, 800000).astype(int)

# Categorical columns
neighborhoods  = ['NorthAmes','CollgCr','OldTown','Edwards','Somerst',
                  'NridgHt','Gilbert','Sawyer','NWAmes','Mitchel']
neighborhood   = np.random.choice(neighborhoods, size=n,
                                   p=[0.14,0.13,0.11,0.10,0.09,0.09,0.08,0.08,0.09,0.09])

bldg_types     = ['1Fam','TwnhsE','Duplex','Twnhs','2fmCon']
bldg_type      = np.random.choice(bldg_types, size=n,
                                   p=[0.72,0.12,0.06,0.05,0.05])

house_styles   = ['1Story','2Story','1.5Fin','SLvl','SFoyer']
house_style    = np.random.choice(house_styles, size=n,
                                   p=[0.45,0.30,0.12,0.08,0.05])

sale_conditions = ['Normal','Abnorml','Partial','AdjLand','Alloca','Family']
sale_condition  = np.random.choice(sale_conditions, size=n,
                                    p=[0.82,0.07,0.06,0.01,0.02,0.02])

# Additional numeric — some with intentional skew
pool_area      = np.where(np.random.rand(n) < 0.03,
                          np.random.randint(100, 800, size=n), 0)   # ~97% zeros → extreme right skew
misc_val       = np.where(np.random.rand(n) < 0.05,
                          np.random.randint(500, 15000, size=n), 0) # similar

# A column intentionally stored as object (will need dtype correction)
overall_cond_str = np.random.choice(['5','6','7','4','3','8','9','2','1','10'], size=n)

# Introduce missing values
def add_nulls(arr, frac):
    arr = arr.astype(object)
    idx = np.random.choice(len(arr), size=int(frac * len(arr)), replace=False)
    arr[idx] = np.nan
    return arr

garage_cars_with_nulls   = add_nulls(garage_cars.astype(float),  0.05)
total_bsmt_sf_with_nulls = add_nulls(total_bsmt_sf.astype(float), 0.08)
lot_area_with_nulls      = add_nulls(lot_area.astype(float),      0.03)
pool_area_with_nulls     = add_nulls(pool_area.astype(float),     0.06)
misc_val_with_nulls      = add_nulls(misc_val.astype(float),      0.04)

# A column with > 20% nulls (will be flagged and dropped/ignored)
alley = add_nulls(
    np.random.choice(['Grvl','Pave'], size=n).astype(object), 0.93
)

# Introduce ~30 exact-duplicate rows
dup_indices = np.random.choice(n, size=30, replace=False)

df = pd.DataFrame({
    'Id':              range(1, n + 1),
    'Neighborhood':    neighborhood,
    'BldgType':        bldg_type,
    'HouseStyle':      house_style,
    'OverallQual':     overall_qual,
    'OverallCond':     overall_cond_str,          # dtype bug: should be int
    'YearBuilt':       year_built,
    'TotalBsmtSF':     total_bsmt_sf_with_nulls,
    'GrLivArea':       gr_liv_area,
    'FullBath':        full_bath,
    'BedroomAbvGr':    bedrooms,
    'GarageCars':      garage_cars_with_nulls,
    'PoolArea':        pool_area_with_nulls,
    'MiscVal':         misc_val_with_nulls,
    'LotArea':         lot_area_with_nulls,
    'SaleCondition':   sale_condition,
    'Alley':           alley,                      # >20% null column
    'SalePrice':       sale_price,
})

# Append duplicates
dup_rows = df.iloc[dup_indices].copy()
df = pd.concat([df, dup_rows], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)   # shuffle

df.to_csv('housing_data.csv', index=False)
print(f"Dataset saved: {df.shape[0]} rows × {df.shape[1]} cols")
print(df.dtypes)
