import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONNECT TO DATABASE
# =========================

conn = sqlite3.connect("data/gas_monitoring.db")

# Load table into dataframe
df = pd.read_sql_query("SELECT * FROM gas_monitoring", conn)

# Close database connection
conn.close()

# =========================
# BASIC INFORMATION
# =========================

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# =========================
# REMOVE DUPLICATES
# =========================

df = df.drop_duplicates()

# =========================
# HANDLE MISSING VALUES
# =========================

# Numerical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

for col in numerical_cols:
    df[col] = df[col].fillna(df[col].median())

# Categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# =========================
# REMOVE OUTLIERS USING IQR
# =========================

for col in numerical_cols:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df = df[
        (df[col] >= lower_bound) &
        (df[col] <= upper_bound)
    ]

# =========================
# CHECK RAW CATEGORICAL VALUES
# =========================

print(f"\n'HVAC Operation Mode' raw unique values ({df['HVAC Operation Mode'].nunique()} variants):")
print(sorted(df['HVAC Operation Mode'].unique()), '\n')

print(f"'Activity Level' raw unique values ({df['Activity Level'].nunique()} variants):")
print(sorted(df['Activity Level'].unique()))

# =========================
# VISUALISE CATEGORY ISSUES
# =========================

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# HVAC raw variant counts
hvac_raw = df['HVAC Operation Mode'].value_counts()

axes[0].barh(hvac_raw.index, hvac_raw.values, color='#e07b54')
axes[0].set_title('HVAC Mode — Raw Variants (Before Cleaning)')
axes[0].set_xlabel('Count')
axes[0].invert_yaxis()

# Activity raw variant counts
act_raw = df['Activity Level'].value_counts()

axes[1].barh(act_raw.index, act_raw.values, color='#6b8cba')
axes[1].set_title('Activity Level — Raw Variants (Before Cleaning)')
axes[1].set_xlabel('Count')
axes[1].invert_yaxis()

plt.tight_layout()

plt.suptitle(
    '⚠️ Data Quality Issue: Inconsistent Categorical Casing',
    fontsize=13,
    y=1.02,
    fontweight='bold'
)

plt.show()

print('\n📌 Implication: These must be normalised before group-based analysis.')

# =========================
# CLEAN CATEGORICAL VALUES
# =========================

df['HVAC_clean'] = (
    df['HVAC Operation Mode']
    .str.strip()
    .str.lower()
)

df['Activity_clean'] = (
    df['Activity Level']
    .str.strip()
    .str.lower()
    .str.replace('_', ' ', regex=False)
)

print("\nHVAC after cleaning:")
print(sorted(df['HVAC_clean'].unique()))

print("\nActivity after cleaning:")
print(sorted(df['Activity_clean'].unique()))

# =========================
# FINAL CHECK
# =========================

print("\nCleaned Dataset Shape:")
print(df.shape)

print("\nRemaining Missing Values:")
print(df.isnull().sum())

print("\nRemaining Duplicate Rows:")
print(df.duplicated().sum())

# =========================
# SAVE CLEANED DATASET
# =========================

df.to_csv("cleaned_gas_monitoring.csv", index=False)

print("\nCleaned dataset saved successfully.")