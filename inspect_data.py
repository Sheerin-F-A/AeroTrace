import pandas as pd
from pathlib import Path

csv_path = Path("data/Obfuscated-MalMem2022.csv")

if not csv_path.exists():
    print(f"ERROR: Dataset not found at {csv_path}")
    print("Check the filename inside the data/ folder.")
    raise SystemExit(1)

df = pd.read_csv(csv_path)

print("\n===== DATASET INFO =====")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\n===== COLUMN NAMES =====")
for i, column in enumerate(df.columns, start=1):
    print(f"{i}. {column}")

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== CLASS DISTRIBUTION =====")
if "Class" in df.columns:
    print(df["Class"].value_counts())
else:
    print("'Class' column not found.")