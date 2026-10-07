import pandas as pd
import re

CSV_PATH = "data/Obfuscated-MalMem2022.csv"

df = pd.read_csv(CSV_PATH)

print("=" * 70)
print("AEROTRACE DATASET SANITY CHECK")
print("=" * 70)

# ------------------------------------------------------------
# 1. Remove exact duplicates
# ------------------------------------------------------------

before = len(df)

df = df.drop_duplicates().reset_index(drop=True)

print(f"\nOriginal rows       : {before}")
print(f"Exact duplicates    : {before - len(df)}")
print(f"Unique rows         : {len(df)}")


# ------------------------------------------------------------
# 2. Inspect Category structure
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CATEGORY EXAMPLES")
print("=" * 70)

for value in df["Category"].drop_duplicates().head(20):
    print(value)


# ------------------------------------------------------------
# 3. Extract underlying sample/group ID
# ------------------------------------------------------------

def extract_group(category):
    """
    Example:

    Ransomware-Maze-abc123-1.raw
    Ransomware-Maze-abc123-2.raw

    become something like:

    Ransomware-Maze-abc123

    Benign stays Benign.
    """

    if category == "Benign":
        return "Benign"

    # Remove the final -number.raw
    return re.sub(r"-\d+\.raw$", "", str(category))


df["sample_group"] = df["Category"].apply(extract_group)


# ------------------------------------------------------------
# 4. Number of groups
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("GROUP INFORMATION")
print("=" * 70)

print(f"Unique Category values : {df['Category'].nunique()}")
print(f"Unique sample groups   : {df['sample_group'].nunique()}")


# ------------------------------------------------------------
# 5. Group sizes
# ------------------------------------------------------------

group_sizes = (
    df["sample_group"]
    .value_counts()
)

print("\nGroup size distribution:")
print(group_sizes.describe())


print("\nLargest groups:")

print(
    group_sizes
    .head(20)
    .to_string()
)


# ------------------------------------------------------------
# 6. Check whether groups contain multiple rows
# ------------------------------------------------------------

multi_row_groups = group_sizes[group_sizes > 1]

print("\nGroups with multiple rows:")
print(len(multi_row_groups))

print(
    "\nPercentage of rows belonging to multi-row groups:",
    round(
        df["sample_group"]
        .isin(multi_row_groups.index)
        .mean() * 100,
        2
    ),
    "%"
)


# ------------------------------------------------------------
# 7. Class distribution by group
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SAMPLE GROUP CLASS CONSISTENCY")
print("=" * 70)

group_class_counts = (
    df.groupby("sample_group")["Class"]
    .nunique()
)

mixed_groups = group_class_counts[
    group_class_counts > 1
]

print(
    f"Groups containing both Benign and Malware labels: "
    f"{len(mixed_groups)}"
)


# ------------------------------------------------------------
# 8. Check constant features
# ------------------------------------------------------------

numeric = df.select_dtypes(include="number")

constant_features = [
    column
    for column in numeric.columns
    if numeric[column].nunique() <= 1
]

print("\n" + "=" * 70)
print("CONSTANT FEATURES")
print("=" * 70)

if constant_features:
    for feature in constant_features:
        print(feature)
else:
    print("No constant features.")


# ------------------------------------------------------------
# 9. Very strong class separation
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FEATURE CLASS SEPARATION CHECK")
print("=" * 70)

numeric["Class_num"] = (
    df["Class"]
    .eq("Malware")
    .astype(int)
)

correlations = (
    numeric
    .corr(numeric_only=True)["Class_num"]
    .drop("Class_num")
    .abs()
    .sort_values(ascending=False)
)

print("\nTop 15 features correlated with the target:")

print(
    correlations
    .head(15)
    .to_string()
)


print("\n" + "=" * 70)
print("✅ SANITY CHECK COMPLETE")
print("=" * 70)