# ============================================================
# AeroTrace
# Leakage-Aware / Group-Aware Evaluation
# ============================================================

import json
import re
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. PATHS
# ============================================================

CSV_PATH = Path("data/Obfuscated-MalMem2022.csv")

RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")

RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(CSV_PATH)

print("=" * 70)
print("AEROTRACE - GROUP-AWARE EVALUATION")
print("=" * 70)

print(f"Original rows: {len(df)}")


# ============================================================
# 3. REMOVE DUPLICATES
# ============================================================

before = len(df)

df = df.drop_duplicates().reset_index(drop=True)

print(f"Duplicates removed: {before - len(df)}")
print(f"Clean rows: {len(df)}")


# ============================================================
# 4. TARGET
# ============================================================

df["y"] = (
    df["Class"]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("malware")
    .astype(int)
)


# ============================================================
# 5. MALWARE FAMILY
# ============================================================

df["family"] = (
    df["Category"]
    .astype(str)
    .str.strip()
    .str.split("-")
    .str[0]
)


# ============================================================
# 6. BUILD LEAKAGE-AWARE GROUP ID
# ============================================================

def make_group_id(category, row_index):
    """
    For malware:
        Ransomware-Ako-ABC-1.raw
        Ransomware-Ako-ABC-2.raw

    become the SAME group:

        Ransomware-Ako-ABC

    For benign:
        Category is simply 'Benign', so there is no
        underlying filename/group identifier in this CSV.

        Therefore every benign row gets its own group.
    """

    category = str(category).strip()

    if category == "Benign":
        return f"BENIGN_{row_index}"

    # Remove final -number.raw
    return re.sub(
        r"-\d+\.raw$",
        "",
        category,
    )


df["group_id"] = [
    make_group_id(
        category,
        index,
    )
    for index, category in enumerate(df["Category"])
]


# ============================================================
# 7. NUMERIC FEATURES
# ============================================================

excluded = {
    "Category",
    "Class",
    "y",
    "family",
    "group_id",
}

features = []

for column in df.columns:
    if (
        column not in excluded
        and pd.api.types.is_numeric_dtype(df[column])
    ):
        features.append(column)

X = df[features].copy()

X = X.fillna(0)

# Remove constant features
constant_features = [
    column
    for column in X.columns
    if X[column].nunique() <= 1
]

if constant_features:

    print("\nConstant features removed:")

    for feature in constant_features:
        print(" -", feature)

    X = X.drop(
        columns=constant_features
    )

    features = list(X.columns)


print("\nNumber of final features:", len(features))


# ============================================================
# 8. GROUP INFORMATION
# ============================================================

groups = df["group_id"]

print("\nUnique groups:", groups.nunique())

print(
    "Rows in largest groups:"
)

print(
    groups.value_counts()
    .head(10)
    .to_string()
)


# ============================================================
# 9. GROUP-AWARE TRAIN / TEST SPLIT
# ============================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

train_positions, test_positions = next(
    splitter.split(
        X,
        df["y"],
        groups=groups,
    )
)

X_train = X.iloc[train_positions]
X_test = X.iloc[test_positions]

y_train = df["y"].iloc[train_positions]
y_test = df["y"].iloc[test_positions]

train_groups = set(
    groups.iloc[train_positions]
)

test_groups = set(
    groups.iloc[test_positions]
)


# ============================================================
# 10. CHECK GROUP OVERLAP
# ============================================================

overlap = train_groups.intersection(
    test_groups
)

print("\n" + "=" * 70)
print("GROUP SPLIT AUDIT")
print("=" * 70)

print("Training rows:", len(X_train))
print("Testing rows :", len(X_test))

print("Training groups:", len(train_groups))
print("Testing groups :", len(test_groups))

print(
    "Cross-partition group overlap:",
    len(overlap),
)

if overlap:
    raise RuntimeError(
        "ERROR: train/test groups overlap!"
    )


# ============================================================
# 11. CLASS DISTRIBUTION
# ============================================================

print("\nTraining class distribution:")

print(
    y_train.value_counts()
    .rename({
        0: "Benign",
        1: "Malware",
    })
    .to_string()
)

print("\nTesting class distribution:")

print(
    y_test.value_counts()
    .rename({
        0: "Benign",
        1: "Malware",
    })
    .to_string()
)


# ============================================================
# 12. MODEL 1 — LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 70)

lr = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        max_iter=2000,
        random_state=42,
    ),
)

lr.fit(
    X_train,
    y_train,
)


# ============================================================
# 13. MODEL 2 — RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
)

rf.fit(
    X_train,
    y_train,
)


# ============================================================
# 14. MODEL 3 — BENIGN-ONLY ISOLATION FOREST
# ============================================================

print("\n" + "=" * 70)
print("TRAINING ISOLATION FOREST")
print("=" * 70)

benign_train = X_train[
    y_train == 0
]

iso = IsolationForest(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
)

iso.fit(
    benign_train
)

# Higher score = more anomalous
benign_scores = -iso.score_samples(
    benign_train
)

threshold = np.percentile(
    benign_scores,
    95,
)

print(
    f"Benign training samples: {len(benign_train)}"
)

print(
    f"Anomaly threshold: {threshold:.4f}"
)


# ============================================================
# 15. PREDICTIONS
# ============================================================

lr_pred = lr.predict(X_test)

lr_score = lr.predict_proba(
    X_test
)[:, 1]


rf_pred = rf.predict(X_test)

rf_score = rf.predict_proba(
    X_test
)[:, 1]


iso_score = -iso.score_samples(
    X_test
)

iso_pred = (
    iso_score > threshold
).astype(int)


# ============================================================
# 16. METRIC FUNCTION
# ============================================================

def evaluate(
    model_name,
    y_true,
    prediction,
    score,
):

    return {
        "Model": model_name,

        "Accuracy": accuracy_score(
            y_true,
            prediction,
        ),

        "Precision": precision_score(
            y_true,
            prediction,
            zero_division=0,
        ),

        "Recall": recall_score(
            y_true,
            prediction,
            zero_division=0,
        ),

        "F1": f1_score(
            y_true,
            prediction,
            zero_division=0,
        ),

        "ROC-AUC": roc_auc_score(
            y_true,
            score,
        ),
    }


# ============================================================
# 17. RESULTS
# ============================================================

results = pd.DataFrame(
    [
        evaluate(
            "Logistic Regression",
            y_test,
            lr_pred,
            lr_score,
        ),

        evaluate(
            "Random Forest",
            y_test,
            rf_pred,
            rf_score,
        ),

        evaluate(
            "Isolation Forest (Benign Only)",
            y_test,
            iso_pred,
            iso_score,
        ),
    ]
).round(4)


print("\n" + "=" * 70)
print("GROUP-AWARE RESULTS")
print("=" * 70)

print(
    results.to_string(
        index=False
    )
)


results.to_csv(
    RESULTS_DIR
    / "table_A_group_aware.csv",
    index=False,
)


# ============================================================
# 18. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("RANDOM FOREST CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        rf_pred,
        target_names=[
            "Benign",
            "Malware",
        ],
        zero_division=0,
    )
)


# ============================================================
# 19. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    rf_pred,
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Benign",
        "Malware",
    ],
)

display.plot()

plt.title(
    "Random Forest - Group-Aware Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR
    / "group_fig1_confusion.png",
    dpi=200,
)

plt.close()


# ============================================================
# 20. ROC CURVES
# ============================================================

plt.figure(
    figsize=(8, 6)
)

for name, score in [
    ("Logistic Regression", lr_score),
    ("Random Forest", rf_score),
    ("Isolation Forest", iso_score),
]:

    fpr, tpr, _ = roc_curve(
        y_test,
        score,
    )

    plt.plot(
        fpr,
        tpr,
        label=name,
    )


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Baseline",
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "Group-Aware ROC Curves"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    RESULTS_DIR
    / "group_fig2_roc.png",
    dpi=200,
)

plt.close()


# ============================================================
# 21. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

importance = pd.Series(
    rf.feature_importances_,
    index=features,
).sort_values(
    ascending=False
)

top = importance.head(15)

plt.figure(
    figsize=(8, 6)
)

top.sort_values().plot.barh()

plt.xlabel(
    "Importance"
)

plt.title(
    "Top 15 Random Forest Features"
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR
    / "group_fig3_feature_importance.png",
    dpi=200,
)

plt.close()


# ============================================================
# 22. ISOLATION FOREST DISTRIBUTION
# ============================================================

benign_scores_test = iso_score[
    y_test.values == 0
]

malware_scores_test = iso_score[
    y_test.values == 1
]

plt.figure(
    figsize=(8, 6)
)

plt.hist(
    benign_scores_test,
    bins=40,
    alpha=0.6,
    label="Benign",
)

plt.hist(
    malware_scores_test,
    bins=40,
    alpha=0.6,
    label="Malware",
)

plt.axvline(
    threshold,
    linestyle="--",
    label="Threshold",
)

plt.xlabel(
    "Anomaly Score"
)

plt.ylabel(
    "Number of Samples"
)

plt.title(
    "Isolation Forest - Group-Aware Scores"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    RESULTS_DIR
    / "group_fig4_isolation_distribution.png",
    dpi=200,
)

plt.close()


# ============================================================
# 23. SAVE MODEL
# ============================================================

benign_baseline = X_train[
    y_train == 0
]

median = benign_baseline.median()

mad = (
    (
        benign_baseline
        - median
    )
    .abs()
    .median()
    * 1.4826
    + 1e-6
)


artifact = {
    "random_forest": rf,
    "isolation_forest": iso,

    "isolation_threshold":
        float(threshold),

    "features": features,

    "benign_median":
        median,

    "benign_mad":
        mad,
}

joblib.dump(
    artifact,
    MODELS_DIR
    / "aerotrace_grouped.joblib",
)


# ============================================================
# 24. SAVE METADATA
# ============================================================

metadata = {
    "original_rows":
        int(before),

    "duplicates_removed":
        int(before - len(df)),

    "clean_rows":
        int(len(df)),

    "final_features":
        int(len(features)),

    "constant_features_removed":
        constant_features,

    "training_rows":
        int(len(X_train)),

    "testing_rows":
        int(len(X_test)),

    "training_groups":
        int(len(train_groups)),

    "testing_groups":
        int(len(test_groups)),

    "group_overlap":
        int(len(overlap)),

    "isolation_threshold":
        float(threshold),
}


with open(
    RESULTS_DIR
    / "group_meta.json",
    "w",
) as f:

    json.dump(
        metadata,
        f,
        indent=4,
    )


print("\n" + "=" * 70)
print("✅ GROUP-AWARE EVALUATION COMPLETE")
print("=" * 70)