# ============================================================
# AeroTrace
# Memory Forensics Triage using Machine Learning
# ============================================================

import json
from pathlib import Path

import joblib
import matplotlib

# Prevent matplotlib from trying to open a GUI on some systems
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
from sklearn.model_selection import train_test_split
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
# 2. LOAD DATASET
# ============================================================

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {CSV_PATH}\n"
        "Make sure the CSV is inside the data/ folder."
    )

df = pd.read_csv(CSV_PATH)

print("=" * 60)
print("AEROTRACE - DATASET LOADED")
print("=" * 60)

print(f"Original rows      : {len(df)}")
print(f"Original columns   : {len(df.columns)}")


# ============================================================
# 3. REMOVE EXACT DUPLICATES
# ============================================================

before = len(df)

df = df.drop_duplicates().reset_index(drop=True)

removed = before - len(df)

print(f"Duplicate rows removed: {removed}")
print(f"Rows after cleaning   : {len(df)}")


# ============================================================
# 4. CREATE BINARY TARGET
# ============================================================

# Benign = 0
# Malware = 1

df["y"] = (
    df["Class"]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("malware")
    .astype(int)
)


# ============================================================
# 5. EXTRACT MALWARE FAMILY
# ============================================================

# Examples:
#
# Ransomware-Maze-xxxx.raw -> Ransomware
# Spyware-xxxx.raw         -> Spyware
# Trojan-xxxx.raw          -> Trojan
# Benign                   -> Benign

df["family"] = (
    df["Category"]
    .astype(str)
    .str.strip()
    .str.split("-")
    .str[0]
)

print("\n" + "=" * 60)
print("MALWARE FAMILY DISTRIBUTION")
print("=" * 60)

print(df["family"].value_counts().to_string())


# ============================================================
# 6. SELECT NUMERIC FORENSIC FEATURES
# ============================================================

# We don't use:
# Category -> text identifier
# Class    -> target label
# y        -> our numeric target
# family   -> malware family information
#
# Everything else that is numeric becomes a feature.

excluded = {"Category", "Class", "y", "family"}

features = [
    column
    for column in df.columns
    if column not in excluded
    and pd.api.types.is_numeric_dtype(df[column])
]

X = df[features].copy()
y = df["y"].copy()

# Replace any unexpected missing numeric values with zero
X = X.fillna(0)

print("\n" + "=" * 60)
print("FEATURE INFORMATION")
print("=" * 60)

print(f"Number of features: {len(features)}")

print("\nFeatures:")

for i, feature in enumerate(features, start=1):
    print(f"{i:2}. {feature}")


# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

train_idx, test_idx = train_test_split(
    df.index,
    test_size=0.20,
    stratify=y,
    random_state=42,
)

X_train = X.loc[train_idx]
X_test = X.loc[test_idx]

y_train = y.loc[train_idx]
y_test = y.loc[test_idx]

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")

print("\nTraining class distribution:")
print(y_train.value_counts().rename({0: "Benign", 1: "Malware"}))

print("\nTesting class distribution:")
print(y_test.value_counts().rename({0: "Benign", 1: "Malware"}))


# ============================================================
# 8. MODEL 1 — LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 60)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 60)

logistic_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000, random_state=42)
)

logistic_model.fit(X_train, y_train)

print("Logistic Regression trained.")


# ============================================================
# 9. MODEL 2 — RANDOM FOREST
# ============================================================

print("\n" + "=" * 60)
print("TRAINING RANDOM FOREST")
print("=" * 60)

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
)

rf_model.fit(X_train, y_train)

print("Random Forest trained.")


# ============================================================
# 10. MODEL 3 — ISOLATION FOREST
# ============================================================

print("\n" + "=" * 60)
print("TRAINING ISOLATION FOREST")
print("=" * 60)

# IMPORTANT:
# Isolation Forest is trained ONLY using BENIGN samples.

benign_train = X_train[y_train == 0]

iso_model = IsolationForest(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
)

iso_model.fit(benign_train)

# Isolation Forest gives lower values to anomalies.
# We invert the score so:
#
# Higher score = more anomalous

benign_train_scores = -iso_model.score_samples(benign_train)

# Choose the 95th percentile as threshold.
# Roughly 5% of benign training data will be flagged.

iso_threshold = np.percentile(benign_train_scores, 95)

print(f"Benign training samples: {len(benign_train)}")
print(f"Anomaly threshold      : {iso_threshold:.4f}")


# ============================================================
# 11. EVALUATION FUNCTION
# ============================================================

def evaluate_model(name, y_true, predictions, scores):
    result = {
        "Model": name,
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "F1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "ROC-AUC": roc_auc_score(
            y_true,
            scores,
        ),
    }

    return result


# ============================================================
# 12. PREDICTIONS
# ============================================================

# Logistic Regression
lr_predictions = logistic_model.predict(X_test)
lr_scores = logistic_model.predict_proba(X_test)[:, 1]


# Random Forest
rf_predictions = rf_model.predict(X_test)
rf_scores = rf_model.predict_proba(X_test)[:, 1]


# Isolation Forest
iso_scores = -iso_model.score_samples(X_test)

iso_predictions = (
    iso_scores > iso_threshold
).astype(int)


# ============================================================
# 13. MODEL METRICS
# ============================================================

results = [
    evaluate_model(
        "Logistic Regression",
        y_test,
        lr_predictions,
        lr_scores,
    ),
    evaluate_model(
        "Random Forest",
        y_test,
        rf_predictions,
        rf_scores,
    ),
    evaluate_model(
        "Isolation Forest (Benign Only)",
        y_test,
        iso_predictions,
        iso_scores,
    ),
]

results_df = pd.DataFrame(results)

results_df = results_df.round(4)

print("\n" + "=" * 60)
print("EXPERIMENT A — RANDOM 80/20 SPLIT")
print("=" * 60)

print(results_df.to_string(index=False))


# Save table
results_df.to_csv(
    RESULTS_DIR / "table_A_random_split.csv",
    index=False,
)


# ============================================================
# 14. RANDOM FOREST CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("RANDOM FOREST CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        rf_predictions,
        target_names=["Benign", "Malware"],
        zero_division=0,
    )
)


# ============================================================
# 15. RANDOM FOREST CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    rf_predictions,
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Benign", "Malware"],
)

disp.plot()

plt.title("Random Forest Confusion Matrix")
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "fig1_confusion.png",
    dpi=200,
)

plt.close()


# ============================================================
# 16. ROC CURVES
# ============================================================

plt.figure(figsize=(8, 6))

# Logistic Regression
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_scores)
plt.plot(
    fpr_lr,
    tpr_lr,
    label="Logistic Regression",
)

# Random Forest
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_scores)
plt.plot(
    fpr_rf,
    tpr_rf,
    label="Random Forest",
)

# Isolation Forest
fpr_iso, tpr_iso, _ = roc_curve(y_test, iso_scores)
plt.plot(
    fpr_iso,
    tpr_iso,
    label="Isolation Forest",
)

# Random baseline
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Baseline",
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "fig2_roc.png",
    dpi=200,
)

plt.close()


# ============================================================
# 17. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

importance = pd.Series(
    rf_model.feature_importances_,
    index=features,
)

importance = importance.sort_values(
    ascending=False
)

top_features = importance.head(15)

plt.figure(figsize=(8, 6))

top_features.sort_values().plot.barh()

plt.xlabel("Importance")
plt.title("Top 15 Random Forest Features")

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "fig3_feature_importance.png",
    dpi=200,
)

plt.close()


# ============================================================
# 18. ISOLATION FOREST SCORE DISTRIBUTION
# ============================================================

benign_test_scores = iso_scores[y_test.values == 0]
malware_test_scores = iso_scores[y_test.values == 1]

plt.figure(figsize=(8, 6))

plt.hist(
    benign_test_scores,
    bins=40,
    alpha=0.6,
    label="Benign",
)

plt.hist(
    malware_test_scores,
    bins=40,
    alpha=0.6,
    label="Malware",
)

plt.axvline(
    iso_threshold,
    linestyle="--",
    label="Threshold",
)

plt.xlabel("Anomaly Score")
plt.ylabel("Number of Samples")
plt.title("Isolation Forest Anomaly Score Distribution")
plt.legend()

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "fig4_isolation_distribution.png",
    dpi=200,
)

plt.close()


# ============================================================
# 19. EXPERIMENT B — UNSEEN MALWARE FAMILY
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENT B — UNSEEN MALWARE FAMILY")
print("=" * 60)

families_to_test = [
    family
    for family in ["Trojan", "Spyware", "Ransomware"]
    if family in set(df["family"])
]

experiment_b_results = []


for held_out_family in families_to_test:

    print(f"\n----- Holding out: {held_out_family} -----")

    # Training:
    # benign + all malware except held-out family
    training_mask = df.loc[train_idx, "family"] != held_out_family

    held_X_train = X.loc[train_idx][training_mask]
    held_y_train = y.loc[train_idx][training_mask]

    # Testing:
    # all held-out malware + held-out benign test samples
    held_malware_idx = df.index[
        df["family"] == held_out_family
    ]

    held_benign_idx = test_idx[
        df.loc[test_idx, "y"] == 0
    ]

    held_test_idx = list(
        held_benign_idx
    ) + list(
        held_malware_idx
    )

    held_X_test = X.loc[held_test_idx]
    held_y_test = y.loc[held_test_idx]

    # -----------------------------
    # Logistic Regression
    # -----------------------------

    lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            random_state=42,
        ),
    )

    lr.fit(
        held_X_train,
        held_y_train,
    )

    lr_pred = lr.predict(held_X_test)
    lr_score = lr.predict_proba(
        held_X_test
    )[:, 1]

    result = evaluate_model(
        "Logistic Regression",
        held_y_test,
        lr_pred,
        lr_score,
    )

    result["Held-out Family"] = held_out_family
    experiment_b_results.append(result)

    # -----------------------------
    # Random Forest
    # -----------------------------

    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    rf.fit(
        held_X_train,
        held_y_train,
    )

    rf_pred = rf.predict(held_X_test)
    rf_score = rf.predict_proba(
        held_X_test
    )[:, 1]

    result = evaluate_model(
        "Random Forest",
        held_y_test,
        rf_pred,
        rf_score,
    )

    result["Held-out Family"] = held_out_family
    experiment_b_results.append(result)

    # -----------------------------
    # Isolation Forest
    # -----------------------------

    benign_only = held_X_train[
        held_y_train == 0
    ]

    iso = IsolationForest(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    iso.fit(benign_only)

    anomaly_scores = -iso.score_samples(
        held_X_test
    )

    benign_scores = -iso.score_samples(
        benign_only
    )

    threshold = np.percentile(
        benign_scores,
        95,
    )

    iso_pred = (
        anomaly_scores > threshold
    ).astype(int)

    result = evaluate_model(
        "Isolation Forest (Benign Only)",
        held_y_test,
        iso_pred,
        anomaly_scores,
    )

    result["Held-out Family"] = held_out_family
    experiment_b_results.append(result)


# Convert results to dataframe
results_B_df = pd.DataFrame(
    experiment_b_results
).round(4)


print("\n")
print(results_B_df.to_string(index=False))


results_B_df.to_csv(
    RESULTS_DIR / "table_B_unseen_family.csv",
    index=False,
)


# ============================================================
# 20. UNSEEN FAMILY RECALL GRAPH
# ============================================================

recall_table = results_B_df.pivot(
    index="Held-out Family",
    columns="Model",
    values="Recall",
)

recall_table.plot(
    kind="bar",
    figsize=(9, 6),
)

plt.ylabel("Recall")
plt.title("Detection of Unseen Malware Families")
plt.ylim(0, 1.05)
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "fig5_unseen_family_recall.png",
    dpi=200,
)

plt.close()


# ============================================================
# 21. SAVE TRAINED MODELS
# ============================================================

# Robust benign baseline for the future Streamlit explanation
benign_baseline = X_train[
    y_train == 0
]

median = benign_baseline.median()

mad = (
    benign_baseline - median
).abs().median() * 1.4826 + 1e-6


artifact = {
    "random_forest": rf_model,
    "isolation_forest": iso_model,
    "isolation_threshold": float(iso_threshold),
    "features": features,
    "benign_median": median,
    "benign_mad": mad,
}

joblib.dump(
    artifact,
    MODELS_DIR / "aerotrace.joblib",
)


# ============================================================
# 22. SAVE DEMO DATA
# ============================================================

demo = X_test.copy()

demo["true_label"] = (
    y_test
    .map({
        0: "Benign",
        1: "Malware",
    })
)

demo["family"] = df.loc[
    test_idx,
    "family",
]

demo_sample = demo.sample(
    min(300, len(demo)),
    random_state=42,
)

demo_sample.to_csv(
    RESULTS_DIR / "demo_samples.csv",
    index=False,
)


# ============================================================
# 23. SAVE METADATA
# ============================================================

metadata = {
    "original_rows": int(before),
    "duplicates_removed": int(removed),
    "clean_rows": int(len(df)),
    "number_of_features": int(len(features)),
    "training_samples": int(len(X_train)),
    "testing_samples": int(len(X_test)),
    "isolation_threshold": float(iso_threshold),
    "families": families_to_test,
}

with open(
    RESULTS_DIR / "meta.json",
    "w",
) as file:
    json.dump(
        metadata,
        file,
        indent=4,
    )


# ============================================================
# 24. FINISHED
# ============================================================

print("\n" + "=" * 60)
print("✅ AEROTRACE TRAINING COMPLETE")
print("=" * 60)

print("\nGenerated files:")

for file in sorted(RESULTS_DIR.iterdir()):
    print(f"  results/{file.name}")

print("\nSaved model:")

print("  models/aerotrace.joblib")

print("\nYou can now build the Streamlit dashboard.")