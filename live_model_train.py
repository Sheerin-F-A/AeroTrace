# ============================================================
# AeroTrace - Live Deployment Model
#
# Trains models using only CIC-MalMem-2022 features that
# AeroTrace can reconstruct from our Volatility 3 pipeline.
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

from sklearn.ensemble import (
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS
# ============================================================

CSV_PATH = Path(
    "data/Obfuscated-MalMem2022.csv"
)

MODEL_PATH = Path(
    "models/aerotrace_live.joblib"
)

RESULTS_DIR = Path("results")

RESULTS_DIR.mkdir(
    exist_ok=True
)

MODEL_PATH.parent.mkdir(
    exist_ok=True
)


# ============================================================
# FEATURES THAT OUR LIVE VOLATILITY PIPELINE CAN REBUILD
# ============================================================

CANDIDATE_FEATURES = [

    # pslist
    "pslist.nproc",
    "pslist.avg_threads",
    "pslist.avg_handlers",

    # dlllist
    "dlllist.ndlls",
    "dlllist.avg_dlls_per_proc",

    # handles
    "handles.nhandles",
    "handles.nfile",
    "handles.nevent",
    "handles.ndesktop",
    "handles.nkey",
    "handles.nthread",
    "handles.ndirectory",
    "handles.nsemaphore",
    "handles.ntimer",
    "handles.nsection",
    "handles.nmutant",

    # ldrmodules
    "ldrmodules.not_in_load",
    "ldrmodules.not_in_init",
    "ldrmodules.not_in_mem",
    "ldrmodules.not_in_load_avg",
    "ldrmodules.not_in_init_avg",
    "ldrmodules.not_in_mem_avg",

    # malfind
    "malfind.ninjections",
    "malfind.commitCharge",
]


# ============================================================
# LOAD
# ============================================================

if not CSV_PATH.exists():

    raise FileNotFoundError(
        f"Dataset not found: {CSV_PATH}"
    )

df = pd.read_csv(
    CSV_PATH
)

print("=" * 70)
print("AEROTRACE - LIVE DEPLOYMENT MODEL")
print("=" * 70)

print(
    f"Original rows: {len(df)}"
)


# ============================================================
# REMOVE EXACT DUPLICATES
# ============================================================

before = len(df)

df = (
    df
    .drop_duplicates()
    .reset_index(drop=True)
)

print(
    f"Duplicates removed: "
    f"{before - len(df)}"
)

print(
    f"Clean rows: {len(df)}"
)


# ============================================================
# TARGET
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
# CREATE GROUP IDENTIFIER
# ============================================================

def make_group_id(
    category,
    row_index,
):

    category = str(
        category
    ).strip()

    # Benign records don't expose individual
    # source filenames in this CSV.
    if category == "Benign":

        return (
            f"BENIGN_{row_index}"
        )

    # Malware names:
    #
    # Ransomware-Ako-XXXXX-1.raw
    # Ransomware-Ako-XXXXX-2.raw
    #
    # become the same group.

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
    for index, category
    in enumerate(df["Category"])
]


# ============================================================
# CHECK REQUIRED FEATURES
# ============================================================

missing = [
    feature
    for feature in CANDIDATE_FEATURES
    if feature not in df.columns
]

if missing:

    raise RuntimeError(
        "Missing expected CIC features:\n"
        + "\n".join(missing)
    )


# ============================================================
# FEATURE MATRIX
# ============================================================

X = (
    df[CANDIDATE_FEATURES]
    .apply(
        pd.to_numeric,
        errors="coerce",
    )
    .fillna(0)
)


# ============================================================
# REMOVE CONSTANT FEATURES
# ============================================================

constant_features = [
    feature
    for feature in X.columns
    if X[feature].nunique() <= 1
]

if constant_features:

    print("\nConstant features removed:")

    for feature in constant_features:

        print(
            f"  - {feature}"
        )

    X = X.drop(
        columns=constant_features
    )


features = list(
    X.columns
)


print(
    f"\nFinal live-model features: "
    f"{len(features)}"
)

for feature in features:

    print(
        f"  - {feature}"
    )


# ============================================================
# GROUP-AWARE SPLIT
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
        groups=df["group_id"],
    )
)

X_train = X.iloc[
    train_positions
]

X_test = X.iloc[
    test_positions
]

y_train = df["y"].iloc[
    train_positions
]

y_test = df["y"].iloc[
    test_positions
]

train_groups = set(
    df["group_id"].iloc[
        train_positions
    ]
)

test_groups = set(
    df["group_id"].iloc[
        test_positions
    ]
)

overlap = (
    train_groups
    .intersection(test_groups)
)


if overlap:

    raise RuntimeError(
        "Train/test groups overlap!"
    )


print("\n" + "=" * 70)
print("GROUP-AWARE SPLIT")
print("=" * 70)

print(
    f"Training rows: {len(X_train)}"
)

print(
    f"Testing rows : {len(X_test)}"
)

print(
    f"Training groups: "
    f"{len(train_groups)}"
)

print(
    f"Testing groups: "
    f"{len(test_groups)}"
)

print(
    f"Group overlap: {len(overlap)}"
)


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

print(
    "\nTraining Logistic Regression..."
)

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

lr_pred = lr.predict(
    X_test
)

lr_score = (
    lr.predict_proba(X_test)[:, 1]
)


# ============================================================
# RANDOM FOREST
# ============================================================

print(
    "Training Random Forest..."
)

rf = RandomForestClassifier(
    n_estimators=250,
    random_state=42,
    n_jobs=-1,
)

rf.fit(
    X_train,
    y_train,
)

rf_pred = rf.predict(
    X_test
)

rf_score = (
    rf.predict_proba(X_test)[:, 1]
)


# ============================================================
# ISOLATION FOREST
# ============================================================

print(
    "Training benign-only Isolation Forest..."
)

benign_train = X_train[
    y_train == 0
]

iso = IsolationForest(
    n_estimators=250,
    random_state=42,
    n_jobs=-1,
)

iso.fit(
    benign_train
)

# Our anomaly score:
# higher = more anomalous

iso_train_score = (
    -iso.score_samples(
        benign_train
    )
)

iso_test_score = (
    -iso.score_samples(
        X_test
    )
)

# Threshold chosen so approximately 5% of benign
# training samples are flagged.

iso_threshold = np.percentile(
    iso_train_score,
    95,
)

iso_pred = (
    iso_test_score
    > iso_threshold
).astype(int)


# ============================================================
# METRICS
# ============================================================

def evaluate(
    name,
    truth,
    predictions,
    scores,
):

    return {

        "Model":
            name,

        "Accuracy":
            accuracy_score(
                truth,
                predictions,
            ),

        "Precision":
            precision_score(
                truth,
                predictions,
                zero_division=0,
            ),

        "Recall":
            recall_score(
                truth,
                predictions,
                zero_division=0,
            ),

        "F1":
            f1_score(
                truth,
                predictions,
                zero_division=0,
            ),

        "ROC-AUC":
            roc_auc_score(
                truth,
                scores,
            ),
    }


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
            iso_test_score,
        ),
    ]

).round(4)


print("\n" + "=" * 70)
print("LIVE MODEL GROUP-AWARE RESULTS")
print("=" * 70)

print(
    results.to_string(
        index=False
    )
)


results.to_csv(
    RESULTS_DIR
    / "table_live_model.csv",
    index=False,
)


# ============================================================
# RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

importance = pd.Series(
    rf.feature_importances_,
    index=features,
).sort_values(
    ascending=False
)

importance.to_csv(
    RESULTS_DIR
    / "live_feature_importance.csv"
)


plt.figure(
    figsize=(9, 7)
)

importance.head(15).sort_values().plot.barh()

plt.xlabel(
    "Random Forest Feature Importance"
)

plt.title(
    "AeroTrace Live Model - Top Features"
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR
    / "live_feature_importance.png",
    dpi=200,
)

plt.close()


# ============================================================
# SAVE BENIGN BASELINE FOR EXPLANATIONS
# ============================================================

benign_baseline = X_train[
    y_train == 0
]

benign_median = (
    benign_baseline
    .median()
)

benign_mad = (
    (
        benign_baseline
        - benign_median
    )
    .abs()
    .median()
    * 1.4826
    + 1e-6
)


# ============================================================
# SAVE EVERYTHING REQUIRED BY THE APP
# ============================================================

artifact = {

    "random_forest":
        rf,

    "isolation_forest":
        iso,

    "isolation_threshold":
        float(
            iso_threshold
        ),

    "features":
        features,

    "benign_median":
        benign_median,

    "benign_mad":
        benign_mad,

    "candidate_features":
        CANDIDATE_FEATURES,
}


joblib.dump(
    artifact,
    MODEL_PATH,
)


# ============================================================
# METADATA
# ============================================================

metadata = {

    "dataset_rows":
        int(before),

    "duplicates_removed":
        int(
            before - len(df)
        ),

    "clean_rows":
        int(len(df)),

    "features":
        features,

    "features_removed":
        constant_features,

    "training_rows":
        int(len(X_train)),

    "testing_rows":
        int(len(X_test)),

    "group_overlap":
        int(len(overlap)),

    "isolation_threshold":
        float(
            iso_threshold
        ),
}


with open(
    RESULTS_DIR
    / "live_model_meta.json",
    "w",
) as file:

    json.dump(
        metadata,
        file,
        indent=4,
    )


print("\n" + "=" * 70)
print("✅ LIVE DEPLOYMENT MODEL READY")
print("=" * 70)

print(
    f"Saved: {MODEL_PATH}"
)