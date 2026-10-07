# ============================================================
# AeroTrace
# Automated Memory Forensics Triage
# ============================================================

import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from extract_features import (
    extract_memory_features,
)


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="AeroTrace",
    page_icon="🛡️",
    layout="wide",
)


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = Path(
    "models/aerotrace_live.joblib"
)

RESEARCH_RESULTS = Path(
    "results"
)

DEMO_DUMP = Path(
    "memory_dumps/challenge.raw"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return joblib.load(
        MODEL_PATH
    )


artifact = load_model()

rf = artifact[
    "random_forest"
]

iso = artifact[
    "isolation_forest"
]

iso_threshold = artifact[
    "isolation_threshold"
]

features = artifact[
    "features"
]

benign_median = artifact[
    "benign_median"
]

benign_mad = artifact[
    "benign_mad"
]


# ============================================================
# HEADER
# ============================================================

st.title(
    "🛡️ AeroTrace"
)

st.subheader(
    "Automated Memory Forensics Triage"
)

st.caption(
    "Volatility 3 → forensic feature engineering "
    "→ Random Forest + Isolation Forest → triage"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Investigation"
)

mode = st.sidebar.radio(
    "Mode",
    [
        "Live Memory Investigation",
        "Research Benchmark",
    ],
)


# ============================================================
# LIVE INVESTIGATION
# ============================================================

if mode == "Live Memory Investigation":

    st.sidebar.markdown(
        "### Evidence Input"
    )

    demo_available = (
        DEMO_DUMP.exists()
    )

    use_demo = False

    if demo_available:

        use_demo = st.sidebar.checkbox(
            "Use bundled demo memory image",
            value=False,
        )


    uploaded_file = None

    if not use_demo:

        uploaded_file = (
            st.sidebar.file_uploader(
                "Upload Windows memory image",
                type=[
                    "raw",
                    "vmem",
                    "dmp",
                    "bin",
                ],
                help=(
                    "AeroTrace expects an existing "
                    "Windows memory image. "
                    "It analyzes the image; it "
                    "does not acquire RAM itself."
                ),
            )
        )


    # --------------------------------------------------------
    # Determine input
    # --------------------------------------------------------

    memory_bytes = None
    memory_path = None
    display_name = None


    if use_demo:

        memory_path = (
            DEMO_DUMP
        )

        memory_bytes = (
            DEMO_DUMP.read_bytes()
        )

        display_name = (
            DEMO_DUMP.name
        )


    elif uploaded_file is not None:

        memory_bytes = (
            uploaded_file.getvalue()
        )

        display_name = (
            uploaded_file.name
        )


    # --------------------------------------------------------
    # Ready state
    # --------------------------------------------------------

    if memory_bytes is None:

        st.info(
            "Upload a Windows memory image "
            "or select the bundled demo sample "
            "from the sidebar."
        )

        st.markdown(
            """
            ### Investigation workflow

            **1. Evidence ingestion**  
            Memory image is hashed for integrity identification.

            **2. Volatility 3 extraction**  
            AeroTrace runs process, DLL, handle,
            module-loader and memory-injection analysis.

            **3. Feature engineering**  
            Volatility outputs are converted into a
            structured forensic feature vector.

            **4. ML triage**  
            Random Forest performs supervised classification
            while Isolation Forest evaluates deviation
            from a benign baseline.

            **5. Investigation report**  
            Results, evidence hash and feature explanations
            can be exported.
            """
        )

    else:

        st.success(
            f"Evidence ready: **{display_name}**"
        )

        file_size_mb = (
            len(memory_bytes)
            / 1024
            / 1024
        )

        sha256 = hashlib.sha256(
            memory_bytes
        ).hexdigest()


        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Evidence Size",
            f"{file_size_mb:.2f} MB",
        )

        c2.metric(
            "SHA-256",
            f"{sha256[:16]}...",
        )

        c3.metric(
            "Model Features",
            str(len(features)),
        )


        st.caption(
            f"SHA-256: `{sha256}`"
        )


        # ----------------------------------------------------
        # Analyze button
        # ----------------------------------------------------

        if st.button(
            "🔎 Run Forensic Investigation",
            type="primary",
            use_container_width=True,
        ):

            # -----------------------------------------------
            # Save uploaded bytes temporarily
            # -----------------------------------------------

            if memory_path is None:

                suffix = (
                    Path(
                        display_name
                    ).suffix
                    or ".raw"
                )

                temp_file = (
                    tempfile.NamedTemporaryFile(
                        suffix=suffix,
                        delete=False,
                    )
                )

                temp_file.write(
                    memory_bytes
                )

                temp_file.close()

                memory_path = Path(
                    temp_file.name
                )


            case_timestamp = (
                datetime.now(
                    timezone.utc
                )
                .strftime(
                    "%Y%m%d_%H%M%S"
                )
            )


            case_dir = (
                Path(
                    "live_output"
                )
                / "cases"
                / case_timestamp
            )


            case_dir.mkdir(
                parents=True,
                exist_ok=True,
            )


            progress = st.progress(
                0
            )

            status = st.empty()


            def update_progress(
                value,
                message,
            ):

                progress.progress(
                    min(
                        max(
                            int(
                                value * 100
                            ),
                            0,
                        ),
                        100,
                    )
                )

                status.info(
                    message
                )


            try:

                # =========================================
                # VOLATILITY EXTRACTION
                # =========================================

                extraction = (
                    extract_memory_features(
                        memory_path,
                        output_dir=case_dir,
                        progress_callback=update_progress,
                    )
                )


                progress.progress(
                    100
                )

                status.success(
                    "Forensic extraction complete."
                )


                X = (
                    extraction[
                        "features"
                    ]
                    .reindex(
                        columns=features
                    )
                    .fillna(0)
                )


                # =========================================
                # RANDOM FOREST
                # =========================================

                rf_probability = float(
                    rf.predict_proba(
                        X
                    )[0, 1]
                )

                rf_flag = (
                    rf_probability >= 0.5
                )


                # =========================================
                # ISOLATION FOREST
                # =========================================

                anomaly_score = float(
                    -iso.score_samples(
                        X
                    )[0]
                )

                iso_flag = (
                    anomaly_score
                    > iso_threshold
                )


                # =========================================
                # FUSION
                # =========================================

                if (
                    rf_flag
                    and iso_flag
                ):

                    verdict = (
                        "MALICIOUS"
                    )

                elif (
                    rf_flag
                    or iso_flag
                ):

                    verdict = (
                        "SUSPICIOUS"
                    )

                else:

                    verdict = (
                        "BENIGN"
                    )


                # =========================================
                # WHY?
                # =========================================

                z_scores = (
                    (
                        X.iloc[0]
                        - benign_median
                        .reindex(
                            features
                        )
                    )
                    /
                    benign_mad
                    .reindex(
                        features
                    )
                )


                top = (
                    z_scores
                    .abs()
                    .sort_values(
                        ascending=False
                    )
                    .head(10)
                )


                explanation = pd.DataFrame(
                    {

                        "Feature":
                            top.index,

                        "Sample Value":
                            X.iloc[0]
                            [top.index]
                            .values,

                        "Benign Median":
                            benign_median
                            [top.index]
                            .values,

                        "Robust Z-Score":
                            z_scores
                            [top.index]
                            .round(2)
                            .values,
                    }
                )


                # =========================================
                # CASE SUMMARY
                # =========================================

                summary = (
                    extraction[
                        "summary"
                    ]
                )


                # =========================================
                # DASHBOARD
                # =========================================

                st.markdown(
                    "---"
                )

                st.header(
                    "Forensic Triage Result"
                )


                c1, c2, c3, c4 = (
                    st.columns(4)
                )

                c1.metric(
                    "Processes",
                    summary[
                        "processes"
                    ],
                )

                c2.metric(
                    "DLL Records",
                    summary[
                        "dll_records"
                    ],
                )

                c3.metric(
                    "Handle Records",
                    summary[
                        "handle_records"
                    ],
                )

                c4.metric(
                    "Malfind Findings",
                    summary[
                        "malfind_findings"
                    ],
                )


                st.markdown(
                    "### Automated Verdict"
                )


                if verdict == "MALICIOUS":

                    st.error(
                        "🚨 MALICIOUS"
                    )

                elif verdict == "SUSPICIOUS":

                    st.warning(
                        "⚠️ SUSPICIOUS"
                    )

                else:

                    st.success(
                        "✅ BENIGN"
                    )


                st.write(
                    "This is an automated triage result "
                    "and should be followed by analyst review."
                )


                c1, c2, c3 = (
                    st.columns(3)
                )

                c1.metric(
                    "RF P(Malware)",
                    f"{rf_probability:.3f}",
                )

                c2.metric(
                    "Isolation Score",
                    f"{anomaly_score:.3f}",
                )

                c3.metric(
                    "IF Threshold",
                    f"{iso_threshold:.3f}",
                )


                # =========================================
                # MODEL FLAGS
                # =========================================

                st.markdown(
                    "### Model Decisions"
                )


                decision_df = pd.DataFrame(
                    {
                        "Component":
                            [
                                "Random Forest",
                                "Isolation Forest",
                            ],

                        "Result":
                            [
                                (
                                    "FLAG"
                                    if rf_flag
                                    else "OK"
                                ),

                                (
                                    "ANOMALY"
                                    if iso_flag
                                    else "NORMAL"
                                ),
                            ],
                    }
                )


                st.dataframe(
                    decision_df,
                    hide_index=True,
                    use_container_width=True,
                )


                # =========================================
                # EVIDENCE INTEGRITY
                # =========================================

                with st.expander(
                    "📋 Evidence Integrity",
                    expanded=True,
                ):

                    st.write(
                        f"**File:** {display_name}"
                    )

                    st.code(
                        sha256,
                        language="text",
                    )

                    st.caption(
                        "SHA-256 identifies the exact evidence "
                        "file supplied to AeroTrace."
                    )


                # =========================================
                # WHY?
                # =========================================

                st.markdown(
                    "### 🧠 Feature Deviations"
                )

                st.caption(
                    "Large robust z-scores indicate features "
                    "that differ strongly from the benign "
                    "training baseline."
                )


                st.dataframe(
                    explanation,
                    hide_index=True,
                    use_container_width=True,
                )


                st.bar_chart(
                    explanation.set_index(
                        "Feature"
                    )[
                        "Robust Z-Score"
                    ]
                )


                # =========================================
                # RAW FORENSIC ARTIFACTS
                # =========================================

                st.markdown(
                    "### 🔬 Extracted Forensic Artifacts"
                )


                pslist_df = pd.read_csv(
                    case_dir
                    / "pslist.csv"
                )

                malfind_df = pd.read_csv(
                    case_dir
                    / "malfind.csv"
                )

                handles_df = pd.read_csv(
                    case_dir
                    / "handles.csv"
                )


                with st.expander(
                    "Process List (Volatility pslist)"
                ):

                    st.dataframe(
                        pslist_df,
                        use_container_width=True,
                        hide_index=True,
                    )


                with st.expander(
                    "Potentially Suspicious Memory Regions "
                    "(Volatility malfind)"
                ):

                    if len(malfind_df) == 0:

                        st.info(
                            "No malfind findings were returned."
                        )

                    else:

                        display_columns = [
                            column
                            for column in [
                                "PID",
                                "Process",
                                "Start VPN",
                                "End VPN",
                                "Tag",
                                "Protection",
                                "CommitCharge",
                                "PrivateMemory",
                            ]
                            if column
                            in malfind_df.columns
                        ]

                        st.dataframe(
                            malfind_df[
                                display_columns
                            ],
                            use_container_width=True,
                            hide_index=True,
                        )


                with st.expander(
                    "Handle Type Distribution"
                ):

                    handle_counts = (
                        handles_df[
                            "Type"
                        ]
                        .value_counts()
                        .head(15)
                    )

                    st.bar_chart(
                        handle_counts
                    )


                with st.expander(
                    "Windows Image Information"
                ):

                    st.code(
                        extraction[
                            "info_text"
                        ],
                        language="text",
                    )


                # =========================================
                # FEATURE VECTOR
                # =========================================

                with st.expander(
                    "📊 Engineered Feature Vector"
                ):

                    st.dataframe(
                        X.T.rename(
                            columns={
                                X.index[0]:
                                    "Value"
                            }
                        ),
                        use_container_width=True,
                    )


                # =========================================
                # REPORT
                # =========================================

                report = {

                    "tool":
                        "AeroTrace",

                    "case_timestamp_utc":
                        datetime.now(
                            timezone.utc
                        ).isoformat(),

                    "evidence_file":
                        display_name,

                    "sha256":
                        sha256,

                    "verdict":
                        verdict,

                    "random_forest_probability":
                        rf_probability,

                    "random_forest_flag":
                        bool(rf_flag),

                    "isolation_forest_score":
                        anomaly_score,

                    "isolation_forest_threshold":
                        float(
                            iso_threshold
                        ),

                    "isolation_forest_flag":
                        bool(iso_flag),

                    "forensic_summary":
                        summary,

                    "top_feature_deviations":
                        explanation
                        .to_dict(
                            orient="records"
                        ),
                }


                st.markdown(
                    "### 📄 Investigation Report"
                )


                report_json = json.dumps(
                    report,
                    indent=2,
                    default=float,
                )


                st.download_button(
                    "⬇️ Download AeroTrace Report",
                    data=report_json,
                    file_name=(
                        "AeroTrace_Investigation_Report.json"
                    ),
                    mime="application/json",
                    use_container_width=True,
                )


            except Exception as error:

                progress.empty()
                status.empty()

                st.error(
                    "Investigation failed."
                )

                st.exception(
                    error
                )


# ============================================================
# RESEARCH BENCHMARK PAGE
# ============================================================

else:

    st.header(
        "📈 CIC-MalMem-2022 Benchmark"
    )

    st.caption(
        "Research evaluation of the AeroTrace machine-learning "
        "component using group-aware train/test separation."
    )


    table_path = (
        RESEARCH_RESULTS
        / "table_A_group_aware.csv"
    )

    if table_path.exists():

        research_table = (
            pd.read_csv(
                table_path
            )
        )

        st.subheader(
            "Group-Aware Model Performance"
        )

        st.dataframe(
            research_table,
            use_container_width=True,
            hide_index=True,
        )


    table_b_path = (
        RESEARCH_RESULTS
        / "table_B_unseen_family.csv"
    )

    if table_b_path.exists():

        st.subheader(
            "Unseen Malware Family Experiment"
        )

        st.dataframe(
            pd.read_csv(
                table_b_path
            ),
            use_container_width=True,
            hide_index=True,
        )


    st.subheader(
        "Research Visualizations"
    )


    images = [

        (
            "group_fig1_confusion.png",
            "Random Forest Confusion Matrix",
        ),

        (
            "group_fig2_roc.png",
            "Group-Aware ROC Curves",
        ),

        (
            "group_fig3_feature_importance.png",
            "Top Random Forest Features",
        ),

        (
            "group_fig4_isolation_distribution.png",
            "Isolation Forest Score Distribution",
        ),

        (
            "group_fig5_unseen_family_recall.png",
            "Unseen Malware Family Recall",
        ),

    ]


    for filename, caption in images:

        image_path = (
            RESEARCH_RESULTS
            / filename
        )

        if image_path.exists():

            st.image(
                str(image_path),
                caption=caption,
            )


    st.subheader(
        "System Architecture"
    )

    st.markdown(
        """
        **Memory Image**

        ↓

        **Volatility 3**

        ↓

        **Forensic Artifact Extraction**

        (`pslist`, `dlllist`, `handles`,
        `ldrmodules`, `malfind`)

        ↓

        **Feature Engineering**

        ↓

        **Random Forest + Isolation Forest**

        ↓

        **Automated Triage**

        ↓

        **Explanation + Evidence Integrity + Report**
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    "---"
)

st.caption(
    "AeroTrace is a research prototype for automated "
    "memory-forensics triage. It analyzes existing Windows "
    "memory images using Volatility 3 and machine learning; "
    "it does not perform RAM acquisition."
)