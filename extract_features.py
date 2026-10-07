# ============================================================
# AeroTrace - Automatic Volatility 3 Feature Extraction
# ============================================================

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

VOLATILITY = Path(
    "tools/volatility3/vol.py"
)

DEFAULT_OUTPUT = Path(
    "live_output"
)


# ============================================================
# FEATURES
# ============================================================

FEATURE_NAMES = [

    "pslist.nproc",
    "pslist.avg_threads",
    "pslist.avg_handlers",

    "dlllist.ndlls",
    "dlllist.avg_dlls_per_proc",

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

    "ldrmodules.not_in_load",
    "ldrmodules.not_in_init",
    "ldrmodules.not_in_mem",
    "ldrmodules.not_in_load_avg",
    "ldrmodules.not_in_init_avg",
    "ldrmodules.not_in_mem_avg",

    "malfind.ninjections",
    "malfind.commitCharge",
]


# ============================================================
# PLUGIN CONFIGURATION
# ============================================================

CSV_PLUGINS = {

    "pslist":
        "windows.pslist",

    "dlllist":
        "windows.dlllist",

    "handles":
        "windows.handles",

    "ldrmodules":
        "windows.malware.ldrmodules",

    "malfind":
        "windows.malware.malfind",
}


# ============================================================
# RUN VOLATILITY
# ============================================================

def run_volatility_csv(
    memory_path,
    plugin,
    output_path,
):

    command = [

        sys.executable,

        str(VOLATILITY),

        "-f",
        str(memory_path),

        "-q",

        "-r",
        "csv",

        plugin,
    ]

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as output_file:

        process = subprocess.run(

            command,

            stdout=output_file,

            stderr=subprocess.PIPE,

            text=True,

            timeout=600,
        )

    if process.returncode != 0:

        raise RuntimeError(

            f"Volatility failed for {plugin}\n\n"
            + process.stderr[
                -4000:
            ]
        )


def run_volatility_info(
    memory_path,
    output_path,
):

    command = [

        sys.executable,

        str(VOLATILITY),

        "-f",
        str(memory_path),

        "-q",

        "-r",
        "pretty",

        "windows.info",
    ]

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as output_file:

        process = subprocess.run(

            command,

            stdout=output_file,

            stderr=subprocess.PIPE,

            text=True,

            timeout=600,
        )

    if process.returncode != 0:

        raise RuntimeError(

            "Volatility windows.info failed\n\n"
            + process.stderr[
                -4000:
            ]
        )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_feature_vector(
    output_dir,
):

    pslist = pd.read_csv(
        output_dir
        / "pslist.csv"
    )

    dlllist = pd.read_csv(
        output_dir
        / "dlllist.csv"
    )

    handles = pd.read_csv(
        output_dir
        / "handles.csv"
    )

    ldr = pd.read_csv(
        output_dir
        / "ldrmodules.csv"
    )

    malfind = pd.read_csv(
        output_dir
        / "malfind.csv"
    )


    # --------------------------------------------------------
    # PSList
    # --------------------------------------------------------

    features = {}

    features[
        "pslist.nproc"
    ] = len(pslist)

    features[
        "pslist.avg_threads"
    ] = pd.to_numeric(
        pslist["Threads"],
        errors="coerce",
    ).mean()

    features[
        "pslist.avg_handlers"
    ] = pd.to_numeric(
        pslist["Handles"],
        errors="coerce",
    ).mean()


    # --------------------------------------------------------
    # DLLList
    # --------------------------------------------------------

    features[
        "dlllist.ndlls"
    ] = len(dlllist)

    dlls_per_process = (
        dlllist
        .groupby("PID")
        .size()
    )

    features[
        "dlllist.avg_dlls_per_proc"
    ] = (
        dlls_per_process
        .mean()
    )


    # --------------------------------------------------------
    # HANDLES
    # --------------------------------------------------------

    features[
        "handles.nhandles"
    ] = len(handles)


    HANDLE_TYPES = {

        "handles.nfile":
            "File",

        "handles.nevent":
            "Event",

        "handles.ndesktop":
            "Desktop",

        "handles.nkey":
            "Key",

        "handles.nthread":
            "Thread",

        "handles.ndirectory":
            "Directory",

        "handles.nsemaphore":
            "Semaphore",

        "handles.ntimer":
            "Timer",

        "handles.nsection":
            "Section",

        "handles.nmutant":
            "Mutant",
    }


    for feature, handle_type in (
        HANDLE_TYPES.items()
    ):

        features[feature] = int(
            (
                handles["Type"]
                == handle_type
            ).sum()
        )


    # --------------------------------------------------------
    # LDRMODULES
    # --------------------------------------------------------

    ldr_rows = len(ldr)

    if ldr_rows == 0:

        features[
            "ldrmodules.not_in_load"
        ] = 0

        features[
            "ldrmodules.not_in_init"
        ] = 0

        features[
            "ldrmodules.not_in_mem"
        ] = 0

        features[
            "ldrmodules.not_in_load_avg"
        ] = 0

        features[
            "ldrmodules.not_in_init_avg"
        ] = 0

        features[
            "ldrmodules.not_in_mem_avg"
        ] = 0

    else:

        not_load = int(
            (~ldr["InLoad"])
            .sum()
        )

        not_init = int(
            (~ldr["InInit"])
            .sum()
        )

        not_mem = int(
            (~ldr["InMem"])
            .sum()
        )

        features[
            "ldrmodules.not_in_load"
        ] = not_load

        features[
            "ldrmodules.not_in_init"
        ] = not_init

        features[
            "ldrmodules.not_in_mem"
        ] = not_mem

        # Ratios of modules absent from
        # their corresponding loader lists.

        features[
            "ldrmodules.not_in_load_avg"
        ] = not_load / ldr_rows

        features[
            "ldrmodules.not_in_init_avg"
        ] = not_init / ldr_rows

        features[
            "ldrmodules.not_in_mem_avg"
        ] = not_mem / ldr_rows


    # --------------------------------------------------------
    # MALFIND
    # --------------------------------------------------------

    features[
        "malfind.ninjections"
    ] = len(malfind)

    commit_charge = pd.to_numeric(
        malfind[
            "CommitCharge"
        ],
        errors="coerce",
    ).fillna(0)

    features[
        "malfind.commitCharge"
    ] = float(
        commit_charge.sum()
    )


    # --------------------------------------------------------
    # CLEAN NUMBERS
    # --------------------------------------------------------

    result = {}

    for name in FEATURE_NAMES:

        value = features.get(
            name,
            0,
        )

        if pd.isna(value):

            value = 0

        result[name] = float(
            value
        )


    return pd.DataFrame(
        [result]
    )


# ============================================================
# MAIN FUNCTION
# ============================================================

def extract_memory_features(
    memory_path,
    output_dir=None,
    progress_callback=None,
):

    memory_path = Path(
        memory_path
    )

    if not memory_path.exists():

        raise FileNotFoundError(
            f"Memory image not found: "
            f"{memory_path}"
        )


    if not VOLATILITY.exists():

        raise FileNotFoundError(
            f"Volatility not found: "
            f"{VOLATILITY}"
        )


    if output_dir is None:

        output_dir = (
            DEFAULT_OUTPUT
        )


    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # --------------------------------------------------------
    # Step 1: OS information
    # --------------------------------------------------------

    if progress_callback:

        progress_callback(
            0.05,
            "Identifying memory image..."
        )

    info_path = (
        output_dir
        / "windows_info.txt"
    )

    run_volatility_info(
        memory_path,
        info_path,
    )


    # --------------------------------------------------------
    # Step 2-6: forensic plugins
    # --------------------------------------------------------

    total = len(
        CSV_PLUGINS
    )

    for index, (
        name,
        plugin
    ) in enumerate(
        CSV_PLUGINS.items(),
        start=1,
    ):

        if progress_callback:

            progress_callback(
                0.05
                + (
                    0.80
                    * index
                    / total
                ),

                f"Running Volatility: "
                f"{plugin}"
            )


        output_path = (
            output_dir
            / f"{name}.csv"
        )

        run_volatility_csv(
            memory_path,
            plugin,
            output_path,
        )


    # --------------------------------------------------------
    # Step 7: feature engineering
    # --------------------------------------------------------

    if progress_callback:

        progress_callback(
            0.90,
            "Building forensic feature vector..."
        )


    feature_vector = (
        build_feature_vector(
            output_dir
        )
    )


    feature_path = (
        output_dir
        / "aerotrace_features.csv"
    )

    feature_vector.to_csv(
        feature_path,
        index=False,
    )


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    pslist = pd.read_csv(
        output_dir
        / "pslist.csv"
    )

    dlllist = pd.read_csv(
        output_dir
        / "dlllist.csv"
    )

    handles = pd.read_csv(
        output_dir
        / "handles.csv"
    )

    malfind = pd.read_csv(
        output_dir
        / "malfind.csv"
    )


    summary = {

        "processes":
            int(len(pslist)),

        "dll_records":
            int(len(dlllist)),

        "handle_records":
            int(len(handles)),

        "malfind_findings":
            int(len(malfind)),

        "features_extracted":
            int(len(feature_vector.columns)),

        "feature_file":
            str(feature_path),

        "info_file":
            str(info_path),
    }


    with open(
        output_dir
        / "summary.json",
        "w",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )


    if progress_callback:

        progress_callback(
            1.0,
            "Forensic extraction complete."
        )


    return {
        "features":
            feature_vector,

        "summary":
            summary,

        "output_dir":
            output_dir,

        "info_text":
            info_path.read_text(
                encoding="utf-8",
                errors="replace",
            ),
    }


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "Usage:\n"
            "python extract_features.py "
            "<memory_dump>"
        )

        raise SystemExit(1)


    memory_dump = Path(
        sys.argv[1]
    )

    result = (
        extract_memory_features(
            memory_dump
        )
    )


    print(
        "\n" + "=" * 70
    )

    print(
        "AEROTRACE FEATURE EXTRACTION"
    )

    print(
        "=" * 70
    )

    print(
        json.dumps(
            result["summary"],
            indent=2,
        )
    )

    print(
        "\nFEATURE VECTOR:"
    )

    print(
        result["features"]
        .T
        .to_string(
            header=False
        )
    )

    print(
        "\n✅ COMPLETE"
    )