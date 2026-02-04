"""
Build 0: Data Analysis Pipeline ( ASSIGNMENT 1)

You will complete a baseline data analysis pipeline. This file is meant to be run
from the terminal, and later reused as a module (tool) in agentic builds.

WHAT YOU MUST DO
1) Fill in the 10 code blanks marked BLANK 1 ... BLANK 10.
- Each blank is a single line (or small block) that has been removed.
- Replace the placeholder with correct Python code.

2) Write TWO new functions (these are NOT "blanks"—you write them fully):
A) missingness_table(df)
    - Return a DataFrame with columns:
        ["column", "missing_rate", "missing_count"]
    - Sorted by missing_rate descending

B) multiple_linear_regression(df, outcome, predictors=None)
    - Fit a multiple linear regression model.
    - outcome: name of outcome column (must be numeric)
    - predictors: optional list of predictor column names.
        If None, use ALL numeric columns except the outcome.
    - Raise a ValueError if outcome is not numeric.
    - Return a dictionary of JSON-safe results (no numpy/pandas scalars),
        including at least:
        {
        "outcome": ...,
        "predictors": [...],
        "n_rows_used": ...,
        "r_squared": ...,
        "adj_r_squared": ...,
        "coefficients": {predictor: coef, ...},
        "intercept": ...,
        }

HOW TO RUN (example): You can copy and paste this command (all one line) in your terminal after
replacing Target_Column, Outcome_Column, Predictor1, Predictor2 with actual column names from your dataset:

python builds/Build0_data_analysis_pipeline_assignment_1.py --data data/penguins.csv --target Target_Column --outcome Outcome_Column --predictors Predictor1,Predictor2 --report_dir reports/


Outputs will be written to:
reports/
data_profile.json
summary_numeric.csv
summary_categorical.csv
missingness_by_column.csv
correlations.csv (if available)
regression_results.json (if you provide --outcome)
figures/
    missingness.png
    corr_heatmap.png
    hist_<col>.png
    bar_<col>.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Utilities
# -----------------------------


def ensure_dirs(reports: Path) -> None:
    """Create output folders."""
    # BLANK 1: create the figures folder
    # HINT: (reports / "figures").mkdir(...)
    ___BLANK_1___


def read_data(path: Path) -> pd.DataFrame:
    """Read a CSV file into a DataFrame with basic error handling."""
    # BLANK 2: raise FileNotFoundError if path does not exist
    ___BLANK_2___

    # BLANK 3: read the CSV into df
    ___BLANK_3___

    if df.empty:
        raise ValueError("Loaded dataframe is empty.")
    return df


# -----------------------------
# Data profiling
# -----------------------------


def basic_profile(df: pd.DataFrame) -> dict:
    """Return a basic JSON-serializable profile of the dataset."""
    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        # BLANK 4: list of column names
        "columns": ___BLANK_4___,
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "n_missing_total": int(df.isna().sum().sum()),
        "missing_by_col": df.isna().sum().to_dict(),
        "memory_mb": float(df.memory_usage(deep=True).sum() / (1024**2)),
    }


def split_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Identify and split numeric vs categorical columns into numeric and categorical lists."""
    # BLANK 5: list numeric column names
    # HINT: df.select_dtypes(include=["number"]).columns.______
    numeric_cols = ___BLANK_5___

    # Treat everything else as categorical
    cat_cols = [c for c in df.columns if c not in numeric_cols]
    return numeric_cols, cat_cols


# -----------------------------
# Summaries
# -----------------------------


def summarize_numeric(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """Compute descriptive statistics for numeric columns."""
    if not numeric_cols:
        return pd.DataFrame(
            columns=[
                "column",
                "count",
                "mean",
                "std",
                "min",
                "p25",
                "median",
                "p75",
                "max",
            ]
        )

    # BLANK 6: Create a transposed describe table with percentiles 0.25, 0.5, 0.75
    # HINT: df[numeric_cols].describe(...).T
    summary = ___BLANK_6___

    summary = summary.rename(columns={"50%": "median", "25%": "p25", "75%": "p75"})
    summary.insert(0, "column", summary.index)
    summary.reset_index(drop=True, inplace=True)
    return summary


def summarize_categorical(
    df: pd.DataFrame, cat_cols: List[str], top_k: int = 10
) -> pd.DataFrame:
    """Compute descriptive statistics for categorical columns."""
    rows = []
    for c in cat_cols:
        series = df[c].astype("string")
        n = int(series.shape[0])
        n_missing = int(series.isna().sum())
        n_unique = int(series.nunique(dropna=True))

        # BLANK 7: top_k value counts (drop missing)
        top = ___BLANK_7___

        rows.append(
            {
                "column": c,
                "count": n,
                "missing": n_missing,
                "unique": n_unique,
                "top_values": "; ".join([f"{idx} ({val})" for idx, val in top.items()]),
            }
        )
    return pd.DataFrame(rows)


# -----------------------------
# REQUIRED: Student-built functions
# -----------------------------


def missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    TODO (Student task): Create a missingness table.

    Requirements:
    - Compute missing_rate for each column (fraction missing)
    - Compute missing_count for each column
    - Return a DataFrame with columns:
        column, missing_rate, missing_count
    - Sort by missing_rate descending

    Hints:
    - df.isna().mean() gives missing rates
    - df.isna().sum() gives missing counts
    """
    raise NotImplementedError("Student must implement missingness_table(df).")


def multiple_linear_regression(
    df: pd.DataFrame, outcome: str, predictors: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    TODO (Student task): Fit a multiple linear regression model.

    Requirements:
    - Outcome must be numeric; raise ValueError otherwise
    - If predictors is None:
        use ALL numeric columns except outcome
    - Drop rows with missing values in outcome or predictors before fitting
    - Fit the model using least squares:
        y = intercept + b1*x1 + b2*x2 + ...
    - Return a JSON-safe dictionary containing:
        outcome, predictors, n_rows_used, r_squared, adj_r_squared,
        intercept, coefficients (dict)

    Hints: use statsmodels package:
    import statsmodels.api as sm
    X = df[predictors]
    X = sm.add_constant(X)
    y = df[outcome]
    model = sm.OLS(y, X).fit()

    IMPORTANT:
    - Convert any numpy/pandas scalars to Python floats/ints before returning.
    """
    raise NotImplementedError(
        "Student must implement multiple_linear_regression(df, outcome, predictors=None)."
    )


def correlations(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """Compute correlations for numeric columns."""
    if len(numeric_cols) < 2:
        return pd.DataFrame()
    # BLANK 8: compute correlation matrix for numeric columns
    corr = ___BLANK_8___
    return corr


# -----------------------------
# Plots
# -----------------------------


def plot_missingness(miss_df: pd.DataFrame, out_path: Path, top_n: int = 30) -> None:
    """Plot missing data in a horizontal bar chart."""
    plot_df = miss_df.head(top_n).iloc[::-1]
    plt.figure()
    # BLANK 9: create a horizontal bar chart using column names and missing_rate
    ___BLANK_9___
    plt.xlabel("Missing rate")
    plt.title(f"Top {min(top_n, len(miss_df))} columns by missingness")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_corr_heatmap(corr: pd.DataFrame, out_path: Path) -> None:
    """Create a heatmap of correlations."""
    if corr.empty:
        return
    plt.figure()
    plt.imshow(corr.values, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
    plt.yticks(range(len(corr.index)), corr.index, fontsize=7)
    plt.title("Correlation heatmap (numeric columns)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_histograms(
    df: pd.DataFrame, numeric_cols: List[str], fig_dir: Path, max_cols: int = 12
) -> None:
    """Plot histograms for numeric columns."""
    for c in numeric_cols[:max_cols]:
        series = df[c].dropna()
        if series.empty:
            continue
        plt.figure()
        plt.hist(series, bins=30)
        plt.title(f"Histogram: {c}")
        plt.xlabel(c)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(fig_dir / f"hist_{c}.png", dpi=200)
        plt.close()


def plot_bar_charts(
    df: pd.DataFrame,
    cat_cols: List[str],
    fig_dir: Path,
    max_cols: int = 12,
    top_k: int = 20,
) -> None:
    """Plot bar charts for categorical columns."""
    for c in cat_cols[:max_cols]:
        series = df[c].astype("string").dropna()
        if series.empty:
            continue
        vc = series.value_counts().head(top_k)
        plt.figure()
        plt.bar(vc.index.astype(str), vc.values)
        plt.title(f"Top {min(top_k, len(vc))} values: {c}")
        plt.xticks(rotation=90, fontsize=7)
        plt.tight_layout()
        plt.savefig(fig_dir / f"bar_{c}.png", dpi=200)
        plt.close()


# -----------------------------
# Simple model check
# -----------------------------


def assert_json_safe(obj, context: str = "") -> None:
    """Assert that an object can be serialized to JSON."""
    try:
        json.dumps(obj)
    except TypeError as e:
        raise AssertionError(
            f"Object is not JSON-serializable{': ' + context if context else ''}.\n"
            f"Hint: Convert Pandas / NumPy types to native Python types like "
            f"(str, int, float, list, dict).\n"
            f"Original error: {e}"
        )


def target_check(df: pd.DataFrame, target: str) -> Optional[dict]:
    """Look at a target column and return basic information about it."""
    if target not in df.columns:
        print(f"Column '{target}' not found.")
        return None

    y = df[target]

    results: Dict[str, Any] = {}
    results["target"] = str(target)
    results["dtype"] = str(y.dtype)
    results["missing_rate"] = float(y.isna().mean())
    results["n_unique"] = int(y.nunique(dropna=True))

    if y.dtype.kind in "if":
        results["mean"] = float(y.mean())
        results["std"] = float(y.std())
        results["min"] = float(y.min())
        results["max"] = float(y.max())
    else:
        top = y.astype(str).value_counts().head(5)
        results["top_values"] = {str(k): int(v) for k, v in top.items()}

    assert_json_safe(results, context=f"target_check output for column '{target}'")
    return results


# -----------------------------
# Main pipeline
# -----------------------------


def main():
    """
    The `main()` function is the entry point of this script.

    Why do we define a main function?
    --------------------------------
    - It clearly separates *what the program does* from helper functions.
    - It allows this file to be imported into another Python file without
    automatically running the analysis.
    - It makes the code easier to test, reuse, and later turn into tools
    or agent workflows.

    When this file is run from the command line, Python will call `main()`.
    """

    # ----------------------------------------
    # 1. Set up command-line arguments
    # ----------------------------------------
    # argparse lets users control the program from the terminal
    # without editing the code itself.
    parser = argparse.ArgumentParser()

    # Required argument: path to the dataset
    parser.add_argument("--data", type=str, required=True, help="Path to CSV file")

    # Column to treat as a target for summary checks
    parser.add_argument("--target", type=str, default=None, help="target column")

    # Outcome variable for regression
    parser.add_argument(
        "--outcome", type=str, default=None, help="Optional outcome for regression"
    )

    # Predictors for regression (comma-separated string)
    parser.add_argument(
        "--predictors",
        type=str,
        default=None,
        help="Comma-separated predictors for regression (e.g., 'age,fare')",
    )

    # Directory where outputs will be saved
    parser.add_argument(
        "--report_dir", type=str, default="reports", help="Output directory"
    )

    # Parse all command-line arguments into a single object
    args = parser.parse_args()

    # ----------------------------------------
    # 2. Prepare output directories
    # ----------------------------------------
    report_dir = Path(args.report_dir)

    # Ensure the output directory (and subfolders) exist
    ensure_dirs(report_dir)

    # ----------------------------------------
    # 3. Load data and identify column types
    # ----------------------------------------
    df = read_data(Path(args.data))

    # Split columns into numeric vs categorical
    numeric_cols, cat_cols = split_columns(df)

    # ----------------------------------------
    # 4. Generate summary outputs
    # ----------------------------------------
    profile = basic_profile(df)
    miss_df = missingness_table(df)
    num_summary = summarize_numeric(df, numeric_cols)
    cat_summary = summarize_categorical(df, cat_cols)
    corr = correlations(df, numeric_cols)

    # ----------------------------------------
    # 5. Save tabular outputs to disk
    # ----------------------------------------
    (report_dir / "data_profile.json").write_text(json.dumps(profile, indent=2))
    miss_df.to_csv(report_dir / "missingness_by_column.csv", index=False)
    num_summary.to_csv(report_dir / "summary_numeric.csv", index=False)
    cat_summary.to_csv(report_dir / "summary_categorical.csv", index=False)

    # Only save correlations if at least one exists
    if not corr.empty:
        corr.to_csv(report_dir / "correlations.csv")

    # ----------------------------------------
    # 6. Generate and save plots
    # ----------------------------------------
    plot_missingness(miss_df, report_dir / "figures" / "missingness.png")
    plot_corr_heatmap(corr, report_dir / "figures" / "corr_heatmap.png")
    plot_histograms(df, numeric_cols, report_dir / "figures")
    plot_bar_charts(df, cat_cols, report_dir / "figures")

    # ----------------------------------------
    # 7. Target variable checks
    # ----------------------------------------
    # Only run this section if --target was provided
    if args.target:
        target_info = target_check(df, args.target)
        (report_dir / "target_overview.json").write_text(
            json.dumps(target_info, indent=2)
        )

    # ----------------------------------------
    # 8. Regression analysis
    # ----------------------------------------
    # Only runs if --outcome is provided
    if args.outcome:
        preds: Optional[List[str]] = None

        # If predictors were provided, convert the comma-separated string
        # into a clean Python list
        if args.predictors:
            # BLANK 10: parse comma-separated predictors into a list of cleaned names
            # HINT: [p.strip() for p in args.predictors.split(",") if p.strip()]
            preds = ___BLANK_10___

        # Run the regression
        reg_results = multiple_linear_regression(
            df, outcome=args.outcome, predictors=preds
        )

        # Ensure the output can be safely saved as JSON
        assert_json_safe(reg_results, context="multiple_linear_regression output")

        # Save regression results
        (report_dir / "regression_results.json").write_text(
            json.dumps(reg_results, indent=2)
        )

    # ----------------------------------------
    # 9. Final user message
    # ----------------------------------------
    print(f"Build0 pipeline complete. Outputs saved to: {report_dir.resolve()}")


# ------------------------------------------------------------
# This conditional ensures that main() only runs when this file
# is executed directly (not when it is imported as a module).
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
