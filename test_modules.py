"""
Import test: verify that all modules can be imported and basic functions run without error.

Run from project root:
python test_modules.py --data data/penguins.csv --report_dir reports
"""

from pathlib import Path
import argparse

from src import (
    ensure_dirs,
    read_data,
    basic_profile,
    split_columns,
    summarize_numeric,
    summarize_categorical,
    correlations,
    missingness_table,
    plot_missingness,
    plot_corr_heatmap,
    plot_histograms,
    plot_bar_charts,
)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test src module imports + basic functions."
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Path to input CSV file."
    )
    parser.add_argument(
        "--report_dir", type=str, default="reports", help="Folder for outputs."
    )
    args = parser.parse_args()

    print("Data file:", args.data)
    print("Report folder:", args.report_dir)

    report_dir = Path(args.report_dir)
    ensure_dirs(report_dir)

    df = read_data(Path(args.data))

    profile = basic_profile(df)
    print("Profile:", profile)

    numeric_cols, cat_cols = split_columns(df)
    print("Numeric columns:", numeric_cols)
    print("Categorical columns:", cat_cols)

    # Run full pipeline behavior (Step 7 requirement)
    miss_df = missingness_table(df)
    num_summary = summarize_numeric(df, numeric_cols)
    cat_summary = summarize_categorical(df, cat_cols)
    corr = correlations(df, numeric_cols)

    (report_dir / "data_profile.json").write_text(str(profile))
    miss_df.to_csv(report_dir / "missingness_by_column.csv", index=False)
    num_summary.to_csv(report_dir / "summary_numeric.csv", index=False)
    cat_summary.to_csv(report_dir / "summary_categorical.csv", index=False)

    if not corr.empty:
        corr.to_csv(report_dir / "correlations.csv")

    plot_missingness(miss_df, report_dir / "figures" / "missingness.png")
    plot_corr_heatmap(corr, report_dir / "figures" / "corr_heatmap.png")
    plot_histograms(df, numeric_cols, report_dir / "figures")
    plot_bar_charts(df, cat_cols, report_dir / "figures")

    print("Test pipeline complete.")

if __name__ == "__main__":
    main()
