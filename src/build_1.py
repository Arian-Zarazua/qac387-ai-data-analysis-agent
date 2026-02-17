import argparse
import json
from pathlib import Path

from src.io_utils import ensure_dirs, read_data
from src.profiling import basic_profile, split_columns
from src.summaries import (
    summarize_numeric,
    summarize_categorical,
    missingness_table,
    correlations,
)
from src.plotting import (
    plot_missingness,
    plot_corr_heatmap,
    plot_histograms,
    plot_bar_charts,
)
from src.modeling import multiple_linear_regression
from src.checks import assert_json_safe, target_check


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--target", default=None)
    parser.add_argument("--outcome", default=None)
    parser.add_argument("--predictors", default=None)
    parser.add_argument("--report_dir", default="reports")
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    ensure_dirs(report_dir)

    df = read_data(Path(args.data))
    numeric_cols, cat_cols = split_columns(df)

    profile = basic_profile(df)
    miss_df = missingness_table(df)
    num_summary = summarize_numeric(df, numeric_cols)
    cat_summary = summarize_categorical(df, cat_cols)
    corr = correlations(df, numeric_cols)

    (report_dir / "data_profile.json").write_text(json.dumps(profile, indent=2))
    miss_df.to_csv(report_dir / "missingness_by_column.csv", index=False)
    num_summary.to_csv(report_dir / "summary_numeric.csv", index=False)
    cat_summary.to_csv(report_dir / "summary_categorical.csv", index=False)

    if not corr.empty:
        corr.to_csv(report_dir / "correlations.csv")

    plot_missingness(miss_df, report_dir / "figures" / "missingness.png")
    plot_corr_heatmap(corr, report_dir / "figures" / "corr_heatmap.png")
    plot_histograms(df, numeric_cols, report_dir / "figures")
    plot_bar_charts(df, cat_cols, report_dir / "figures")

    if args.target:
        target_info = target_check(df, args.target)
        (report_dir / "target_overview.json").write_text(
            json.dumps(target_info, indent=2)
        )

    if args.outcome:
        preds = None
        if args.predictors:
            preds = [p.strip() for p in args.predictors.split(",") if p.strip()]

        reg_results = multiple_linear_regression(
            df, outcome=args.outcome, predictors=preds
        )

        assert_json_safe(reg_results, "regression output")

        (report_dir / "regression_results.json").write_text(
            json.dumps(reg_results, indent=2)
        )

    print(f"Build1 pipeline complete. Outputs saved to: {report_dir.resolve()}")


if __name__ == "__main__":
    main()
