from typing import List
import pandas as pd


def summarize_numeric(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    if not numeric_cols:
        return pd.DataFrame(columns=[
            "column","count","mean","std","min","p25","median","p75","max"
        ])

    summary = df[numeric_cols].describe(
        percentiles=[0.25, 0.5, 0.75]
    ).T.rename(columns={
        "25%": "p25",
        "50%": "median",
        "75%": "p75"
    })

    summary.insert(0, "column", summary.index)
    summary.reset_index(drop=True, inplace=True)
    return summary


def summarize_categorical(df: pd.DataFrame, cat_cols: List[str], top_k: int = 10) -> pd.DataFrame:
    rows = []

    for c in cat_cols:
        series = df[c].astype("string")
        top = series.dropna().value_counts().head(top_k)

        rows.append({
            "column": c,
            "count": int(series.shape[0]),
            "missing": int(series.isna().sum()),
            "unique": int(series.nunique(dropna=True)),
            "top_values": "; ".join(
                [f"{idx} ({val})" for idx, val in top.items()]
            ),
        })

    return pd.DataFrame(rows)


def correlations(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    if len(numeric_cols) < 2:
        return pd.DataFrame()
    return df[numeric_cols].corr()


def missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    missing_rate = df.isna().mean()
    missing_count = df.isna().sum()

    result = pd.DataFrame({
        "column": missing_rate.index,
        "missing_rate": missing_rate.values,
        "missing_count": missing_count.values,
    })

    return result.sort_values(
        "missing_rate", ascending=False
    ).reset_index(drop=True)
