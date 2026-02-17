from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import statsmodels.api as sm


def _is_numeric_series(s: pd.Series) -> bool:
    return np.issubdtype(s.dtype, np.number)


def multiple_linear_regression(
    df: pd.DataFrame,
    outcome: str,
    predictors: Optional[List[str]] = None,
) -> Dict[str, Any]:

    if outcome not in df.columns:
        raise ValueError(f"Outcome column '{outcome}' not found.")

    if not _is_numeric_series(df[outcome]):
        raise ValueError("Outcome variable must be numeric.")

    if predictors is None:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        predictors = [c for c in numeric_cols if c != outcome]

    if not predictors:
        raise ValueError("No valid predictors available.")

    model_df = df[[outcome] + predictors].dropna()

    X = sm.add_constant(model_df[predictors])
    y = model_df[outcome]

    model = sm.OLS(y, X).fit()

    return {
        "outcome": str(outcome),
        "predictors": list(predictors),
        "n_rows_used": int(model_df.shape[0]),
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "intercept": float(model.params["const"]),
        "coefficients": {
            str(k): float(v)
            for k, v in model.params.items()
            if k != "const"
        },
    }
