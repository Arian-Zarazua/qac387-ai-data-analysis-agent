import json
from typing import Optional, Dict, Any
import pandas as pd


def assert_json_safe(obj, context: str = "") -> None:
    try:
        json.dumps(obj)
    except TypeError as e:
        raise AssertionError(
            f"Object not JSON-serializable"
            f"{': ' + context if context else ''}. {e}"
        )


def target_check(df: pd.DataFrame, target: str) -> Optional[dict]:
    if target not in df.columns:
        return None

    y = df[target]

    results: Dict[str, Any] = {
        "target": str(target),
        "dtype": str(y.dtype),
        "missing_rate": float(y.isna().mean()),
        "n_unique": int(y.nunique(dropna=True)),
    }

    if y.dtype.kind in "if":
        results.update({
            "mean": float(y.mean()),
            "std": float(y.std()),
            "min": float(y.min()),
            "max": float(y.max()),
        })
    else:
        top = y.astype(str).value_counts().head(5)
        results["top_values"] = {
            str(k): int(v) for k, v in top.items()
        }

    assert_json_safe(results, f"target_check({target})")
    return results
