from pathlib import Path
import pandas as pd


def ensure_dirs(reports: Path) -> None:
    (reports / "figures").mkdir(parents=True, exist_ok=True)


def read_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Loaded dataframe is empty.")

    return df
