import pandas as pd
from pathlib import Path
from typing import Dict

REQUIRED_SHEETS = [
    "zips", "facilities", "demand", "injection_distribution",
    "mileage_bands", "timing_params", "cost_params",
    "container_params", "package_mix", "run_settings", "scenarios"
]

def load_workbook(path: Path) -> Dict[str, pd.DataFrame]:
    xl = pd.ExcelFile(path)
    missing = [s for s in REQUIRED_SHEETS if s not in xl.sheet_names]
    if missing:
        raise ValueError(f"Missing sheets: {missing}")

    dfs = {s: xl.parse(s).pipe(_trim_cols) for s in REQUIRED_SHEETS}
    dfs["timing_params"] = dfs["timing_params"].dropna(how="all")
    dfs["cost_params"] = dfs["cost_params"].dropna(how="all")
    dfs["run_settings"] = dfs["run_settings"].dropna(how="all")
    dfs["scenarios"] = dfs["scenarios"].dropna(how="all")
    return dfs

def _trim_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    if "notes" in df.columns:
        # allow optional notes column
        pass
    return df

def params_to_dict(df_kv: pd.DataFrame) -> dict:
    # expects columns: key, value
    if not {"key", "value"}.issubset(df_kv.columns):
        raise ValueError("Params sheet must have columns: key, value")
    out = {}
    for _, r in df_kv.iterrows():
        k = str(r["key"]).strip()
        v = r["value"]
        out[k] = v
    return out
