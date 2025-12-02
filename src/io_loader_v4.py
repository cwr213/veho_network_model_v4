"""
Input Loading Module

Loads Excel workbook and converts to validated DataFrames.
All required sheets must be present - NO OPTIONAL SHEETS.
"""

import pandas as pd
from pathlib import Path
from typing import Dict

REQUIRED_SHEETS = [
    "facilities",
    "zips",
    "demand",
    "injection_distribution",
    "mileage_bands",
    "timing_params",
    "cost_params",
    "container_params",
    "package_mix",
    "run_settings",
    "scenarios"
]


def load_workbook(path: Path) -> Dict[str, pd.DataFrame]:
    """
    Load Excel workbook and return validated DataFrames.

    Args:
        path: Path to input Excel file

    Returns:
        Dictionary of sheet_name -> DataFrame

    Raises:
        ValueError: If required sheets missing or file cannot be read
    """
    try:
        excel_file = pd.ExcelFile(path)
    except Exception as e:
        raise ValueError(f"Could not read Excel file {path}: {e}")

    # Check for missing sheets
    missing_sheets = [s for s in REQUIRED_SHEETS if s not in excel_file.sheet_names]
    if missing_sheets:
        raise ValueError(
            f"Input file missing required sheets: {sorted(missing_sheets)}\n"
            f"Required sheets: {REQUIRED_SHEETS}"
        )

    # Load all sheets
    dfs = {}
    for sheet_name in REQUIRED_SHEETS:
        df = excel_file.parse(sheet_name)
        df = _clean_columns(df)
        dfs[sheet_name] = df

    # Clean up parameter sheets (remove empty rows)
    for param_sheet in ["timing_params", "cost_params", "run_settings"]:
        dfs[param_sheet] = dfs[param_sheet].dropna(how="all")

    dfs["scenarios"] = dfs["scenarios"].dropna(how="all")

    return dfs


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column names - strip whitespace.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with cleaned column names
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def params_to_dict(df_params: pd.DataFrame) -> Dict[str, any]:
    """
    Convert parameter sheet (key-value format) to dictionary.

    Args:
        df_params: DataFrame with columns 'key' and 'value'

    Returns:
        Dictionary of key -> value

    Raises:
        ValueError: If required columns missing
    """
    if not {"key", "value"}.issubset(df_params.columns):
        raise ValueError(
            f"Parameter sheet must have columns 'key' and 'value', "
            f"found: {list(df_params.columns)}"
        )

    params_dict = {}
    for _, row in df_params.iterrows():
        key = str(row["key"]).strip()
        value = row["value"]
        params_dict[key] = value

    return params_dict