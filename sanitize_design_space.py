#!/usr/bin/env python3
"""
Produce a shareable design-space CSV containing only the columns required by
the active-learning campaigns in this repository.

Operations:
1. Keep only the elemental fractions used by the models (Nb, Mo, Ta, V, W, Cr).
   These are renamed to anonymised placeholders element_01 ... element_06.
2. Retain the objective/prior columns consumed by the scripts.
3. Preserve all phase-fraction columns that include both "600C" and "BCC"
   (used to build the single-phase BCC indicator).
4. Min-max scale any column whose name starts with PROP or EQUIL into [0, 1].

Usage:
    python sanitize_design_space.py \
        --input design_space.xlsx \
        --output design_space_sanitized.csv

The default locations match the files distributed with the project.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd

ELEMENT_COLUMNS: Sequence[str] = ("Nb", "Mo", "Ta", "V", "W", "Cr")

REQUIRED_COLUMNS: Sequence[str] = (
    "YS 600 C PRIOR",
    "YS 25C PRIOR",
    "PROP 25C Density (g/cm3)",
    "Density Avg",
    "Pugh_Ratio_PRIOR",
    "PROP ST (K)",
    "Tm Avg",
    "VEC Avg",
)

DEFAULT_INPUT = Path("design_space.xlsx")
DEFAULT_OUTPUT = Path("design_space_sanitized.csv")


def load_design_space(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols: List[str] = []

    # Elemental fractions (in declared order)
    for col in ELEMENT_COLUMNS:
        if col in df.columns:
            keep_cols.append(col)

    # Required scalar columns
    for col in REQUIRED_COLUMNS:
        if col in df.columns and col not in keep_cols:
            keep_cols.append(col)

    # All 600C BCC phase columns
    bcc_cols = [c for c in df.columns if "600C" in c and "BCC" in c]
    keep_cols.extend([c for c in bcc_cols if c not in keep_cols])

    missing = [c for c in ELEMENT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing elemental columns: {missing}")

    required_missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if required_missing:
        raise ValueError(f"Missing required columns: {required_missing}")

    if "VEC" in df.columns and "VEC" not in keep_cols:
        keep_cols.append("VEC")
    elif "VEC" not in df.columns and "VEC Avg" not in df.columns:
        raise ValueError("Neither 'VEC' nor 'VEC Avg' present in design space.")

    if not bcc_cols:
        raise ValueError("No 600C BCC columns found in design space.")

    return df.loc[:, keep_cols].copy()


def rename_element_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping: Dict[str, str] = {
        col: f"element_{idx+1:02d}" for idx, col in enumerate(ELEMENT_COLUMNS)
    }
    return df.rename(columns=mapping)


def scale_prop_equil(df: pd.DataFrame, prefixes: Iterable[str] = ("PROP", "EQUIL")) -> pd.DataFrame:
    df_scaled = df.copy()
    prefixes_lower = tuple(p.lower() for p in prefixes)
    for col in df_scaled.columns:
        lower = col.lower()
        if lower.startswith(prefixes_lower):
            series = pd.to_numeric(df_scaled[col], errors="coerce")
            col_min = series.min(skipna=True)
            col_max = series.max(skipna=True)
            if pd.isna(col_min) or pd.isna(col_max) or col_max == col_min:
                df_scaled[col] = 0.0
            else:
                df_scaled[col] = (series - col_min) / (col_max - col_min)
    return df_scaled


def sanitize(input_path: Path, output_path: Path) -> None:
    df = load_design_space(input_path)
    df = select_columns(df)
    df = rename_element_columns(df)
    df = scale_prop_equil(df)
    df.to_csv(output_path, index=False)
    print(f"Sanitized dataset saved to {output_path} ({df.shape[0]} rows, {df.shape[1]} columns).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Input design-space file (.xlsx or .csv).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output CSV path (default: design_space_sanitized.csv).")
    args = parser.parse_args()
    sanitize(args.input, args.output)


if __name__ == "__main__":
    main()
