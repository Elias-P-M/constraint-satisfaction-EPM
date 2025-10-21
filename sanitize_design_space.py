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
4. Min-max scale the prior columns as well as any column with prefix PROP/EQUIL.
5. Emit a JSON file containing the scaled constraint thresholds for downstream use.

Usage:
    python sanitize_design_space.py \
        --input design_space.xlsx \
        --output design_space_sanitized.csv \
        --threshold-output constraints_scaled.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

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

ADDITIONAL_SCALE_COLUMNS: Sequence[str] = (
    "YS 600 C PRIOR",
    "YS 25C PRIOR",
    "PROP 25C Density (g/cm3)",
    "Density Avg",
    "Pugh_Ratio_PRIOR",
    "PROP ST (K)",
    "Tm Avg",
    "VEC",
    "VEC Avg",
)

PREFIXES_TO_SCALE: Sequence[str] = ("PROP", "EQUIL")

THRESHOLD_VALUES: Dict[str, float] = {
    "PROP 25C Density (g/cm3)": 9.0,
    "YS 600 C PRIOR": 700.0,
    "Pugh_Ratio_PRIOR": 2.5,
    "PROP ST (K)": 2200.0 + 273.0,
    "VEC": 6.87,
    "VEC Avg": 6.87,
}

DEFAULT_INPUT = Path("design_space.xlsx")
DEFAULT_OUTPUT = Path("design_space_sanitized.csv")
DEFAULT_THRESHOLDS_OUTPUT = Path("constraints_scaled.json")


def load_design_space(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols: List[str] = []

    for col in ELEMENT_COLUMNS:
        if col in df.columns:
            keep_cols.append(col)

    for col in REQUIRED_COLUMNS:
        if col in df.columns and col not in keep_cols:
            keep_cols.append(col)

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


def scale_columns(
    df: pd.DataFrame,
    prefixes: Iterable[str],
    extra: Iterable[str],
) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    df_scaled = df.copy()
    prefixes_lower = tuple(p.lower() for p in prefixes)
    extra_set = set(extra)
    min_max: Dict[str, Tuple[float, float]] = {}

    for col in df_scaled.columns:
        lower = col.lower()
        should_scale = lower.startswith(prefixes_lower) or col in extra_set
        if not should_scale:
            continue
        series = pd.to_numeric(df_scaled[col], errors="coerce")
        col_min = series.min(skipna=True)
        col_max = series.max(skipna=True)
        if pd.isna(col_min) or pd.isna(col_max) or col_max == col_min:
            df_scaled[col] = 0.0
            min_max[col] = (float(col_min if pd.notna(col_min) else 0.0), float(col_max if pd.notna(col_max) else 0.0))
        else:
            df_scaled[col] = (series - col_min) / (col_max - col_min)
            min_max[col] = (float(col_min), float(col_max))
    return df_scaled, min_max


def sanitize(input_path: Path, output_path: Path, thresholds_output: Path) -> None:
    df = load_design_space(input_path)
    df = select_columns(df)
    df = rename_element_columns(df)
    df, min_max = scale_columns(df, PREFIXES_TO_SCALE, ADDITIONAL_SCALE_COLUMNS)
    df.to_csv(output_path, index=False)
    print(f"Sanitized dataset saved to {output_path} ({df.shape[0]} rows, {df.shape[1]} columns).")

    thresholds_scaled: Dict[str, float] = {}
    for col, value in THRESHOLD_VALUES.items():
        if col not in df.columns or col not in min_max:
            continue
        cmin, cmax = min_max[col]
        if cmax == cmin:
            scaled = 0.0
        else:
            scaled = (value - cmin) / (cmax - cmin)
        thresholds_scaled[col] = max(0.0, min(1.0, scaled))

    if thresholds_scaled:
        with thresholds_output.open("w", encoding="utf-8") as f:
            json.dump(thresholds_scaled, f, indent=2, sort_keys=True)
        print(f"Scaled constraint thresholds written to {thresholds_output}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Input design-space file (.xlsx or .csv).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output CSV path (default: design_space_sanitized.csv).")
    parser.add_argument("--threshold-output", type=Path, default=DEFAULT_THRESHOLDS_OUTPUT,
                        help="JSON file to store scaled constraint thresholds.")
    args = parser.parse_args()
    sanitize(args.input, args.output, args.threshold_output)


if __name__ == "__main__":
    main()
