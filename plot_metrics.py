#!/usr/bin/env python3
"""
Plot metric vs iteration curves (mean ± std) for campaign results.

Assumes per-seed CSVs in each results directory. Each CSV should have an
Iteration column. This script aggregates across seeds per iteration and
plots selected metrics for the three approaches:
  - no prior (results_const_no_prior)
  - prior (results_const_prior)
  - opt (results_opt)
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class SeriesStats:
    mean: np.ndarray
    std: np.ndarray


def _load_campaign_csvs(results_dir: str, prefix: str = "campaign_") -> Optional[pd.DataFrame]:
    if not os.path.isdir(results_dir):
        return None
    files = [f for f in os.listdir(results_dir) if f.startswith(prefix) and f.endswith(".csv")]
    if not files:
        return None
    dfs = []
    for f in sorted(files):
        path = os.path.join(results_dir, f)
        if os.path.getsize(path) == 0:
            continue
        dfs.append(pd.read_csv(path))
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def _scan_results_dirs(base_dir: str) -> List[Tuple[str, str]]:
    dirs = []
    for name in sorted(os.listdir(base_dir)):
        if not name.startswith("results_"):
            continue
        full = os.path.join(base_dir, name)
        if os.path.isdir(full):
            dirs.append((name, full))
    return dirs


def _aggregate_by_iteration(df: pd.DataFrame, col: str, iterations: Optional[int]) -> SeriesStats:
    if col not in df.columns:
        raise KeyError(f"Missing column '{col}' in results.")
    if "Iteration" not in df.columns:
        raise KeyError("Missing column 'Iteration' in results.")

    if iterations is None:
        max_it = int(df["Iteration"].max())
        iters = range(0, max_it + 1)
    else:
        iters = range(0, iterations)

    means: List[float] = []
    stds: List[float] = []
    for it in iters:
        sub = df[df["Iteration"] == it][col]
        if sub.empty:
            means.append(np.nan)
            stds.append(np.nan)
        else:
            means.append(float(sub.mean()))
            stds.append(float(sub.std(ddof=1)) if len(sub) > 1 else 0.0)
    return SeriesStats(mean=np.array(means), std=np.array(stds))


def _plot_metric(
    metric: str,
    x: np.ndarray,
    series: List[tuple[str, SeriesStats]],
    out_dir: str,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> None:
    plt.figure(figsize=(8, 5))
    for label, stats in series:
        color = None
        plt.plot(x, stats.mean, color=color, linewidth=2, label=label)
        plt.fill_between(x, stats.mean - stats.std, stats.mean + stats.std, color=color, alpha=0.2)

    plt.title(title or metric)
    plt.xlabel("Iteration")
    plt.ylabel(ylabel or metric)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{metric}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot metric vs iteration for campaign results.")
    p.add_argument("--no-prior-dir", default="results_const_no_prior", help="Results dir for no-prior campaign")
    p.add_argument("--prior-dir", default="results_const_prior", help="Results dir for prior campaign")
    p.add_argument("--opt-dir", default="results_opt", help="Results dir for pEHVI campaign")
    p.add_argument("--iterations", type=int, help="Number of iterations to plot (default: infer from data)")
    p.add_argument("--out-dir", default="plots_metrics", help="Output directory for plots")
    p.add_argument("--auto", action="store_true", help="Auto-scan all results_* directories in CWD")
    p.add_argument(
        "--metrics",
        default="Accuracy,Precision,Recall,F1,LogLoss,BrierLoss,TrueParetoCount,TrueMeasPass_All_With_BCC,TrueMeasPass_All_No_BCC,Hypervolume_Scaled_FixedRange",
        help="Comma-separated list of metric columns to plot",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    datasets: List[Tuple[str, pd.DataFrame]] = []
    if args.auto:
        for name, path in _scan_results_dirs(os.getcwd()):
            df = _load_campaign_csvs(path)
            if df is not None:
                datasets.append((name, df))
    else:
        df_no_prior = _load_campaign_csvs(args.no_prior_dir)
        df_prior = _load_campaign_csvs(args.prior_dir)
        df_opt = _load_campaign_csvs(args.opt_dir)
        if df_no_prior is not None:
            datasets.append(("Constraint Satisfaction (No Prior)", df_no_prior))
        if df_prior is not None:
            datasets.append(("Constraint Satisfaction (Prior)", df_prior))
        if df_opt is not None:
            datasets.append(("Optimization (pEHVI)", df_opt))

    if not datasets:
        raise FileNotFoundError("No result CSVs found in any provided results directories.")

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    iterations = args.iterations
    if iterations is None:
        max_it = max(int(df["Iteration"].max()) for _, df in datasets)
        iterations = max_it + 1
    x = np.arange(iterations)

    for metric in metrics:
        series: List[tuple[str, SeriesStats]] = []
        for label, df in datasets:
            if metric not in df.columns:
                continue
            series.append((label, _aggregate_by_iteration(df, metric, iterations)))
        if not series:
            continue
        _plot_metric(metric, x, series, args.out_dir)

    print(f"Wrote plots to {args.out_dir}")


if __name__ == "__main__":
    main()
