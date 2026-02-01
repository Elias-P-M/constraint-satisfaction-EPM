#!/usr/bin/env python3
"""
Reproduce campaign runs from results_* directory names, then stitch and plot.

Naming scheme:
{date}_{method}{-prior|-noprior}_{acq}_{data}_{seeds}x{iters}
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import datetime
from pathlib import Path


RUN_RE = re.compile(
    r"^(?P<date>\\d{4}-\\d{2}-\\d{2})_"
    r"(?P<method>[^_]+)_"
    r"(?P<acq>[^_]+)_"
    r"(?P<data>[^_]+)_"
    r"(?P<seeds>\\d+)x(?P<iters>\\d+)$"
)


def infer_scaled_space(path: str | None) -> bool:
    import pandas as pd
    if path is None:
        for cand in ("design_space.xlsx", "design_space.csv"):
            if Path(cand).exists():
                path = cand
                break
    if path is None:
        raise FileNotFoundError("design_space.xlsx or design_space.csv not found")
    if path.lower().endswith(".xlsx"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    return float(df["PROP 25C Density (g/cm3)"].max(skipna=True)) <= 1.5


def build_run_name(method: str, acq: str, data_tag: str, seeds: int, iters: int) -> str:
    date_str = datetime.date.today().isoformat()
    return f"{date_str}_{method}_{acq}_{data_tag}_{seeds}x{iters}"


def parse_run_name(name: str) -> dict:
    m = RUN_RE.match(name)
    if not m:
        raise ValueError(f"Unrecognized run name: {name}")
    return m.groupdict()


def script_for_method(method_tag: str, acq: str) -> str:
    if method_tag == "const-prior":
        return "campaign_const_w_priors.py"
    if method_tag == "const-noprior":
        return "campaign_const_NO_priors.py"
    if method_tag == "random-noprior":
        return "campaign_random.py"
    if method_tag.startswith("opt"):
        if acq == "ehvipof":
            return "campaign_ehvi_pof.py"
        if acq == "ehvipofcorr":
            return "campaign_ehvi_pofcorr.py"
        return "campaign_pehvi.py"
    raise ValueError(f"Unknown method tag: {method_tag}")


def run_campaign(run_name: str, args: dict, base: Path, cfg: argparse.Namespace) -> None:
    method_tag = args["method"]
    seeds = cfg.seeds if cfg.seeds is not None else f"1-{int(args['seeds'])}"
    iters = cfg.iterations if cfg.iterations is not None else int(args["iters"])
    workers = cfg.workers if cfg.workers is not None else 3

    results_dir = base / f"results_{run_name}"
    plots_dir = base / f"plots_{run_name}"
    if cfg.results_dir:
        results_dir = Path(cfg.results_dir) / run_name
    if cfg.plots_dir:
        plots_dir = Path(cfg.plots_dir) / run_name

    if cfg.reset:
        if results_dir.exists():
            shutil.rmtree(results_dir)
        if plots_dir.exists():
            shutil.rmtree(plots_dir)

    script = script_for_method(method_tag, args["acq"])
    cmd = [
        "python",
        script,
        "--seeds", str(seeds),
        "--iterations", str(iters),
        "--workers", str(workers),
        "--results-dir", str(results_dir),
        "--plots-dir", str(plots_dir),
    ]
    if cfg.data_path:
        cmd += ["--data-path", cfg.data_path]
    if cfg.density_thresh is not None:
        cmd += ["--density-thresh", str(cfg.density_thresh)]
    if cfg.ys_thresh is not None:
        cmd += ["--ys-thresh", str(cfg.ys_thresh)]
    if cfg.pugh_thresh is not None:
        cmd += ["--pugh-thresh", str(cfg.pugh_thresh)]
    if cfg.st_thresh is not None:
        cmd += ["--st-thresh", str(cfg.st_thresh)]
    if cfg.vec_thresh is not None:
        cmd += ["--vec-thresh", str(cfg.vec_thresh)]
    if cfg.plot_affine:
        cmd += ["--plot-affine"]
    if cfg.plot_every is not None:
        cmd += ["--plot-every", str(cfg.plot_every)]
    if script == "campaign_pehvi.py" and cfg.fixed_range_scope:
        cmd += ["--fixed-range-scope", cfg.fixed_range_scope]
    # keep affine plots off by default; user can rerun separately if needed
    subprocess.run(cmd, cwd=str(base), check=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Reproduce all results_* runs, stitch, and plot.")
    p.add_argument("--reset", action="store_true", help="Remove existing results/plots before rerun")
    p.add_argument("--seeds", help="Seeds spec (e.g. 1-5 or 1,2,5-8)")
    p.add_argument("--iterations", type=int, help="Iterations per seed override")
    p.add_argument("--workers", type=int, help="Worker count override")
    p.add_argument("--data-path", help="Path to design_space.xlsx or .csv")
    p.add_argument("--results-dir", help="Base output dir for results (per run subdir)")
    p.add_argument("--plots-dir", help="Base output dir for plots (per run subdir)")
    p.add_argument("--density-thresh", type=float, help="Density threshold override")
    p.add_argument("--ys-thresh", type=float, help="Yield strength threshold override")
    p.add_argument("--pugh-thresh", type=float, help="Pugh ratio threshold override")
    p.add_argument("--st-thresh", type=float, help="Solidus temperature threshold override")
    p.add_argument("--vec-thresh", type=float, help="VEC threshold override")
    p.add_argument("--plot-affine", action="store_true", help="Enable affine plots")
    p.add_argument("--plot-every", type=int, help="Plot every N iterations")
    p.add_argument("--fixed-range-scope", choices=["ALL", "BCC_ONLY"], help="pEHVI fixed range scope")
    args = p.parse_args()

    base = Path(__file__).resolve().parent
    runs = []
    for path in sorted(base.glob("results_*")):
        if not path.is_dir():
            continue
        run_name = path.name[len("results_") :]
        try:
            info = parse_run_name(run_name)
        except ValueError:
            continue
        runs.append((run_name, info))

    if not runs:
        data_tag = "scaled" if infer_scaled_space(args.data_path) else "raw"
        seeds = 1
        iters = 1
        if args.seeds:
            # only for naming; campaigns will use the actual override
            if "-" in args.seeds:
                start, end = args.seeds.split("-", 1)
                seeds = abs(int(end) - int(start)) + 1
            else:
                seeds = len(args.seeds.split(","))
        if args.iterations:
            iters = args.iterations
        runs = [
            (build_run_name("const-prior", "feasibility", data_tag, seeds, iters), {"method": "const-prior", "acq": "feasibility", "seeds": str(seeds), "iters": str(iters)}),
            (build_run_name("const-noprior", "feasibility", data_tag, seeds, iters), {"method": "const-noprior", "acq": "feasibility", "seeds": str(seeds), "iters": str(iters)}),
            (build_run_name("opt-noprior", "pehvi", data_tag, seeds, iters), {"method": "opt-noprior", "acq": "pehvi", "seeds": str(seeds), "iters": str(iters)}),
            (build_run_name("opt-noprior", "ehvipof", data_tag, seeds, iters), {"method": "opt-noprior", "acq": "ehvipof", "seeds": str(seeds), "iters": str(iters)}),
            (build_run_name("opt-noprior", "ehvipofcorr", data_tag, seeds, iters), {"method": "opt-noprior", "acq": "ehvipofcorr", "seeds": str(seeds), "iters": str(iters)}),
            (build_run_name("random-noprior", "random", data_tag, seeds, iters), {"method": "random-noprior", "acq": "random", "seeds": str(seeds), "iters": str(iters)}),
        ]

    completed_counts = {}
    for run_name, info in runs:
        print(f"Reproducing {run_name}...")
        run_campaign(run_name, info, base, args)
        key = f"{info['method']}_{info['acq']}"
        completed_counts[key] = completed_counts.get(key, 0) + 1

    # Stitch and plot
    subprocess.run(["python", "stitch.py", "--auto"], cwd=str(base), check=True)
    subprocess.run(["python", "plot_metrics.py", "--auto", "--out-dir", "plots_metrics_all"], cwd=str(base), check=True)

    if completed_counts:
        print("\nCompleted runs by campaign type:")
        for k in sorted(completed_counts.keys()):
            print(f"  {k}: {completed_counts[k]} run(s)")


if __name__ == "__main__":
    main()
