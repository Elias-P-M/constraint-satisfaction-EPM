import argparse
import os
import time
from pathlib import Path
import pandas as pd


def stitch_results(results_dir: str, prefix: str, output: str) -> None:
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Results dir not found: {results_dir}")

    print(results_dir)
    print(os.listdir(results_dir))

    filenames = [f for f in os.listdir(results_dir) if prefix in f]
    filenames = sorted(filenames)
    print(filenames)

    df = pd.DataFrame()
    for file in filenames:
        filepath = os.path.join(results_dir, file)
        size = os.path.getsize(filepath)
        print(f"Reading '{file}' (size: {size} bytes)")

        if size == 0:
            print(f"⚠️ Skipping empty file: {file}")
            continue

        try:
            dfi = pd.read_csv(filepath)
            print(f"✅ Read {file}: shape = {dfi.shape}")
            df = pd.concat([df, dfi], axis=0, ignore_index=True)
        except pd.errors.EmptyDataError:
            print(f"❌ EmptyDataError in file: {file}")
        except Exception as e:
            print(f"❌ Error reading {file}: {e}")

    df = df.sort_index()
    print(len(df))
    df.to_csv(output, index=False)
    print(f"Wrote {output}")


def main() -> None:
    p = argparse.ArgumentParser(description="Stitch per-seed campaign CSVs.")
    p.add_argument("--results-dir", help="Directory containing campaign CSVs")
    p.add_argument("--prefix", default="campaign_", help="Filename prefix to match")
    p.add_argument("--output", help="Output stitched CSV path")
    p.add_argument("--auto", action="store_true", help="Auto-scan all results_* dirs in CWD")
    args = p.parse_args()

    tic = time.time()
    if args.auto:
        base = Path.cwd()
        for path in sorted(base.glob("results_*")):
            if not path.is_dir():
                continue
            out_name = f"{path.name}_stitch.csv"
            stitch_results(str(path), args.prefix, str(path / out_name))
    else:
        if not args.results_dir or not args.output:
            raise SystemExit("Provide --results-dir and --output (or use --auto).")
        stitch_results(args.results_dir, args.prefix, args.output)
    print(f"Done in {time.time() - tic:.2f}s")


if __name__ == "__main__":
    main()
