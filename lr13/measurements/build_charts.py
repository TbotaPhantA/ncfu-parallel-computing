#!/usr/bin/env python3
"""
plot_mpi_perf.py

Reads one or more CSV files containing MPI execution results and builds three charts:
 - execution time vs numprocs
 - speedup vs numprocs
 - efficiency vs numprocs

Usage:
    python build_charts.py file1.csv file2.csv ... \
        --out-prefix results --show --save-dir ./plots

If a CSV lacks 'speedup' or 'efficiency' columns the script will compute them:
 - baseline_time is taken from the smallest numprocs row (or --baseline-procs if provided).
 - speedup = baseline_time / time
 - efficiency = speedup / numprocs
"""

import argparse
import os
import sys
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt

def find_col_ci(columns, name):
    """Find column in columns by case-insensitive match; return actual column name or None."""
    name = name.lower()
    for c in columns:
        if c.lower() == name:
            return c
    return None

def load_and_prepare(path: str, baseline_procs: Optional[int] = None):
    df = pd.read_csv(path, skipinitialspace=True)
    # map important columns case-insensitively
    col_numprocs = find_col_ci(df.columns, "numprocs")
    col_time = find_col_ci(df.columns, "time")
    col_speedup = find_col_ci(df.columns, "speedup")
    col_eff = find_col_ci(df.columns, "efficiency")

    if col_numprocs is None:
        raise ValueError(f"'{path}': required column 'numprocs' not found.")

    # normalize column names
    df = df.rename(columns={col_numprocs: "numprocs"})
    if col_time:
        df = df.rename(columns={col_time: "time"})
    if col_speedup:
        df = df.rename(columns={col_speedup: "speedup"})
    if col_eff:
        df = df.rename(columns={col_eff: "efficiency"})

    # coerce numeric types
    df["numprocs"] = pd.to_numeric(df["numprocs"], errors="coerce").astype(int)
    if "time" in df.columns:
        df["time"] = pd.to_numeric(df["time"], errors="coerce")
    if "speedup" in df.columns:
        df["speedup"] = pd.to_numeric(df["speedup"], errors="coerce")
    if "efficiency" in df.columns:
        df["efficiency"] = pd.to_numeric(df["efficiency"], errors="coerce")

    df = df.sort_values("numprocs").reset_index(drop=True)

    # determine baseline time for computing speedup (if needed)
    if baseline_procs is not None and baseline_procs in df["numprocs"].values:
        baseline_time = float(df.loc[df["numprocs"] == baseline_procs, "time"].iloc[0])
    else:
        # default baseline: smallest numprocs row
        if "time" not in df.columns:
            raise ValueError(f"'{path}': cannot compute baselines because 'time' column is missing.")
        baseline_time = float(df["time"].iloc[0])

    # compute missing speedup/efficiency
    if "speedup" not in df.columns:
        df["speedup"] = baseline_time / df["time"]

    if "efficiency" not in df.columns:
        # efficiency = speedup / numprocs
        df["efficiency"] = df["speedup"] / df["numprocs"]

    return df

def plot_series(all_series, ylabel: str, out_path: str, title: str):
    """
    all_series: list of tuples (label, x_values, y_values)
    """
    plt.figure(figsize=(8,5))
    for label, x, y in all_series:
        plt.plot(x, y, marker='o', label=label)  # do not set colors (matplotlib default)
    plt.xlabel("numprocs")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle='--', linewidth=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot MPI execution time, speedup, and efficiency from CSV files.")
    parser.add_argument("csv_files", nargs="+", help="CSV files to read (each file becomes one series).")
    parser.add_argument("--out-prefix", default="mpi_results", help="Prefix for output files (default: mpi_results).")
    parser.add_argument("--save-dir", default=".", help="Directory to save PNG outputs (default: current directory).")
    parser.add_argument("--baseline-procs", type=int, default=None,
                        help="If provided, use the row with this numprocs as baseline when computing speedup.")
    parser.add_argument("--show", action="store_true", help="Also show plots interactively (if display is available).")
    args = parser.parse_args(argv)

    os.makedirs(args.save_dir, exist_ok=True)

    # collect series
    time_series = []
    speedup_series = []
    efficiency_series = []

    for path in args.csv_files:
        label = os.path.basename(path)
        try:
            df = load_and_prepare(path, baseline_procs=args.baseline_procs)
        except Exception as e:
            print(f"Error reading '{path}': {e}", file=sys.stderr)
            continue

        time_series.append((label, df["numprocs"].values, df["time"].values))
        speedup_series.append((label, df["numprocs"].values, df["speedup"].values))
        efficiency_series.append((label, df["numprocs"].values, df["efficiency"].values))

    # Plot 1: execution time
    out_time = os.path.join(args.save_dir, f"{args.out_prefix}_time.png")
    plot_series(time_series, ylabel="execution time (s)", out_path=out_time,
                title="Execution time vs numprocs")

    # Plot 2: speedup
    out_speedup = os.path.join(args.save_dir, f"{args.out_prefix}_speedup.png")
    plot_series(speedup_series, ylabel="speedup", out_path=out_speedup,
                title="Speedup vs numprocs")

    # Plot 3: efficiency
    out_eff = os.path.join(args.save_dir, f"{args.out_prefix}_efficiency.png")
    plot_series(efficiency_series, ylabel="efficiency", out_path=out_eff,
                title="Efficiency vs numprocs")

    print("Saved:")
    print(" -", out_time)
    print(" -", out_speedup)
    print(" -", out_eff)

    if args.show:
        # show the three images (open them in new figures)
        # we will re-create visible plots (matplotlib figures)
        for label, x, y in time_series:
            plt.figure(figsize=(8,5))
            plt.plot(x, y, marker='o', label=label)
        plt.xlabel("numprocs"); plt.ylabel("execution time (s)"); plt.title("Execution time vs numprocs")
        plt.legend(); plt.grid(True, linestyle='--', linewidth=0.3); plt.show()

        for label, x, y in speedup_series:
            plt.figure(figsize=(8,5))
            plt.plot(x, y, marker='o', label=label)
        plt.xlabel("numprocs"); plt.ylabel("speedup"); plt.title("Speedup vs numprocs")
        plt.legend(); plt.grid(True, linestyle='--', linewidth=0.3); plt.show()

        for label, x, y in efficiency_series:
            plt.figure(figsize=(8,5))
            plt.plot(x, y, marker='o', label=label)
        plt.xlabel("numprocs"); plt.ylabel("efficiency"); plt.title("Efficiency vs numprocs")
        plt.legend(); plt.grid(True, linestyle='--', linewidth=0.3); plt.show()

if __name__ == "__main__":
    main()
