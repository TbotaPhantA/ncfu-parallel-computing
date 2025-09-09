# TODO:
# 1. improve the slay and slay_simplified using tikhonov correction
# 2. use  ||r|| < epsilon instead of s = N
# 3. every program appends to file the number of processes and execution time
# 4. generate two charts for every programm execution time to number of processes

#!/usr/bin/env python3
"""
plot_speedup_efficiency.py

Читает два CSV-файла с колонками:
nprocs,time_seconds

и строит:
 - график Speedup (T(base) / T(p)) с линией идеального ускорения,
 - график Efficiency (Speedup / ideal_speedup).

По умолчанию ожидает файлы:
 - timings_slay.csv
 - timings_slay_simplified.csv

Можно передать пути к файлам через параметры командной строки.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_csv_safe(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл не найден: {path}")
    df = pd.read_csv(path)
    if 'nprocs' not in df.columns or 'time_seconds' not in df.columns:
        raise ValueError(f"В файле {path} ожидаются колонки 'nprocs' и 'time_seconds'")
    # приведение типов и сортировка по nprocs
    df = df.copy()
    df['nprocs'] = df['nprocs'].astype(int)
    df['time_seconds'] = df['time_seconds'].astype(float)
    df = df.sort_values('nprocs').reset_index(drop=True)
    return df

def compute_metrics(df, baseline_nprocs=None):
    df = df.copy()
    if baseline_nprocs is None:
        p0 = int(df['nprocs'].min())
    else:
        p0 = int(baseline_nprocs)
        if p0 not in df['nprocs'].values:
            raise ValueError("Запрошенный baseline nprocs не найден в данных.")
    T0 = float(df.loc[df['nprocs'] == p0, 'time_seconds'].iloc[0])
    df['speedup'] = T0 / df['time_seconds']
    df['ideal_speedup'] = df['nprocs'] / p0
    df['efficiency'] = df['speedup'] / df['ideal_speedup']
    return df, p0, T0

def plot_and_save(df_list, labels, out_prefix="output"):
    """
    df_list: list of dataframes with computed metrics (must contain nprocs,speedup,ideal_speedup,efficiency)
    labels: list of labels for legend matching df_list
    """
    # collect all unique nprocs for consistent xticks
    all_nprocs = sorted(set(np.concatenate([df['nprocs'].values for df in df_list])))
    # --- Speedup plot ---
    fig1, ax1 = plt.subplots(figsize=(9,6))
    for df, label in zip(df_list, labels):
        ax1.plot(df['nprocs'], df['speedup'], marker='o', linestyle='-', label=label)
    # plot ideal relative to the baseline p0 of the first dataset (for visual reference)
    # We'll draw ideal line using first df's baseline
    baseline_p0 = int(df_list[0]['nprocs'].min())
    ideal = np.array(all_nprocs) / baseline_p0
    ax1.plot(all_nprocs, ideal, linestyle='--', label=f"Ideal (baseline {baseline_p0})")
    ax1.set_xscale('log', base=2)
    ax1.set_xticks(all_nprocs)
    ax1.get_xaxis().set_major_formatter(lambda val, pos: f"{int(val)}")
    ax1.set_xlabel('Number of processes (nprocs)')
    ax1.set_ylabel('Speedup (T(base) / T(n))')
    ax1.set_title('Speedup vs Number of processes')
    ax1.grid(True, which='both', linestyle=':', linewidth=0.6)
    ax1.legend()
    fig1.tight_layout()
    speedup_path = f"{out_prefix}_speedup.png"
    fig1.savefig(speedup_path, dpi=200)
    print(f"Saved speedup plot to: {speedup_path}")

    # --- Efficiency plot ---
    fig2, ax2 = plt.subplots(figsize=(9,6))
    for df, label in zip(df_list, labels):
        ax2.plot(df['nprocs'], df['efficiency'], marker='o', linestyle='-', label=label)
    ax2.set_xscale('log', base=2)
    ax2.set_xticks(all_nprocs)
    ax2.get_xaxis().set_major_formatter(lambda val, pos: f"{int(val)}")
    ax2.set_xlabel('Number of processes (nprocs)')
    ax2.set_ylabel('Efficiency (fraction of ideal, 1.0 = 100%)')
    ax2.set_title('Parallel Efficiency vs Number of processes')
    ax2.grid(True, which='both', linestyle=':', linewidth=0.6)
    ax2.legend()
    fig2.tight_layout()
    eff_path = f"{out_prefix}_efficiency.png"
    fig2.savefig(eff_path, dpi=200)
    print(f"Saved efficiency plot to: {eff_path}")

    # also show the plots (this will open windows in interactive environments)
    plt.show()

def print_table(df, name):
    df_print = df.copy()
    df_print['time_seconds'] = df_print['time_seconds'].map(lambda x: f"{x:.6f}")
    df_print['speedup'] = df_print['speedup'].map(lambda x: f"{x:.6f}")
    df_print['ideal_speedup'] = df_print['ideal_speedup'].map(lambda x: f"{x:.1f}")
    df_print['efficiency'] = df_print['efficiency'].map(lambda x: f"{100*x:.3f}%")
    print(f"\nResults for {name}:")
    print(df_print[['nprocs','time_seconds','speedup','ideal_speedup','efficiency']].to_string(index=False))

def main():
    parser = argparse.ArgumentParser(description="Plot Speedup and Efficiency from two timing CSV files.")
    parser.add_argument("file1", nargs='?', default="timings_slay.csv", help="Первый CSV-файл (по умолчанию timings_slay.csv)")
    parser.add_argument("file2", nargs='?', default="timings_slay_simplified.csv", help="Второй CSV-файл (по умолчанию timings_slay_simplified.csv)")
    parser.add_argument("--baseline", type=int, default=None, help="Использовать конкретный baseline nprocs (если задан).")
    parser.add_argument("--out-prefix", default="timings", help="Префикс для сохраняемых файлов (по умолчанию 'timings').")
    args = parser.parse_args()

    try:
        df1 = read_csv_safe(args.file1)
        df2 = read_csv_safe(args.file2)
    except Exception as e:
        print("Ошибка при чтении входных файлов:", e, file=sys.stderr)
        sys.exit(1)

    try:
        df1m, p0_1, T0_1 = compute_metrics(df1, baseline_nprocs=args.baseline)
        df2m, p0_2, T0_2 = compute_metrics(df2, baseline_nprocs=args.baseline)
    except Exception as e:
        print("Ошибка при вычислении метрик:", e, file=sys.stderr)
        sys.exit(1)

    print(f"Baselines used: file1 baseline p0 = {p0_1}, T(p0) = {T0_1:.9f} s")
    print(f"                file2 baseline p0 = {p0_2}, T(p0) = {T0_2:.9f} s")
    print_table(df1m, os.path.basename(args.file1))
    print_table(df2m, os.path.basename(args.file2))

    plot_and_save([df1m, df2m], [os.path.basename(args.file1), os.path.basename(args.file2)], out_prefix=args.out_prefix)

if __name__ == "__main__":
    main()
