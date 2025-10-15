#!/usr/bin/env python3
"""
plot_parallel_metrics.py

Считывает несколько CSV-файлов с колонками:
  numprocs, N, M, time, speedup, efficiency

Строит 3 графика (в одном окне, 3 subplot):
  1) время выполнения (time) vs numprocs
  2) speedup vs numprocs
  3) efficiency vs numprocs

Использование:
  python plot_parallel_metrics.py file1.csv file2.csv ... [--out output.png] [--no-show]

Если в одном CSV есть несколько строк с одинаковым numprocs, берётся среднее по этим строкам.

Автор: автоматически сгенерировано ChatGPT
"""

import argparse
import os
import sys
from typing import List

import pandas as pd
import matplotlib.pyplot as plt


REQUIRED_COLUMNS = {"numprocs", "time", "speedup", "efficiency"}


def read_and_prepare(path: str) -> pd.DataFrame:
    """Читает CSV и возвращает DataFrame, сгруппированный по numprocs и отсортированный по нему."""
    df = pd.read_csv(path)
    if not REQUIRED_COLUMNS.issubset(df.columns):
        raise ValueError(f"Файл {path} должен содержать колонки: {REQUIRED_COLUMNS}. Найдено: {list(df.columns)}")

    # Группируем на случай, если есть несколько измерений для одного числа процессов
    df = df.groupby("numprocs", as_index=False)[["time", "speedup", "efficiency"]].mean()
    df = df.sort_values("numprocs")
    return df


def plot_metrics(dfs: List[pd.DataFrame], labels: List[str], out: str = None, show: bool = True, logx: bool = False, logy: bool = False):
    """Строит три графика в вертикальной раскладке."""
    if len(dfs) == 0:
        raise ValueError("Нет данных для построения.")

    fig, axes = plt.subplots(3, 1, figsize=(9, 12), sharex=True)

    markers = ['o', 's', '^', 'D', 'v', 'x', 'P', '*']

    for i, (df, lbl) in enumerate(zip(dfs, labels)):
        m = markers[i % len(markers)]
        x = df['numprocs']
        axes[0].plot(x, df['time'], marker=m, linestyle='-', label=lbl)
        axes[1].plot(x, df['speedup'], marker=m, linestyle='-', label=lbl)
        axes[2].plot(x, df['efficiency'], marker=m, linestyle='-', label=lbl)

    axes[0].set_ylabel('time (s)')
    axes[1].set_ylabel('speedup')
    axes[2].set_ylabel('efficiency')
    axes[2].set_xlabel('numprocs')

    for ax in axes:
        ax.grid(True, linestyle=':', linewidth=0.6)
        ax.legend(loc='best', fontsize='small')

    if logx:
        axes[0].set_xscale('log', base=2)
        axes[1].set_xscale('log', base=2)
        axes[2].set_xscale('log', base=2)

    if logy:
        axes[0].set_yscale('log')
        axes[1].set_yscale('log')
        # efficiency обычно в [0,1], логарифм может не иметь смысла

    plt.suptitle('Parallel metrics: time / speedup / efficiency')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    if out:
        plt.savefig(out, dpi=300, bbox_inches='tight')
        print(f"Saved plots to {out}")

    if show:
        plt.show()


def parse_args(argv):
    p = argparse.ArgumentParser(description='Plot time, speedup and efficiency from CSV files')
    p.add_argument('files', nargs='+', help='CSV files to plot')
    p.add_argument('--out', '-o', help='Save figure to this file (png, pdf, ...)', default=None)
    p.add_argument('--no-show', action='store_true', help='Do not call plt.show() (useful for servers)')
    p.add_argument('--logx', action='store_true', help='Use log scale for x axis (base 2)')
    p.add_argument('--logy', action='store_true', help='Use log scale for y axes (not recommended for efficiency)')
    return p.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    dfs = []
    labels = []
    for f in args.files:
        if not os.path.exists(f):
            print(f"Warning: файл {f} не найден — пропускаю", file=sys.stderr)
            continue
        try:
            df = read_and_prepare(f)
        except Exception as e:
            print(f"Ошибка при чтении {f}: {e}", file=sys.stderr)
            continue
        dfs.append(df)
        labels.append(os.path.basename(f))

    if len(dfs) == 0:
        print("Нет корректных файлов для построения графиков.", file=sys.stderr)
        sys.exit(2)

    plot_metrics(dfs, labels, out=args.out, show=not args.no_show, logx=args.logx, logy=args.logy)


if __name__ == '__main__':
    main(sys.argv[1:])
