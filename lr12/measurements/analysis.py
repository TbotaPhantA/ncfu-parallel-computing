#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def ensure_example_csv(path):
    if not os.path.exists(path):
        content = """type,N,M,time
openmp,100,200000,1.9
openmp,4000,5000,34.08
cuda,100,200000,1.95
cuda,4000,5000,10.44
"""
        with open(path, 'w') as f:
            f.write(content)
        print(f'Пример CSV создан: {path}')

def read_and_prepare(csv_path):
    df = pd.read_csv(csv_path)
    # Приведение типов и проверка
    df['N'] = pd.to_numeric(df['N'], errors='coerce').astype(int)
    df['M'] = pd.to_numeric(df['M'], errors='coerce').astype(int)
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df['elements'] = df['N'] * df['M']
    df['sample'] = df['N'].astype(str) + 'x' + df['M'].astype(str)
    return df

def plot_grouped_by_sample(df, out_png='bar_by_sample.png'):
    # агрегируем: среднее и std по (sample, type)
    agg = df.groupby(['sample','type']).agg(
        time_mean=('time','mean'),
        time_std=('time','std'),
        count=('time','count')
    ).reset_index()
    pivot_mean = agg.pivot(index='sample', columns='type', values='time_mean').fillna(0)
    pivot_std = agg.pivot(index='sample', columns='type', values='time_std').fillna(0)

    samples = pivot_mean.index.tolist()
    types = pivot_mean.columns.tolist()
    x = np.arange(len(samples))
    width = 0.8 / max(1, len(types))

    fig, ax = plt.subplots(figsize=(10,6))
    for i, t in enumerate(types):
        vals = pivot_mean[t].values
        errs = pivot_std[t].fillna(0).values
        ax.bar(x + i*width, vals, width=width, label=t, yerr=errs, capsize=4)
    ax.set_xticks(x + width*(len(types)-1)/2)
    ax.set_xticklabels(samples)
    ax.set_ylabel('Время, с')
    ax.set_title('Время по образцам (NxM) — grouped bar chart')
    ax.grid(axis='y', linestyle='--', linewidth=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    print(f'Сохранён: {out_png}')
    return agg

def plot_grouped_by_elements(df, out_png='bar_by_elements.png'):
    agg = df.groupby(['elements','type']).agg(
        time_mean=('time','mean'),
        time_std=('time','std'),
        count=('time','count')
    ).reset_index()
    pivot_mean = agg.pivot(index='elements', columns='type', values='time_mean').fillna(0)
    pivot_std = agg.pivot(index='elements', columns='type', values='time_std').fillna(0)

    elements = pivot_mean.index.tolist()
    types = pivot_mean.columns.tolist()
    x = np.arange(len(elements))
    width = 0.8 / max(1, len(types))

    fig, ax = plt.subplots(figsize=(10,6))
    for i, t in enumerate(types):
        vals = pivot_mean[t].values
        errs = pivot_std[t].fillna(0).values
        ax.bar(x + i*width, vals, width=width, label=t, yerr=errs, capsize=4)
    # читаемые подписи на оси X
    ax.set_xticks(x + width*(len(types)-1)/2)
    ax.set_xticklabels([f'{int(e):,}' for e in elements], rotation=45, ha='right')
    ax.set_ylabel('Время, с')
    ax.set_title('Время по числу элементов (elements = N*M) — grouped bar chart')
    ax.grid(axis='y', linestyle='--', linewidth=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    print(f'Сохранён: {out_png}')
    return agg

def main():
    parser = argparse.ArgumentParser(description='Plot bar charts from timings CSV.')
    parser.add_argument('csv', nargs='?', default='slay.csv', help='CSV файл (type,N,M,time)')
    parser.add_argument('--out-sample', default='bar_by_sample.png', help='PNG для графика по образцам')
    parser.add_argument('--out-elements', default='bar_by_elements.png', help='PNG для графика по элементам')
    args = parser.parse_args()

    ensure_example_csv(args.csv)
    df = read_and_prepare(args.csv)

    agg_sample = plot_grouped_by_sample(df, out_png=args.out_sample)
    agg_elements = plot_grouped_by_elements(df, out_png=args.out_elements)

    # Вывод агрегированных таблиц в консоль
    print('\nАгрегирование по образцам (sample x type):')
    print(agg_sample.to_string(index=False))
    print('\nАгрегирование по элементам (elements x type):')
    print(agg_elements.to_string(index=False))

if __name__ == '__main__':
    main()
