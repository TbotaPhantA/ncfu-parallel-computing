#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_speedup_efficiency.py
Читает lr8-1.csv, lr8-2.csv, lr8-3.csv,
создаёт speedup.csv и efficiency.csv.

Использует время однопроцессной версии (lr8-1.csv, numprocs==1)
в качестве эталонного (T_serial).
"""

import pandas as pd
import sys

# имена входных файлов (ожидаются в той же папке)
FILES = [
    ("lr8-1.csv", "v1_serial"),
    ("lr8-2.csv", "v2_mp"),
    ("lr8-3.csv", "v3_mp"),
]

def main():
    dfs = []
    for fname, label in FILES:
        try:
            df = pd.read_csv(fname)
        except FileNotFoundError:
            print(f"Ошибка: файл '{fname}' не найден. Поместите файлы в ту же папку и повторите.", file=sys.stderr)
            return
        df = df.copy()
        df['program'] = label
        # обеспечить правильные типы
        df['numprocs'] = df['numprocs'].astype(int)
        df['time'] = df['time'].astype(float)
        dfs.append(df)

    # найти эталонное (serial) время: из lr8-1.csv, numprocs==1
    serial_df = dfs[0]  # lr8-1.csv
    mask1 = serial_df['numprocs'] == 1
    if not mask1.any():
        print("Ошибка: в lr8-1.csv нет записи с numprocs==1. Нужен эталонный однопроцессный замер.", file=sys.stderr)
        return
    T_serial = float(serial_df.loc[mask1, 'time'].iloc[0])
    print(f"Использую эталонное (serial) время из lr8-1.csv: T_serial = {T_serial}")

    # объединяем все данные
    combined = pd.concat(dfs, ignore_index=True, sort=False)
    combined = combined[['program', 'numprocs', 'N', 'M', 'time']]

    # вычисляем ускорение и эффективность:
    # speedup = T_serial / T_p
    combined['speedup'] = T_serial / combined['time']
    combined['efficiency'] = combined['speedup'] / combined['numprocs']

    # сохранить два csv:
    speedup_df = combined[['program', 'numprocs', 'speedup']].sort_values(['program','numprocs'])
    efficiency_df = combined[['program', 'numprocs', 'efficiency']].sort_values(['program','numprocs'])

    speedup_df.to_csv("speedup.csv", index=False)
    efficiency_df.to_csv("efficiency.csv", index=False)

    # также сохранить объединённые времена (для удобства)
    combined.to_csv("combined_times.csv", index=False)

    print("Созданы файлы: speedup.csv, efficiency.csv, combined_times.csv")
    print("Пример (speedup.csv):")
    print(speedup_df.to_string(index=False))

if __name__ == "__main__":
    main()
