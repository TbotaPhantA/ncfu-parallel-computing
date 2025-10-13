import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

# Цвета/стили можно менять; оставил стандартные matplotlib
PROGRAM_LABELS = {
    "v1_serial": "версия 1 (serial)",
    "v2_mp": "версия 2 (mp)",
    "v3_mp": "версия 3 (mp)"
}

def safe_read_csv(fname):
    try:
        return pd.read_csv(fname)
    except FileNotFoundError:
        print(f"Внимание: файл '{fname}' не найден.", file=sys.stderr)
        return None

def prepare_time_df():
    # читаем оригинальные файлы (или объединённый combined_times.csv, если есть)
    combined = safe_read_csv("combined_times.csv")
    if combined is None:
        # читаем по-отдельности и объединяем
        dfs = []
        mapping = [("lr8-1.csv", "v1_serial"), ("lr8-2.csv", "v2_mp"), ("lr8-3.csv", "v3_mp")]
        for fname, label in mapping:
            df = safe_read_csv(fname)
            if df is None:
                continue
            df = df.copy()
            df['program'] = label
            dfs.append(df[['program','numprocs','time']])
        if not dfs:
            raise RuntimeError("Нет входных CSV для времени.")
        combined = pd.concat(dfs, ignore_index=True)
    else:
        combined = combined[['program','numprocs','time']]

    # убедимся в типах
    combined['numprocs'] = combined['numprocs'].astype(int)
    combined['time'] = combined['time'].astype(float)
    return combined

def main():
    time_df = prepare_time_df()

    speedup_df = safe_read_csv("speedup.csv")
    efficiency_df = safe_read_csv("efficiency.csv")

    if speedup_df is None or efficiency_df is None:
        print("Файлы speedup.csv или efficiency.csv не найдены — попытка построить графики без них.", file=sys.stderr)

    # список всех numprocs для выставления xticks
    all_numprocs = sorted(time_df['numprocs'].unique())

    # подготовка фигуры: 3 строки, 1 столбец
    fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
    plt.subplots_adjust(hspace=0.35)

    # --- 1) Время выполнения ---
    ax = axes[0]
    for prog_key, label in PROGRAM_LABELS.items():
        sub = time_df[time_df['program'] == prog_key]
        if sub.empty:
            continue
        # сортировка по numprocs
        sub = sub.sort_values('numprocs')
        ax.plot(sub['numprocs'], sub['time'], marker='o', label=label)
    ax.set_xscale('linear')
    ax.set_xlabel("")  # общий xlabel снизу
    ax.set_ylabel("Время (s)")
    ax.set_title("Время выполнения")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()

    # --- 2) Ускорение ---
    ax = axes[1]
    if speedup_df is not None:
        for prog_key, label in PROGRAM_LABELS.items():
            sub = speedup_df[speedup_df['program'] == prog_key]
            if sub.empty:
                continue
            sub = sub.sort_values('numprocs')
            ax.plot(sub['numprocs'], sub['speedup'], marker='o', label=label)
    else:
        ax.text(0.5, 0.5, "speedup.csv не найден", ha='center', va='center')
    ax.set_ylabel("Ускорение S(p)")
    ax.set_title("Ускорение")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()

    # --- 3) Эффективность ---
    ax = axes[2]
    if efficiency_df is not None:
        for prog_key, label in PROGRAM_LABELS.items():
            sub = efficiency_df[efficiency_df['program'] == prog_key]
            if sub.empty:
                continue
            sub = sub.sort_values('numprocs')
            ax.plot(sub['numprocs'], sub['efficiency'], marker='o', label=label)
    else:
        ax.text(0.5, 0.5, "efficiency.csv не найден", ha='center', va='center')
    ax.set_xlabel("Число процессов (numprocs)")
    ax.set_ylabel("Эффективность E(p)")
    ax.set_title("Эффективность")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()

    # xticks — все уникальные значения numprocs
    axes[-1].set_xticks(all_numprocs)
    axes[-1].set_xticklabels([str(x) for x in all_numprocs])

    # сохранить картинку
    outname = "performance_plots.png"
    fig.savefig(outname, dpi=150, bbox_inches='tight')
    print(f"Графики сохранены в {outname}")

    plt.show()

if __name__ == "__main__":
    main()
