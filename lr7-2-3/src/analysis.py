# -*- coding: utf-8 -*-
"""
plot_parallel_analysis.py
Построение:
 1) время выполнения vs N (лог-оси)
 2) ускорение S_n = T1 / Tn (c идеальной линией S=n)
 3) эффективность E_n = S_n / n

Зависимости строятся для numprocs = {1,2,4,8} и N = {10,100,1000,10000,100000,1000000}
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Данные (взяты из вашего сообщения) ---
seq_data = [
    {"numprocs":1, "N":10, "time":0.0024799000238999724},
    {"numprocs":1, "N":100, "time":0.002984600025229156},
    {"numprocs":1, "N":1000, "time":0.007361299940384924},
    {"numprocs":1, "N":10000, "time":0.05026160005945712},
    {"numprocs":1, "N":100000, "time":0.4558046000311151},
    {"numprocs":1, "N":1000000, "time":4.632463800022379},
]
par_data = [
    {"numprocs":2, "N":10, "time":0.007465700036846101},
    {"numprocs":4, "N":10, "time":0.0036025000736117363},
    {"numprocs":8, "N":10, "time":0.010382100008428097},
    {"numprocs":2, "N":100, "time":0.003797799930907786},
    {"numprocs":4, "N":100, "time":0.005016900016926229},
    {"numprocs":8, "N":100, "time":0.006045399932190776},
    {"numprocs":2, "N":1000, "time":0.007249900023452938},
    {"numprocs":4, "N":1000, "time":0.005853899987414479},
    {"numprocs":8, "N":1000, "time":0.007285700063221157},
    {"numprocs":2, "N":10000, "time":0.04357059998437762},
    {"numprocs":4, "N":10000, "time":0.02882010000757873},
    {"numprocs":8, "N":10000, "time":0.02261079999152571},
    {"numprocs":2, "N":100000, "time":0.37647769995965064},
    {"numprocs":4, "N":100000, "time":0.21551310003269464},
    {"numprocs":8, "N":100000, "time":0.17499800003133714},
    {"numprocs":2, "N":1000000, "time":3.7140806999523193},
    {"numprocs":4, "N":1000000, "time":2.050941599998623},
    {"numprocs":8, "N":1000000, "time":1.3909683000529185},
]

# --- Создаём DataFrame и считаем метрики ---
df_seq = pd.DataFrame(seq_data)
df_par = pd.DataFrame(par_data)
df = pd.concat([df_seq, df_par], ignore_index=True).sort_values(["N","numprocs"]).reset_index(drop=True)

# карта T1 по N
T1_map = df[df["numprocs"]==1].set_index("N")["time"].to_dict()
# присвоим T1 каждой строке (если T1 не найден — NaN)
df["T1"] = df["N"].map(T1_map)

# вычисляем ускорение и эффективность
df["speedup"] = df["T1"] / df["time"]
df["efficiency"] = df["speedup"] / df["numprocs"]

# пригодная для вывода таблица
display_df = df.copy()
display_df["time"] = display_df["time"].map(lambda x: float(f"{x:.9f}"))
display_df["T1"] = display_df["T1"].map(lambda x: float(f"{x:.9f}"))
display_df["speedup"] = display_df["speedup"].map(lambda x: float(f"{x:.4f}"))
display_df["efficiency"] = display_df["efficiency"].map(lambda x: float(f"{x:.4f}"))

print("\nТаблица с вычисленными метриками (time, T1, speedup, efficiency):\n")
print(display_df.to_string(index=False))

# --- Параметры для графиков ---
out_dir = "plots"
os.makedirs(out_dir, exist_ok=True)
marker_size = 6
line_width = 1.25

# --- График 1: время выполнения vs N (лог-ось) ---
plt.figure(figsize=(9,6))
for p in sorted(df["numprocs"].unique()):
    sub = df[df["numprocs"]==p].sort_values("N")
    plt.plot(sub["N"], sub["time"], marker='o', linestyle='-', markersize=marker_size, linewidth=line_width, label=f"{p} proc")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Размер N (логарифм)")
plt.ylabel("Время выполнения (с) (логарифм)")
plt.title("Время выполнения vs N для разных чисел процессов")
plt.grid(True, which="both", ls="--", lw=0.5)
plt.legend()
fname = os.path.join(out_dir, "time_vs_N.png")
plt.tight_layout()
plt.savefig(fname, dpi=300)
plt.show()
print(f"Saved: {fname}")

# --- График 2: ускорение S_n = T1 / Tn для каждого N + идеальная линия S=n ---
plt.figure(figsize=(9,6))
numprocs_unique = sorted(df["numprocs"].unique())
for N in sorted(df["N"].unique()):
    sub = df[df["N"]==N].sort_values("numprocs")
    plt.plot(sub["numprocs"], sub["speedup"], marker='o', linestyle='-', markersize=marker_size, linewidth=line_width, label=f"N={N}")
# идеальная линия S=n
xs = np.array(numprocs_unique)
plt.plot(xs, xs, linestyle='--', linewidth=1.2, label="Идеал S=n")
plt.xlabel("Число процессов n")
plt.ylabel("Ускорение S_n = T1 / Tn")
plt.title("Ускорение S_n для разных N (и идеал S=n)")
plt.xticks(xs)
plt.grid(True, ls="--", lw=0.5)
plt.legend()
fname = os.path.join(out_dir, "speedup.png")
plt.tight_layout()
plt.savefig(fname, dpi=300)
plt.show()
print(f"Saved: {fname}")

# --- График 3: эффективность E_n = S_n / n ---
plt.figure(figsize=(9,6))
for N in sorted(df["N"].unique()):
    sub = df[df["N"]==N].sort_values("numprocs")
    plt.plot(sub["numprocs"], sub["efficiency"], marker='o', linestyle='-', markersize=marker_size, linewidth=line_width, label=f"N={N}")
plt.xlabel("Число процессов n")
plt.ylabel("Эффективность E_n = S_n / n")
plt.title("Эффективность параллельной реализации для разных N")
plt.xticks(xs)
plt.grid(True, ls="--", lw=0.5)
plt.legend()
fname = os.path.join(out_dir, "efficiency.png")
plt.tight_layout()
plt.savefig(fname, dpi=300)
plt.show()
print(f"Saved: {fname}")

# --- Дополнительно: сохранить CSV с метриками ---
csv_fname = os.path.join(out_dir, "metrics_table.csv")
df.to_csv(csv_fname, index=False)
print(f"Saved metrics table to: {csv_fname}")
