# plotting_speedup_efficiency.py
import os
import pandas as pd
import matplotlib.pyplot as plt

# --- Настройки ---
csv_files = ["lr6-2-3.csv", "lr11-3.csv"]  # замените путями к вашим файлам при необходимости

# Если у вас есть реальные T1 (в секундах) для конкретных файлов, укажите их сюда.
# Пример: {"lr3-1.csv": 30.5, "lr3-2.csv": 29.8}
# Если оставите словарь пустым, будет использовано время при минимальном numprocs в каждом файле.
serial_times = {
    # "lr3-1.csv": 30.5,
    # "lr3-2.csv": 29.8,
    # "lr5.csv": 50.0,
}

# --- Функции ---
def read_measurements(path):
    df = pd.read_csv(path)
    # Ожидаем колонки: numprocs, N, M, time
    required = {"numprocs", "time"}
    if not required.issubset(df.columns):
        raise ValueError(f"Файл {path} не содержит требуемых колонок {required}")
    # Убедимся, что numprocs целые
    df["numprocs"] = df["numprocs"].astype(int)
    return df[["numprocs", "time"]].sort_values("numprocs")

def compute_speedup_efficiency(df, t1=None):
    """
    Возвращает DataFrame с колонками: numprocs, time, speedup, efficiency
    Если t1==None, функция вернёт speedup относительно времени на минимальном numprocs в df.
    """
    df = df.copy().reset_index(drop=True)
    if t1 is None:
        baseline_n = int(df["numprocs"].min())
        t_ref = float(df.loc[df["numprocs"] == baseline_n, "time"].iloc[0])
        note = f"baseline_n={baseline_n}"
    else:
        t_ref = float(t1)
        baseline_n = 1
        note = "baseline_n=1 (real T1 provided)"
    df["speedup"] = t_ref / df["time"]
    # Если baseline_n != 1, мы пометим, что эффективность — относительная.
    # Для формулы используем E_n = (t_ref / t_n) / numprocs  (интерпретация зависит от t_ref)
    df["efficiency"] = df["speedup"] / df["numprocs"]
    return df, note

# --- Сбор данных ---
all_results = {}  # filename -> (df_with_metrics, note)
for f in csv_files:
    if not os.path.exists(f):
        print(f"WARNING: файл {f} не найден — пропускаю.")
        continue
    df = read_measurements(f)
    basename = os.path.basename(f)
    if basename in serial_times and serial_times[basename] is not None:
        t1 = serial_times[basename]
    else:
        t1 = None
    metrics_df, note = compute_speedup_efficiency(df, t1=t1)
    all_results[basename] = (metrics_df, note)

if not all_results:
    raise SystemExit("Не найдено ни одного CSV для построения. Проверьте пути в csv_files.")

# --- Рисуем ---
# 1) График ускорения (все программы)
plt.figure(figsize=(8,5))
for name, (dfm, note) in all_results.items():
    plt.plot(dfm["numprocs"], dfm["speedup"], marker="o", label=f"{name} ({note})")
plt.xlabel("Число процессов (numprocs)")
plt.ylabel("Ускорение S_n (T_ref / T_n)")
plt.title("Ускорение для всех программ")
plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(sorted({n for dfm,_ in all_results.values() for n in dfm["numprocs"]}))
plt.legend()
plt.tight_layout()
plt.savefig("speedup_all.png", dpi=150)
plt.show()

# 2) График эффективности (все программы)
plt.figure(figsize=(8,5))
for name, (dfm, note) in all_results.items():
    plt.plot(dfm["numprocs"], dfm["efficiency"], marker="s", label=f"{name} ({note})")
plt.xlabel("Число процессов (numprocs)")
plt.ylabel("Эффективность E_n (S_n / n)")
plt.title("Эффективность для всех программ")
plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(sorted({n for dfm,_ in all_results.values() for n in dfm["numprocs"]}))
plt.legend()
plt.tight_layout()
plt.savefig("efficiency_all.png", dpi=150)
plt.show()

print("Готово. Сохранены файлы: speedup_all.png и efficiency_all.png")
print("Если у вас есть истинные T1 для файлов — укажите их в serial_times для корректных графиков.")
