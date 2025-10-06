# Модифицированный скрипт: для каждой из 4 программ строит ровно 4 отдельных графика (каждый — своя figure):
# 1) speedup (все наборы N,M на одном графике)
# 2) efficiency (все наборы N,M на одном графике)
# 3) time vs numprocs (все наборы N,M на одном графике)
# 4) time_per_op vs numprocs (все наборы N,M на одном графике)
# Результаты также сохраняются в /mnt/data/figures_{program}/*.png
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("/mnt/data/figures", exist_ok=True)

data_tables = {
    "Таблица 1": [
        (2,20,2000000,14.06),
        (4,20,2000000,13.22),
        (8,20,2000000,13.09),
        (11,20,2000000,12.86),
        (2,50,800000,13.19),
        (4,50,800000,12.6),
        (8,50,800000,12.24),
        (11,50,800000,12.1),
        (2,100,200000,6.87),
        (4,100,200000,6.11),
        (8,100,200000,6.12),
        (11,100,200000,6.16),
    ],
    "Табилца 2": [
        (2,2000000,20,11.98),
        (4,2000000,20,12.01),
        (8,2000000,20,12.85),
        (11,2000000,20,12.21),
        (2,800000,50,11.92),
        (4,800000,50,12.16),
        (8,800000,50,12.65),
        (11,800000,50,12.32),
        (2,200000,100,6.27),
        (4,200000,100,6.31),
        (8,200000,100,6.6),
        (11,200000,100,6.32),
    ],
    "Таблица 3": [
        (2,20,2000000,15.056232099974295),
        (4,20,2000000,15.549593500007177),
        (8,20,2000000,14.530880900012562),
        (11,20,2000000,14.988982400012901),
        (2,50,800000,16.218943799991393),
        (4,50,800000,14.812278299999889),
        (8,50,800000,14.864018299995223),
        (11,50,800000,14.510582899994915),
        (2,100,200000,8.093111900001531),
        (4,100,200000,7.909049699985189),
        (8,100,200000,7.826438699994469),
        (11,100,200000,7.67567450000206),
    ],
    "Таблица 4": [
        (2,20,2000000,15.672760999994352),
        (4,20,2000000,15.238140200002817),
        (8,20,2000000,14.99483360000886),
        (11,20,2000000,14.602780599991092),
        (2,50,800000,15.130303400015691),
        (4,50,800000,14.508052300021518),
        (8,50,800000,14.480089200020302),
        (11,50,800000,14.40788870002143),
        (2,100,200000,8.4821735000005),
        (4,100,200000,7.32206970002153),
        (8,100,200000,6.985806099983165),
        (11,100,200000,6.970212299987907),
    ],
}

# вычисления метрик
dfs = {}
for name, rows in data_tables.items():
    df = pd.DataFrame(rows, columns=["numprocs","N","M","time_s"])
    results = []
    for (N,M), group in df.groupby(["N","M"]):
        baseline = group.loc[group['numprocs']==2, 'time_s'].values
        T2 = baseline[0] if len(baseline)>0 else np.nan
        for _, r in group.iterrows():
            n = int(r['numprocs'])
            Tn = float(r['time_s'])
            S_rel2 = T2 / Tn if not np.isnan(T2) else np.nan
            E_rel2 = S_rel2 / n
            ops = r['M'] * r['N']
            time_per_op = Tn / ops
            results.append({
                'numprocs': n, 'N': N, 'M': M, 'time_s': Tn,
                'T2_baseline_s': T2, 'S_rel2': S_rel2, 'E_rel2': E_rel2,
                'time_per_op': time_per_op
            })
    dfs[name] = pd.DataFrame(results).sort_values(['N','M','numprocs']).reset_index(drop=True)

# сохраняем таблицы на диск
for name, df in dfs.items():
    df.to_csv(f"/mnt/data/{name}_metrics.csv", index=False)

# Для каждой программы строим 4 отдельных графика (каждый — отдельная figure)
saved_figs = []
for name, df in dfs.items():
    figdir = f"/mnt/data/figures/{name}"
    os.makedirs(figdir, exist_ok=True)
    # уникальные наборы (N,M)
    groups = list(df.groupby(['N','M']))
    # 1) Speedup: все наборы на одном графике
    # plt.figure(figsize=(6,4))
    # for (N,M), g in groups:
    #     gs = g.sort_values('numprocs')
    #     plt.plot(gs['numprocs'], gs['S_rel2'], marker='o', label=f"N={N},M={M}")
    # plt.title(f"{name}: Ускорение")
    # plt.xlabel("numprocs")
    # plt.ylabel("S = T2 / Tn")
    # plt.grid(True)
    # plt.legend()
    # fname = os.path.join(figdir, f"{name}_speedup.png")
    # plt.savefig(fname, bbox_inches='tight')
    # saved_figs.append(fname)
    # plt.show()
    # plt.close()

    # 2) Efficiency: все наборы на одном графике
    # plt.figure(figsize=(6,4))
    # for (N,M), g in groups:
    #     gs = g.sort_values('numprocs')
    #     plt.plot(gs['numprocs'], gs['E_rel2'], marker='o', label=f"N={N},M={M}")
    # plt.title(f"{name}: Эффективность")
    # plt.xlabel("numprocs")
    # plt.ylabel("E = S / n")
    # plt.grid(True)
    # plt.legend()
    # fname = os.path.join(figdir, f"{name}_efficiency.png")
    # plt.savefig(fname, bbox_inches='tight')
    # saved_figs.append(fname)
    # plt.show()
    # plt.close()

    # # 3) Time vs numprocs: все наборы на одном графике
    # plt.figure(figsize=(6,4))
    # for (N,M), g in groups:
    #     gs = g.sort_values('numprocs')
    #     plt.plot(gs['numprocs'], gs['time_s'], marker='o', label=f"N={N},M={M}")
    # plt.title(f"{name}: time vs numprocs")
    # plt.xlabel("numprocs")
    # plt.ylabel("time (s)")
    # plt.grid(True)
    # plt.legend()
    # fname = os.path.join(figdir, f"{name}_time.png")
    # plt.savefig(fname, bbox_inches='tight')
    # saved_figs.append(fname)
    # plt.show()
    # plt.close()

    # 4) time_per_op vs numprocs (оценка слабой масштабируемости)
    plt.figure(figsize=(6,4))
    for (N,M), g in groups:
        gs = g.sort_values('numprocs')
        plt.plot(gs['numprocs'], gs['time_per_op'], marker='o', label=f"N={N},M={M}")
    plt.title(f"{name}: Слабая масштабируемость")
    plt.xlabel("numprocs")
    plt.ylabel("time per op (s / (M*N))")
    plt.grid(True)
    plt.legend()
    fname = os.path.join(figdir, f"{name}_time_per_op.png")
    plt.savefig(fname, bbox_inches='tight')
    saved_figs.append(fname)
    plt.show()
    plt.close()

# Показать сгенерированные файлы (список) и вывести сводную таблицу с метриками для контроля
summary_files = pd.DataFrame({'figure_path': saved_figs})

print("Готово — для каждой программы сохранено по 4 графика в /mnt/data/figures/{program}/ .")
