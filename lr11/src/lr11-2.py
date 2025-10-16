from mpi4py import MPI
import numpy as np
import time
import os

comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()

N = 10
iters = 10

a = np.full(N, rank, dtype=np.int32)
a_recv = np.empty(N, dtype=np.int32)

# Синхронизация и замер общего времени
comm.Barrier()
start_time = time.time()

if numprocs == 1:
    work_iters = 300_000
    work = 0
    for i in range(work_iters):
        work += (i * (rank + 1)) & 0xFFFF
else:
    dst = (rank + 1) % numprocs
    src = (rank - 1) % numprocs

    req_send = comm.Send_init(a, dest=dst, tag=0)
    req_recv = comm.Recv_init(a_recv, source=src, tag=0)
    requests = [req_send, req_recv]

    work_iters = 300_000

    work_times = []
    wait_after_work_times = []
    total_times = []

    for k in range(iters):
        t0 = MPI.Wtime()
        MPI.Prequest.Startall(requests)

        work = 0
        for i in range(work_iters):
            work += (i * (rank + 1)) & 0xFFFF

        t1 = MPI.Wtime()
        MPI.Request.Waitall(requests)
        t2 = MPI.Wtime()

        a[:] = a_recv[:]

        work_times.append(t1 - t0)
        wait_after_work_times.append(t2 - t1)
        total_times.append(t2 - t0)

    for r in requests:
        r.Free()

# Замер общего времени выполнения
comm.Barrier()
end_time = time.time()
total_elapsed = end_time - start_time

# Вывод результатов работы
if numprocs == 1:
    print(f'Rank {rank}: only one process — nothing to exchange. a={a.tolist()}')
else:
    avg_work = sum(work_times) / len(work_times)
    avg_wait_after_work = sum(wait_after_work_times) / len(wait_after_work_times)
    avg_total = sum(total_times) / len(total_times)

    print(f'Rank {rank}: final a = {a.tolist()}')
    print(f'Rank {rank}: avg work time = {avg_work:.6f} s, '
          f'avg wait-after-work = {avg_wait_after_work:.6f} s, avg total = {avg_total:.6f} s')

# Запись результатов в CSV (только процесс 0)
if rank == 0:
    # Получаем базовое время для 1 процесса
    if numprocs == 1:
        base_time = total_elapsed
    else:
        # Читаем базовое время из файла, если он существует
        if os.path.exists('results_persistent.csv'):
            with open('results_persistent.csv', 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith('1,'):
                        base_time = float(line.split(',')[1])
                        break
                else:
                    base_time = None
        else:
            base_time = None

    # Вычисляем speedup и efficiency
    speedup = base_time / total_elapsed if base_time else None
    efficiency = speedup / numprocs if speedup else None

    # Записываем заголовок и данные
    if not os.path.exists('results_persistent.csv'):
        with open('results_persistent.csv', 'w') as f:
            f.write('numprocs,time,speedup,efficiency\n')

    with open('results_persistent.csv', 'a') as f:
        if speedup and efficiency:
            f.write(f'{numprocs},{total_elapsed:.6f},{speedup:.6f},{efficiency:.6f}\n')
        else:
            f.write(f'{numprocs},{total_elapsed:.6f},,\n')
    