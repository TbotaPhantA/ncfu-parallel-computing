from mpi4py import MPI
import numpy as np
import time
import os

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

N = 10
a = np.full(N, rank, dtype=np.int32)
b = np.empty(N, dtype=np.int32)

# Синхронизация и замер времени
comm.Barrier()
start_time = time.time()

if size == 1:
    work_iters = 500_000
    work = 0
    for i in range(work_iters):
        work += (i * (rank + 1)) & 0xFFFF
    print(f'Rank {rank}: only one process, nothing to exchange. a={a}')
else:
    dst = (rank - 1) % size
    src = (rank + 1) % size

    req_send = comm.Isend(a, dest=dst, tag=0)
    req_recv = comm.Irecv(b, source=src, tag=0)

    work_iters = 500_000
    work = 0
    for i in range(work_iters):
        work += (i * (rank + 1)) & 0xFFFF
    MPI.Request.Waitall([req_send, req_recv])

    print(f'Rank {rank}: received b = {b.tolist()}, work = {work}')

# Замер времени и запись результатов
comm.Barrier()
end_time = time.time()
elapsed = end_time - start_time

if rank == 0:
    # Получаем время выполнения для 1 процесса (базовое)
    if size == 1:
        base_time = elapsed
    else:
        # Читаем базовое время из файла, если он существует
        if os.path.exists('results.csv'):
            with open('results.csv', 'r') as f:
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
    speedup = base_time / elapsed if base_time else None
    efficiency = speedup / size if speedup else None

    # Записываем заголовок и данные
    if not os.path.exists('results.csv'):
        with open('results.csv', 'w') as f:
            f.write('numprocs,time,speedup,efficiency\n')

    with open('results.csv', 'a') as f:
        if speedup and efficiency:
            f.write(f'{size},{elapsed:.6f},{speedup:.6f},{efficiency:.6f}\n')
        else:
            f.write(f'{size},{elapsed:.6f},,\n')
