from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

N = 10

a = np.full(N, rank, dtype=np.int32)
b = np.empty(N, dtype=np.int32)

if size == 1:
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
