import time
from mpi4py import MPI
import numpy as np

try:
    import cupy as cp
except ImportError:
    raise ImportError("cupy не найден. Установите cupy и обеспечьте наличие CUDA/GPU, "
                      "чтобы использовать GPU-ускорение.")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numprocs = comm.Get_size()

prefixPath = '../data/data_A/'
inPath = prefixPath + 'in.dat'
aDataPath = prefixPath + 'AData.dat'
bDataPath = prefixPath + 'bData.dat'

start_time = time.time()

if rank == 0:
    f1 = open(inPath, 'r')
    N = np.array(np.int32(f1.readline()))
    M = np.array(np.int32(f1.readline()))
    f1.close()
else:
    N = np.array(0, dtype=np.int32)

comm.Bcast([N, 1, MPI.INT], root=0)

if rank == 0:
    ave, res = divmod(M, numprocs - 1)
    rcounts = np.empty(numprocs, dtype=np.int32)
    displs = np.empty(numprocs, dtype=np.int32)

    rcounts[0] = 0
    displs[0] = 0

    for k in range(1, numprocs):
        if k <= res:
            rcounts[k] = ave + 1
        else:
            rcounts[k] = ave
        displs[k] = displs[k - 1] + rcounts[k - 1]
else: 
    rcounts = None
    displs = None

M_part = np.array(0, dtype=np.int32)

comm.Scatterv([rcounts, np.ones(numprocs, dtype=np.int32), np.array(range(numprocs)), MPI.INT],
              [M_part, 1, MPI.INT], root=0)

A_part = np.empty((int(M_part), int(N)), dtype=np.float64)
if rank == 0:
    f2 = open(aDataPath, 'r')
    A = np.empty((int(M), int(N)), dtype=np.float64)
    for j in range(int(M)):
        for i in range(int(N)):
            A[j, i] = np.float64(f2.readline())
    f2.close()
    # В Scatterv указываем размеры в элементах (не в байтах) — rcounts*N и displs*N
    comm.Scatterv([A, rcounts * int(N), displs * int(N), MPI.DOUBLE],
                  [A_part, int(M_part) * int(N), MPI.DOUBLE], root=0)
else:
    comm.Scatterv([None, None, None, None],
                  [A_part, int(M_part) * int(N), MPI.DOUBLE], root=0)

x = np.empty(int(N), dtype=np.float64)
if rank == 0:
    f3 = open(bDataPath, 'r')
    for i in range(int(N)):
        x[i] = np.float64(f3.readline())
    f3.close()

comm.Bcast([x, int(N), MPI.DOUBLE], root=0)

A_part_gpu = cp.asarray(A_part)
x_gpu = cp.asarray(x)

b_part_gpu = cp.dot(A_part_gpu, x_gpu)

b_part = cp.asnumpy(b_part_gpu).astype(np.float64)

if rank == 0:
    b = np.empty(int(M), dtype=np.float64)
else:
    b = None

comm.Gatherv([b_part, int(M_part), MPI.DOUBLE], [b, rcounts, displs, MPI.DOUBLE], root=0)

end_time = time.time()
elapsed = end_time - start_time

if rank == 0:
    print('N: ' + str(int(N)))
    print('M: ' + str(int(M)))
    print('elapsed: ' + str(elapsed))
