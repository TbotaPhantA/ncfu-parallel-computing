import time
from mpi4py import MPI
from numpy import empty , array , int32 , float64 , dot, ones

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numprocs = comm.Get_size()

prefixPath = '../data/data_A/'
inPath = prefixPath + 'in.dat'
aDataPath = prefixPath + 'AData.dat'
bDataPath = prefixPath + 'bData.dat'

start_time = time.time()


# считывание N и M
if rank == 0:
  f1 = open(inPath, 'r')
  N = array(int32(f1.readline()))
  M = array(int32(f1.readline()))
  f1.close()
else:
  N = array(0, dtype=int32)

comm.Bcast([N, 1, MPI.INT], root=0)

# Обобщение программы на случай несогласованного числа входных данных и числа процессов, использующихся при расчётах
if rank == 0:
  ave, res = divmod(M, numprocs -1)
  rcounts = empty(numprocs , dtype=int32)
  displs = empty(numprocs , dtype=int32)
  
  rcounts[0] = 0
  displs[0] = 0
  
  for k in range(1, numprocs):
    if k <= res:
      rcounts[k] = ave+1
    else:
      rcounts[k] = ave
    displs[k] = displs[k-1] + rcounts[k-1]
else: # rank != 0
  rcounts = None
  displs = None

M_part = array(0, dtype=int32)

comm.Scatterv([rcounts, ones(numprocs, dtype=int32), array(range(numprocs)), MPI.INT], [M_part, 1, MPI.INT], root=0)

# считывание матрицы
A_part = empty((M_part, N), dtype=float64)
if rank == 0:
    f2 = open(aDataPath, 'r')
    A = empty((M, N), dtype=float64)
    for j in range(M):
        for i in range(N):
            A[j, i] = float64(f2.readline())
    f2.close()
    comm.Scatterv([A, rcounts*N, displs*N, MPI.DOUBLE],
                [A_part, M_part*N, MPI.DOUBLE], root=0)
else: # rank != 0
    comm.Scatterv([None, None, None, None],
                [A_part, M_part*N, MPI.DOUBLE], root=0)

# считывание вектора
x = empty(N, dtype=float64)
if rank == 0:
    f3 = open(bDataPath, 'r')
    for i in range(N):
        x[i] = float64(f3.readline())
    f3.close()

comm.Bcast([x, N, MPI.DOUBLE], root=0)

# вычисления
b_part = dot(A_part, x)

# сборка результата
if rank == 0:
  b = empty(M, dtype=float64)
else:
  b = None

comm.Gatherv([b_part, M_part, MPI.DOUBLE], [b, rcounts, displs, MPI.DOUBLE], root=0)

end_time = time.time()
elapsed = end_time - start_time

if rank == 0:
  print('N: ' + str(N))
  print('M: ' + str(M))
  print('elapsed: ' + str(elapsed))