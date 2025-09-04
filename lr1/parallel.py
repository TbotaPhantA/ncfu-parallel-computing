from mpi4py import MPI
from numpy import empty , array , int32 , float64 , dot

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numprocs = comm.Get_size()

# считывание N и M
if rank == 0:
  f1 = open('in.dat', 'r')
  N = array(int32(f1.readline()))
  M = array(int32(f1.readline()))
  f1.close()
else:
  N = array(0, dtype=int32)
  M = array(0, dtype=int32)

comm.Bcast([N, 1, MPI.INT], root=0)
comm.Bcast([M, 1, MPI.INT], root=0)

# считывание матрицы
if rank == 0:
  f2 = open('AData.dat', 'r')
  for k in range(1, numprocs):
    A_part = empty((M//(numprocs -1), N), dtype=float64)
    for j in range(M//(numprocs -1)):
      for i in range(N):
        A_part[j, i] = float64(f2.readline())
    comm.Send([A_part , M//(numprocs -1)*N, MPI.DOUBLE], dest=k, tag=0)
  f2.close()
else:
 A_part = empty((M//(numprocs -1), N), dtype=float64)
 comm.Recv([A_part , M//(numprocs -1)*N, MPI.DOUBLE], source=0, tag=0, status=None)

# считывание вектора
x = empty(N, dtype=float64)
if rank == 0:
  f3 = open('xData.dat', 'r')
  for i in range(N):
    x[i] = float64(f3.readline())
  f3.close()

comm.Bcast([x, N, MPI.DOUBLE], root=0)

# вычилсение
b_part = empty(M//(numprocs -1), dtype=float64)
if rank != 0:
  b_part = dot(A_part, x)

status = MPI.Status()

# cборка и вывод результата
if rank == 0:
  b = empty(M, dtype=float64)
  for k in range(1, numprocs):
    comm.Recv([b_part, M//(numprocs - 1), MPI.DOUBLE], source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
    for j in range(M//(numprocs-1)):
      b[(status.Get_source()-1)*M//(numprocs - 1) + j] = b_part[j]

else:
  comm.Send([b_part, M//(numprocs-1), MPI.DOUBLE], dest=0, tag=0)

if (rank == 0):
  f4 = open('Results_base_parallel.dat', 'w')
  for j in range(M):
    print(b[j], file=f4)
  f4.close()
