
from mpi4py import MPI
from numpy import array, empty, int32

comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
  f1 = open('in.dat', 'r')
  N = array(int32(f1.readline()))
  M = array(int32(f1.readline()))
  f1.close()
else:
  N = array(0, dtype=int32); M = array(0, dtype=int32)

print('Variable N on process {0} is: {1}'.format(rank, N))