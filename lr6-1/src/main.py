from mpi4py import MPI
from numpy import array, int32

comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()

num_row = 2
num_col = 4

comm_cart = comm.Create_cart(dims=(num_row, num_col), periods=(True, True), reorder=True)

rank_cart = comm_cart.Get_rank()

neighbour_up, neighbour_down = comm_cart.Shift(direction=0, disp=1)
neighbour_left, neighbour_right = comm_cart.Shift(direction=1, disp=1)

a = array([(rank_cart % num_col + 1 + i) * 2 ** (rank_cart // num_col) for i in range(2)], dtype=int32)

sum = a.copy()
for n in range(num_col - 1):
    comm_cart.Sendrecv_replace([a, 2, MPI.INT], dest=neighbour_right, sendtag=0, source=neighbour_left, recvtag=MPI.ANY_TAG, status=None)

    sum = sum + a

print('Process {} has sum={}'.format(rank_cart, sum))