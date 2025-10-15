import csv
import os
from mpi4py import MPI
from numpy import empty, int32, float64, linspace, tanh, meshgrid
from matplotlib.pyplot import style, figure, axes, show

comm = MPI.COMM_WORLD
numprocs = comm.Get_size()

comm_cart = comm.Create_cart(dims=[numprocs], periods=[False], reorder=True)
rank_cart = comm_cart.Get_rank()

def u_init(x, y) :
    u_init = 0.5*tanh(1/eps*((x - 0.5)**2 + (y - 0.5)**2 - 0.35**2)) - 0.17
    return u_init

def u_left(y, t) :
    u_left = 0.33
    return u_left

def u_right(y, t) :
    u_right = 0.33
    return u_right

def u_top(x, t) :
    u_top = 0.33
    return u_top

def u_bottom(x, t) :
    u_bottom = 0.33
    return u_bottom

if rank_cart == 0 :
    start_time = MPI.Wtime()

a = -2.; b = 2.; c = -2.; d = 2.
t_0 = 0.; T = 5.
eps = 10**(-1.0)

N_x = 50; N_y = 50; M = 500

h_x = (b - a)/N_x; x = linspace(a, b, N_x+1)
h_y = (d - c)/N_y; y = linspace(c, d, N_y+1)

tau = (T - t_0)/M; t = linspace(t_0, T, M+1)

def auxiliary_arrays_determination(M, num) : 
    ave, res = divmod(M, num)
    rcounts = empty(num, dtype=int32)
    displs = empty(num, dtype=int32)
    for k in range(0, num) : 
        if k < res :
            rcounts[k] = ave + 1
        else :
            rcounts[k] = ave
        if k == 0 :
            displs[k] = 0
        else :
            displs[k] = displs[k-1] + rcounts[k-1]   
    return rcounts, displs

rcounts_N_x, displs_N_x = auxiliary_arrays_determination(N_x + 1, numprocs)
    
N_x_part = rcounts_N_x[rank_cart]

if rank_cart in [0, numprocs - 1] :
    N_x_part_aux = N_x_part + 1
else :
    N_x_part_aux = N_x_part + 2

displs_N_x_aux = displs_N_x - 1; displs_N_x_aux[0] = 0

displ_x_aux = displs_N_x_aux[rank_cart]
        
u_part_aux = empty((M + 1, N_x_part_aux, N_y + 1), dtype=float64)

for i in range(N_x_part_aux) :
    for j in range(N_y + 1) :
        u_part_aux[0, i, j] = u_init(x[displ_x_aux + i], y[j])
        
for m in range(1, M + 1) :
    for j in range(1, N_y) :
        if rank_cart == 0 :
            u_part_aux[m, 0, j] = u_left(y[j], t[m])
        if rank_cart == numprocs-1 :
            u_part_aux[m, N_x_part_aux - 1, j] = u_right(y[j], t[m])
    for i in range(N_x_part_aux) :
        u_part_aux[m, i, N_y] = u_top(x[displ_x_aux + i], t[m])
        u_part_aux[m, i, 0] = u_bottom(x[displ_x_aux + i], t[m])
    
for m in range(M) :
    
    for i in range(1, N_x_part_aux - 1) :
        for j in range(1, N_y) :
            u_part_aux[m+1, i, j] =  u_part_aux[m,i,j] + \
                tau*(eps*((u_part_aux[m,i+1,j] - 2*u_part_aux[m,i,j] + u_part_aux[m,i-1,j])/h_x**2 +
                          (u_part_aux[m,i,j+1] - 2*u_part_aux[m,i,j] + u_part_aux[m,i,j-1])/h_y**2) +
                     u_part_aux[m,i,j]*((u_part_aux[m,i+1,j] - u_part_aux[m,i-1,j])/(2*h_x) +
                               (u_part_aux[m,i,j+1] - u_part_aux[m,i,j-1])/(2*h_y)) + 
                     u_part_aux[m,i,j]**3)
                
    if rank_cart > 0 :
        comm_cart.Sendrecv(sendbuf=[u_part_aux[m+1, 1, :], N_y+1, MPI.DOUBLE], 
                           dest=rank_cart-1, sendtag=0, 
                           recvbuf=[u_part_aux[m+1, 0, :], N_y+1, MPI.DOUBLE], 
                           source=rank_cart-1, recvtag=MPI.ANY_TAG, status=None)
        
    if rank_cart < numprocs - 1:
        comm_cart.Sendrecv(sendbuf=[u_part_aux[m+1, N_x_part_aux-2, :], N_y+1, MPI.DOUBLE], 
                           dest=rank_cart+1, sendtag=0, 
                           recvbuf=[u_part_aux[m+1, N_x_part_aux-1, :], N_y+1, MPI.DOUBLE], 
                           source=rank_cart+1, recvtag=MPI.ANY_TAG, status=None)

if rank_cart == 0 :
    end_time = MPI.Wtime()
        
if rank_cart == 0 :
    u_T = empty((N_x + 1, N_y + 1), dtype=float64)
else : 
    u_T = None

if rank_cart == 0 :
    comm_cart.Gatherv([u_part_aux[M, 0:N_x_part_aux-1, :], N_x_part*(N_y+1), MPI.DOUBLE], 
                      [u_T, rcounts_N_x*(N_y+1), displs_N_x*(N_y+1), MPI.DOUBLE], root=0)
if rank_cart in range(1, numprocs-1) :
    comm_cart.Gatherv([u_part_aux[M, 1:N_x_part_aux-1, :], N_x_part*(N_y+1), MPI.DOUBLE], 
                      [None, None, None, None], root=0)
if rank_cart == numprocs-1 :
    comm_cart.Gatherv([u_part_aux[M, 1:N_x_part_aux, :], N_x_part*(N_y+1), MPI.DOUBLE], 
                      [None, None, None, None], root=0)
    
if rank_cart == 0 :
    csv_file = "lr10-2.csv"
    need_header = not os.path.exists(csv_file)
    elapsed = end_time - start_time
    t1 = 2.5 # seconds
    s_n = t1 / elapsed
    e_n = s_n / numprocs
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        if need_header:
            writer.writerow(["numprocs", "N_x", "N_y", "M", "time", "speedup", "efficiency"])
        writer.writerow([numprocs, N_x, N_y, M, elapsed, s_n, e_n])
    print(f"nprocs={numprocs}, time={elapsed:.6f} s (written to {csv_file})")
    
    # style.use('dark_background')
    # fig = figure()
    # ax = axes(xlim=(a,b), ylim=(c, d))
    # ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_aspect('equal') 
    # X, Y = meshgrid(x, y)
    # ax.pcolor(X, Y, u_T, shading='auto')
    # show()