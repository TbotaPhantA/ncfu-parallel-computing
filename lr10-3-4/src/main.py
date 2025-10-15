import csv
import os
from mpi4py import MPI
from numpy import empty, int32, float64, linspace, tanh, meshgrid, sqrt
from matplotlib.pyplot import style, figure, axes, show

comm = MPI.COMM_WORLD
numprocs = comm.Get_size()

num_row = num_col = int32(sqrt(numprocs))

comm_cart = comm.Create_cart(dims=(num_row, num_col), periods=(False, False), reorder=True)
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

rcounts_N_x, displs_N_x = auxiliary_arrays_determination(N_x + 1, num_col)
rcounts_N_y, displs_N_y = auxiliary_arrays_determination(N_y + 1, num_row)

my_row, my_col = comm_cart.Get_coords(rank_cart)  
    
N_x_part = rcounts_N_x[my_col]
N_y_part = rcounts_N_y[my_row]
    
if my_col in [0, num_col - 1] :
    N_x_part_aux = N_x_part + 1
else :
    N_x_part_aux = N_x_part + 2
    
if my_row in [0, num_row - 1] :
    N_y_part_aux = N_y_part + 1
else :
    N_y_part_aux = N_y_part + 2
    
displs_N_x_aux = displs_N_x - 1; displs_N_x_aux[0] = 0
displs_N_y_aux = displs_N_y - 1; displs_N_y_aux[0] = 0

displ_x_aux = displs_N_x_aux[my_col]
displ_y_aux = displs_N_y_aux[my_row]

u_part_aux = empty((M + 1, N_x_part_aux, N_y_part_aux), dtype=float64)

for i in range(N_x_part_aux) :
    for j in range(N_y_part_aux) :
        u_part_aux[0, i, j] = u_init(x[displ_x_aux + i], y[displ_y_aux + j])  
        
for m in range(1, M + 1) :
    for j in range(1, N_y_part_aux - 1) :
        if my_col == 0 :
            u_part_aux[m, 0, j] = u_left(y[displ_y_aux + j], t[m])
        if my_col == num_col - 1 :
            u_part_aux[m, N_x_part_aux - 1, j] = u_right(y[displ_y_aux + j], t[m])
    for i in range(N_x_part_aux) :
        if my_row == 0 :
            u_part_aux[m, i, 0] = u_bottom(x[displ_x_aux + i], t[m])
        if my_row == num_row - 1 :
            u_part_aux[m, i, N_y_part_aux - 1] = u_top(x[displ_x_aux + i], t[m])
        
for m in range(M) :
    
    for i in range(1, N_x_part_aux - 1) :
        for j in range(1, N_y_part_aux - 1) :
            u_part_aux[m+1, i, j] =  u_part_aux[m,i,j] + \
                tau*(eps*((u_part_aux[m,i+1,j] - 2*u_part_aux[m,i,j] + u_part_aux[m,i-1,j])/h_x**2 +
                          (u_part_aux[m,i,j+1] - 2*u_part_aux[m,i,j] + u_part_aux[m,i,j-1])/h_y**2) +
                      u_part_aux[m,i,j]*((u_part_aux[m,i+1,j] - u_part_aux[m,i-1,j])/(2*h_x) +
                                (u_part_aux[m,i,j+1] - u_part_aux[m,i,j-1])/(2*h_y)) + 
                      u_part_aux[m,i,j]**3)
                
    if my_col > 0 : 
        comm_cart.Sendrecv(sendbuf=[u_part_aux[m+1, 1, 1:], N_y_part, MPI.DOUBLE], 
                           dest=my_row*num_col + (my_col-1), sendtag=0, 
                           recvbuf=[u_part_aux[m+1, 0, 1:], N_y_part, MPI.DOUBLE], 
                           source=my_row*num_col + (my_col-1), recvtag=MPI.ANY_TAG, status=None)
        
    if my_col < num_col-1 :
        comm_cart.Sendrecv(sendbuf=[u_part_aux[m+1, N_x_part_aux-2, 1:], N_y_part, MPI.DOUBLE], 
                           dest=my_row*num_col + (my_col+1), sendtag=0, 
                           recvbuf=[u_part_aux[m+1, N_x_part_aux-1, 1:], N_y_part, MPI.DOUBLE], 
                           source=my_row*num_col + (my_col+1), recvtag=MPI.ANY_TAG, status=None)
        
    if my_row > 0 :     
        temp_array_send = u_part_aux[m+1, 1:N_x_part+1, 1].copy()
        temp_array_recv = empty(N_x_part, dtype=float64)
        comm_cart.Sendrecv(sendbuf=[temp_array_send, N_x_part, MPI.DOUBLE], 
                           dest=(my_row-1)*num_col + my_col, sendtag=0, 
                           recvbuf=[temp_array_recv, N_x_part, MPI.DOUBLE], 
                           source=(my_row-1)*num_col + my_col, recvtag=MPI.ANY_TAG, status=None)
        u_part_aux[m+1, 1:N_x_part+1, 0] = temp_array_recv
        
    if my_row < num_row-1 :    
        temp_array_send = u_part_aux[m+1, 1:N_x_part+1, N_y_part_aux-2].copy()
        temp_array_recv = empty(N_x_part, dtype=float64)
        comm_cart.Sendrecv(sendbuf=[temp_array_send, N_x_part, MPI.DOUBLE], 
                           dest=(my_row+1)*num_col + my_col, sendtag=0, 
                           recvbuf=[temp_array_recv, N_x_part, MPI.DOUBLE], 
                           source=(my_row+1)*num_col + my_col, recvtag=MPI.ANY_TAG, status=None)
        u_part_aux[m+1, 1:N_x_part+1, N_y_part_aux-1] = temp_array_recv
        
if rank_cart == 0 :
    end_time = MPI.Wtime()

if rank_cart == 0 :
    u_T = empty((N_x + 1, N_y + 1), dtype=float64)
else : 
    u_T = None

if rank_cart == 0 :
    for m in range(num_row) :
        for n in range(num_col) :
            if m == 0 and n == 0 :
                for i in range(N_x_part) :
                    u_T[i, 0:N_y_part] = u_part_aux[M, i, 0:N_y_part]
            else :
                for i in range(rcounts_N_x[n]) :
                    comm_cart.Recv([u_T[displs_N_x[n] + i, displs_N_y[m]:], rcounts_N_y[m], MPI.DOUBLE], 
                                   source=(m*num_col + n), tag=0, status=None)
else :
    for i in range(N_x_part) :
        if my_row == 0 :
            if my_col == 0 :
                comm_cart.Send([u_part_aux[M, i, 0:], N_y_part, MPI.DOUBLE], dest=0, tag=0)
            if my_col in range(1, num_col) :
                comm_cart.Send([u_part_aux[M, 1+i, 0:], N_y_part, MPI.DOUBLE], dest=0, tag=0)
        if my_row in range(1, num_row) :
            if my_col == 0 :
                comm_cart.Send([u_part_aux[M, i, 1:], N_y_part, MPI.DOUBLE], dest=0, tag=0)
            if my_col in range(1, num_col) :
                comm_cart.Send([u_part_aux[M, 1+i, 1:], N_y_part, MPI.DOUBLE], dest=0, tag=0)

if rank_cart == 0 :
    csv_file = "lr10-3-4.csv"
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