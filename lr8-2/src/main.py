import csv
import os
from mpi4py import MPI
from numpy import empty, array, int32, float64, linspace, sin, pi, hstack

comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()

def u_init(x) :
    u_init = sin(3*pi*(x - 1/6))
    return u_init

def u_left(t) :
    u_left = -1.
    return u_left

def u_right(t) :
    u_right = 1.
    return u_right

if rank == 0 : start_time = MPI.Wtime()

a = 0.; b = 1.
t_0 = 0.; T = 6.0
eps = 10**(-1.5)

N = 200; M = 20000

h = (b - a)/N; x = linspace(a, b, N+1)
tau = (T - t_0)/M; t = linspace(t_0, T, M+1)

if rank == 0 :
    ave, res = divmod(N + 1, numprocs)
    rcounts = empty(numprocs, dtype=int32)
    displs = empty(numprocs, dtype=int32)
    for k in range(0, numprocs) : 
        if k < res :
            rcounts[k] = ave + 1
        else :
            rcounts[k] = ave
        if k == 0 :
            displs[k] = 0
        else :
            displs[k] = displs[k-1] + rcounts[k-1]   
else :
    rcounts = None; displs = None
    
N_part = array(0, dtype=int32)

comm.Scatter([rcounts, 1, MPI.INT], [N_part, 1, MPI.INT], root=0) 

if rank == 0 :
    rcounts_from_0 = empty(numprocs, dtype=int32)
    displs_from_0 = empty(numprocs, dtype=int32)
    rcounts_from_0[0] = rcounts[0] + 1
    displs_from_0[0] = 0
    for k in range(1, numprocs-1) :
        rcounts_from_0[k] = rcounts[k] + 2
        displs_from_0[k] = displs[k] - 1
    rcounts_from_0[numprocs-1] = rcounts[numprocs-1] + 1  
    displs_from_0[numprocs-1] = displs[numprocs-1] - 1
else :
    rcounts_from_0 = None; displs_from_0 = None
    
N_part_aux = array(0, dtype=int32)
    
comm.Scatter([rcounts_from_0, 1, MPI.INT], [N_part_aux, 1, MPI.INT], root=0) 

if rank == 0 :
    u = empty((M+1, N+1), dtype=float64)
    for n in range(N + 1) :
        u[0, n] = u_init(x[n]) 
else :
    u = empty((M+1, 0), dtype=float64)
        
u_part = empty(N_part, dtype=float64)
u_part_aux = empty(N_part_aux, dtype=float64)

for m in range(M) :
    
    comm.Scatterv([u[m], rcounts_from_0, displs_from_0, MPI.DOUBLE], 
                  [u_part_aux, N_part_aux, MPI.DOUBLE], root=0)
    
    for n in range(1, N_part_aux - 1) :
        u_part[n-1] = u_part_aux[n] + \
            eps*tau/h**2*(u_part_aux[n+1] - 2*u_part_aux[n] + u_part_aux[n-1]) + \
                tau/(2*h)*u_part_aux[n]*(u_part_aux[n+1] - u_part_aux[n-1]) + \
                    tau*u_part_aux[n]**3
                    
    if rank == 0 :
        u_part = hstack((array(u_left(t[m+1]), dtype=float64), u_part[0:N_part-1]))
    elif rank == numprocs-1 :
        u_part = hstack((u_part[0:N_part-1], array(u_right(t[m+1]), dtype=float64)))
        
    comm.Gatherv([u_part, N_part, MPI.DOUBLE], 
                 [u[m+1], rcounts, displs, MPI.DOUBLE], root=0)

if rank == 0 :
    
    end_time = MPI.Wtime()
    elapsed = end_time-start_time

    csv_file = "lr8-2.csv"
    need_header = not os.path.exists(csv_file)
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        if need_header:
            writer.writerow(["numprocs", "N", "M", "time"])
        writer.writerow([numprocs, N, M, elapsed])
    print(f"nprocs={numprocs}, time={elapsed:.6f} s (written to {csv_file})")
    
    from numpy import savez
    savez('results_of_calculations', x=x, u=u)