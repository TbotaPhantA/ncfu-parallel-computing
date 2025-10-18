from mpi4py import MPI
from numpy import empty, array, zeros, int32, float64, arange, dot, sqrt, hstack
from matplotlib.pyplot import style, figure, axes, show

comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()

prefixPath = '../data/data_B/'
inPath = prefixPath + 'in.dat'
aDataPath = prefixPath + 'AData.dat'
bDataPath = prefixPath + 'bData.dat'

num_row = num_col = int32(sqrt(numprocs))

comm_cart = comm.Create_cart(dims=(num_row, num_col), 
                             periods=(True, True), reorder=True)

rank_cart = comm_cart.Get_rank()

def conjugate_gradient_method(A_part, b_part, x_part, N_part, M_part, 
                              N, comm_cart, num_row, num_col) :
    
    neighbour_up, neighbour_down = comm_cart.Shift(direction=0, disp=1)
    neighbour_left, neighbour_right = comm_cart.Shift(direction=1, disp=1)
    
    Ax_part_temp = empty(M_part, dtype=float64)
    Ax_part_temp_recv = empty(M_part, dtype=float64)
    
    r_part_temp = empty(N_part, dtype=float64)
    r_part_temp_recv = empty(N_part, dtype=float64)
    
    ScalP_temp = empty(1, dtype=float64)
    ScalP_temp_recv = empty(1, dtype=float64)

    requests = [MPI.Request() for i in range(6)]
    
    requests[0] = comm_cart.Send_init([Ax_part_temp, M_part, MPI.DOUBLE], 
                                      dest=neighbour_right, tag=0)
    requests[1] = comm_cart.Recv_init([Ax_part_temp_recv, M_part, MPI.DOUBLE], 
                                      source=neighbour_left, tag=MPI.ANY_TAG)
    requests[2] = comm_cart.Send_init([r_part_temp, N_part, MPI.DOUBLE], 
                                      dest=neighbour_down, tag=0)
    requests[3] = comm_cart.Recv_init([r_part_temp_recv, N_part, MPI.DOUBLE], 
                                      source=neighbour_up, tag=MPI.ANY_TAG)
    requests[4] = comm_cart.Send_init([ScalP_temp, 1, MPI.DOUBLE], 
                                      dest=neighbour_right, tag=0)
    requests[5] = comm_cart.Recv_init([ScalP_temp_recv, 1, MPI.DOUBLE], 
                                      source=neighbour_left, tag=MPI.ANY_TAG)
    
    s = 1
    
    p_part = zeros(N_part, dtype=float64)

    while s <= N :

        if s == 1 :
            Ax_part_temp[:] = dot(A_part, x_part)
            Ax_part = Ax_part_temp.copy()
            for n in range(num_col-1) :
                MPI.Prequest.Startall([requests[0], requests[1]])
                MPI.Request.Waitall([requests[0], requests[1]], statuses=None)
                Ax_part_temp[:] = Ax_part_temp_recv
                Ax_part = Ax_part + Ax_part_temp
            b_part = Ax_part - b_part  
            r_part_temp[:] = dot(A_part.T, b_part)
            r_part = r_part_temp.copy()
            for m in range(num_row-1) :
                MPI.Prequest.Startall([requests[2], requests[3]])
                MPI.Request.Waitall([requests[2], requests[3]], statuses=None)
                r_part_temp[:] = r_part_temp_recv
                r_part = r_part + r_part_temp
        else :
            ScalP_temp[0] = dot(p_part, q_part)
            ScalP = ScalP_temp.copy()
            for n in range(num_col-1) :
                MPI.Prequest.Startall([requests[4], requests[5]])
                MPI.Request.Waitall([requests[4], requests[5]], statuses=None)
                ScalP_temp[0] = ScalP_temp_recv
                ScalP = ScalP + ScalP_temp
            r_part = r_part - q_part/ScalP
        
        ScalP_temp[0] = dot(r_part, r_part)
        ScalP = ScalP_temp.copy()
        for n in range(num_col-1) :
            MPI.Prequest.Startall([requests[4], requests[5]])
            MPI.Request.Waitall([requests[4], requests[5]], statuses=None)
            ScalP_temp[0] = ScalP_temp_recv
            ScalP = ScalP + ScalP_temp
        p_part = p_part + r_part/ScalP
        
        Ax_part_temp[:] = dot(A_part, p_part)
        Ap_part = Ax_part_temp.copy()
        for n in range(num_col-1) :
            MPI.Prequest.Startall([requests[0], requests[1]])
            MPI.Request.Waitall([requests[0], requests[1]], statuses=None)
            Ax_part_temp[:] = Ax_part_temp_recv
            Ap_part = Ap_part + Ax_part_temp
        r_part_temp[:] = dot(A_part.T, Ap_part)
        q_part = r_part_temp.copy()
        for m in range(num_row-1) :
            MPI.Prequest.Startall([requests[2], requests[3]])
            MPI.Request.Waitall([requests[2], requests[3]], statuses=None)
            r_part_temp[:] = r_part_temp_recv
            q_part = q_part + r_part_temp
        
        ScalP_temp[0] = dot(p_part, q_part)
        ScalP = ScalP_temp.copy()
        for n in range(num_col-1) :
            MPI.Prequest.Startall([requests[4], requests[5]])
            MPI.Request.Waitall([requests[4], requests[5]], statuses=None)
            ScalP_temp[0] = ScalP_temp_recv
            ScalP = ScalP + ScalP_temp
        x_part = x_part - p_part/ScalP
        
        s = s + 1
        
    for i in range(len(requests)) :   
        MPI.Request.Free(requests[i])
    
    return x_part

if rank_cart == 0 :
    f1 = open(inPath, 'r')
    N = array(int32(f1.readline()))
    M = array(int32(f1.readline()))
    f1.close()
else :
    N = array(0, dtype=int32)

comm_cart.Bcast([N, 1, MPI.INT], root=0)

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

if rank_cart == 0 :
    rcounts_M, displs_M = auxiliary_arrays_determination(M, num_row)
    rcounts_N, displs_N = auxiliary_arrays_determination(N, num_col)
else :
    rcounts_M = None; displs_M = None
    rcounts_N = None; displs_N = None

M_part = array(0, dtype=int32); N_part = array(0, dtype=int32)

comm_col = comm_cart.Split(rank_cart % num_col, rank_cart)
comm_row = comm_cart.Split(rank_cart // num_col, rank_cart)

if rank_cart in range(num_col) :
    comm_row.Scatter([rcounts_N, 1, MPI.INT], 
                     [N_part, 1, MPI.INT], root=0) 
if rank_cart in range(0, numprocs, num_col) :
    comm_col.Scatter([rcounts_M, 1, MPI.INT], 
                     [M_part, 1, MPI.INT], root=0) 

comm_col.Bcast([N_part, 1, MPI.INT], root=0)
comm_row.Bcast([M_part, 1, MPI.INT], root=0)  

A_part = empty((M_part, N_part), dtype=float64)

group = comm_cart.Get_group()

if rank_cart == 0 :
    f2 = open(aDataPath, 'r')
    for m in range(num_row) :
        a_temp = empty(rcounts_M[m]*N, dtype=float64)
        for j in range(rcounts_M[m]) :
            for n in range(num_col) :
                for i in range(rcounts_N[n]) :
                    a_temp[rcounts_M[m]*displs_N[n] + j*rcounts_N[n] + i] = float64(f2.readline())
        if m == 0 :
            comm_row.Scatterv([a_temp, rcounts_M[m]*rcounts_N, rcounts_M[m]*displs_N, MPI.DOUBLE], 
                              [A_part, M_part*N_part, MPI.DOUBLE], root=0)
        else :
            group_temp = group.Range_incl([(0,0,1), (m*num_col,(m+1)*num_col-1,1)]) 
            comm_temp = comm_cart.Create(group_temp)
            rcounts_N_temp = hstack((array(0, dtype=int32), rcounts_N))
            displs_N_temp = hstack((array(0, dtype=int32), displs_N))
            comm_temp.Scatterv([a_temp, rcounts_M[m]*rcounts_N_temp, rcounts_M[m]*displs_N_temp, MPI.DOUBLE], 
                               [empty(0, dtype=float64), 0, MPI.DOUBLE], root=0)
            group_temp.Free(); comm_temp.Free()
    f2.close()
else :
    if rank_cart in range(num_col) :
        comm_row.Scatterv([None, None, None, None], 
                          [A_part, M_part*N_part, MPI.DOUBLE], root=0)
    for m in range(1, num_row) :
        group_temp = group.Range_incl([(0,0,1), (m*num_col,(m+1)*num_col-1,1)])
        comm_temp = comm_cart.Create(group_temp)
        if rank_cart in range(m*num_col, (m+1)*num_col) :
            comm_temp.Scatterv([None, None, None, None], 
                               [A_part, M_part*N_part, MPI.DOUBLE], root=0)
            comm_temp.Free()
        group_temp.Free()
    
if rank_cart == 0 :
    b = empty(M, dtype=float64)
    f3 = open(bDataPath, 'r')
    for j in range(M) :
        b[j] = float64(f3.readline())
    f3.close()
else :
    b = None
    
b_part = empty(M_part, dtype=float64) 
 	
if rank_cart in range(0, numprocs, num_col) :    
    comm_col.Scatterv([b, rcounts_M, displs_M, MPI.DOUBLE], 
                      [b_part, M_part, MPI.DOUBLE], root=0)
    
comm_row.Bcast([b_part, M_part, MPI.DOUBLE], root=0)

if rank_cart == 0 :
    x = zeros(N, dtype=float64)
else :
    x = None
    
x_part = empty(N_part, dtype=float64)

if rank_cart in range(num_col) :
    comm_row.Scatterv([x, rcounts_N, displs_N, MPI.DOUBLE], 
                      [x_part, N_part, MPI.DOUBLE], root=0)
    
comm_col.Bcast([x_part, N_part, MPI.DOUBLE], root=0)

if rank_cart == 0 :
    start_time = MPI.Wtime()

x_part = conjugate_gradient_method(A_part, b_part, x_part, N_part, M_part, 
                                   N, comm_cart, num_row, num_col)

if rank_cart == 0 :
    end_time = MPI.Wtime()
    
if rank_cart == 0 :
    print('N={}, M={}'.format(N,M))
    print('Number of MPI process is {}'.format(numprocs))
    print('Elapsed time is {:.4f} sec.'.format(end_time-start_time))

if rank_cart in range(num_col) :
    comm_row.Gatherv([x_part, N_part, MPI.DOUBLE], 
                     [x, rcounts_N, displs_N, MPI.DOUBLE], root=0)

# if rank_cart == 0 :
#     style.use('dark_background')
#     fig = figure()
#     ax = axes(xlim=(0, N), ylim=(-1.5, 1.5))
#     ax.set_xlabel('i'); ax.set_ylabel('x[i]')
#     ax.plot(arange(N), x, '-y', lw=3)
#     show()