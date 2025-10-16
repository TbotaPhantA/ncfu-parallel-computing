import csv
import os
from mpi4py import MPI
from numpy import empty, array, zeros, int32, float64, hstack, dot
from threadpoolctl import threadpool_limits

comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()

# используем MPI.Compute_dims вместо sqrt, чтобы получить корректные размеры сетки
dims = MPI.Compute_dims(int(numprocs), [0, 0])
num_row = int(dims[0])
num_col = int(dims[1])

# защитная проверка
if num_row * num_col != numprocs:
    if rank == 0:
        print(f"Cannot create cartesian grid: {num_row}x{num_col} != {numprocs}")
    MPI.Finalize()
    raise SystemExit(1)

comm_cart = comm.Create_cart(dims=(num_row, num_col),
                             periods=(True, True), reorder=True)

# если create_cart по каким-то причинам вернул COMM_NULL — выйти
if comm_cart == MPI.COMM_NULL:
    if rank == 0:
        print("comm_cart is MPI.COMM_NULL, exiting.")
    MPI.Finalize()
    raise SystemExit(1)

rank_cart = comm_cart.Get_rank()

prefixPath = '../../lr4/src/data/datasets/data_C/'
inPath = prefixPath + 'in.dat'
aDataPath = prefixPath + 'AData.dat'
bDataPath = prefixPath + 'bData.dat'

with threadpool_limits(limits=1):
    if (rank == 0):
        t0 = MPI.Wtime()
    else:
        t0 = None

    def conjugate_gradient_method(A_part, b_part, x_part, N_part_arr, M_part_arr,
                                  N_arr, comm_cart, num_row, num_col):
        # N_part_arr, M_part_arr, N_arr are numpy scalars (int32) — конвертируем в Python int
        N_part = int(N_part_arr)
        M_part = int(M_part_arr)
        N = int(N_arr)

        neighbour_up, neighbour_down = comm_cart.Shift(direction=0, disp=1)
        neighbour_left, neighbour_right = comm_cart.Shift(direction=1, disp=1)

        ScalP_temp = empty(1, dtype=float64)

        s = 1

        p_part = zeros(N_part, dtype=float64)

        # helper: perform ring accumulation using persistent Send_init/Recv_init
        # send_arr: numpy array to start from (will be forwarded around ring)
        # steps: how many forwardings (num_col-1 or num_row-1)
        # dest: destination rank for send
        # source: source rank for recv
        def ring_accumulate(send_arr, steps, dest, source):
            # accum starts with local contribution
            accum = send_arr.copy()
            if steps <= 0:
                return accum, send_arr

            # prepare send and recv buffers (contiguous)
            send_local = send_arr.copy()
            recv_local = empty(send_local.shape, dtype=send_local.dtype)

            # create persistent requests on communicator
            send_req = comm_cart.Send_init(send_local, dest=dest, tag=0)
            recv_req = comm_cart.Recv_init(recv_local, source=source, tag=0)
            reqs = [send_req, recv_req]

            for _ in range(steps):
                # prepare send_local as the block to forward
                send_local[:] = send_arr
                # start requests (no Startall in some mpi4py; используем Start() у каждого)
                for r in reqs:
                    r.Start()
                # waitall
                MPI.Request.Waitall(reqs)
                # после приёма добавляем к аккумулятору
                accum = accum + recv_local
                # подготовим для следующего шага: будем отправлять то, что только получили
                send_arr = recv_local.copy()

            # освобождаем persistent-запросы
            send_req.Free()
            recv_req.Free()
            return accum, send_arr

        while s <= N:

            if s == 1:
                Ax_part_temp = dot(A_part, x_part)
                Ax_part = Ax_part_temp.copy()
                if int(num_col) - 1 > 0:
                    Ax_part, Ax_part_temp = ring_accumulate(Ax_part_temp, int(num_col) - 1,
                                                           neighbour_right, neighbour_left)
                b_part = Ax_part - b_part
                r_part_temp = dot(A_part.T, b_part)
                r_part = r_part_temp.copy()
                if int(num_row) - 1 > 0:
                    r_part, r_part_temp = ring_accumulate(r_part_temp, int(num_row) - 1,
                                                         neighbour_down, neighbour_up)
            else:
                ScalP_temp[0] = dot(p_part, q_part)
                ScalP = ScalP_temp.copy()
                if int(num_col) - 1 > 0:
                    ScalP, ScalP_temp = ring_accumulate(ScalP_temp, int(num_col) - 1,
                                                       neighbour_right, neighbour_left)
                r_part = r_part - q_part / ScalP

            ScalP_temp[0] = dot(r_part, r_part)
            ScalP = ScalP_temp.copy()
            if int(num_col) - 1 > 0:
                ScalP, ScalP_temp = ring_accumulate(ScalP_temp, int(num_col) - 1,
                                                   neighbour_right, neighbour_left)
            p_part = p_part + r_part / ScalP

            Ap_part_temp = dot(A_part, p_part)
            Ap_part = Ap_part_temp.copy()
            if int(num_col) - 1 > 0:
                Ap_part, Ap_part_temp = ring_accumulate(Ap_part_temp, int(num_col) - 1,
                                                       neighbour_right, neighbour_left)
            q_part_temp = dot(A_part.T, Ap_part)
            q_part = q_part_temp.copy()
            if int(num_row) - 1 > 0:
                q_part, q_part_temp = ring_accumulate(q_part_temp, int(num_row) - 1,
                                                     neighbour_down, neighbour_up)

            ScalP_temp[0] = dot(p_part, q_part)
            ScalP = ScalP_temp.copy()
            if int(num_col) - 1 > 0:
                ScalP, ScalP_temp = ring_accumulate(ScalP_temp, int(num_col) - 1,
                                                   neighbour_right, neighbour_left)
            x_part = x_part - p_part / ScalP

            s = s + 1

        return x_part

    if rank_cart == 0:
        f1 = open(inPath, 'r')
        N = array(int32(f1.readline()))
        M = array(int32(f1.readline()))
        f1.close()
    else:
        N = array(0, dtype=int32)

    comm_cart.Bcast([N, 1, MPI.INT], root=0)

    def auxiliary_arrays_determination(M_val, num):
        ave, res = divmod(M_val, num)
        rcounts = empty(num, dtype=int32)
        displs = empty(num, dtype=int32)
        for k in range(0, num):
            if k < res:
                rcounts[k] = ave + 1
            else:
                rcounts[k] = ave
            if k == 0:
                displs[k] = 0
            else:
                displs[k] = displs[k - 1] + rcounts[k - 1]
        return rcounts, displs

    if rank_cart == 0:
        rcounts_M, displs_M = auxiliary_arrays_determination(int(M), num_row)
        rcounts_N, displs_N = auxiliary_arrays_determination(int(N), num_col)
    else:
        rcounts_M = None; displs_M = None
        rcounts_N = None; displs_N = None

    M_part = array(0, dtype=int32); N_part = array(0, dtype=int32)

    comm_col = comm_cart.Split(rank_cart % num_col, rank_cart)
    comm_row = comm_cart.Split(rank_cart // num_col, rank_cart)

    # Scatter размеров столбцов/строк
    if rank_cart in range(num_col):
        comm_row.Scatter([rcounts_N, 1, MPI.INT],
                         [N_part, 1, MPI.INT], root=0)
    if rank_cart in range(0, numprocs, num_col):
        comm_col.Scatter([rcounts_M, 1, MPI.INT],
                         [M_part, 1, MPI.INT], root=0)

    comm_col.Bcast([N_part, 1, MPI.INT], root=0)
    comm_row.Bcast([M_part, 1, MPI.INT], root=0)

    A_part = empty((int(M_part), int(N_part)), dtype=float64)

    group = comm_cart.Get_group()

    if rank_cart == 0:
        f2 = open(aDataPath, 'r')
        for m in range(num_row):
            a_temp = empty(int(rcounts_M[m]) * int(N), dtype=float64)
            for j in range(int(rcounts_M[m])):
                for n in range(num_col):
                    for i in range(int(rcounts_N[n])):
                        a_temp[int(rcounts_M[m]) * int(displs_N[n]) + j * int(rcounts_N[n]) + i] = float64(f2.readline())
            if m == 0:
                comm_row.Scatterv([a_temp, rcounts_M[m] * rcounts_N, rcounts_M[m] * displs_N, MPI.DOUBLE],
                                  [A_part, int(M_part) * int(N_part), MPI.DOUBLE], root=0)
            else:
                group_temp = group.Range_incl([(0, 0, 1), (m * num_col, (m + 1) * num_col - 1, 1)])
                comm_temp = comm_cart.Create(group_temp)
                rcounts_N_temp = hstack((array(0, dtype=int32), rcounts_N))
                displs_N_temp = hstack((array(0, dtype=int32), displs_N))
                comm_temp.Scatterv([a_temp, rcounts_M[m] * rcounts_N_temp, rcounts_M[m] * displs_N_temp, MPI.DOUBLE],
                                   [empty(0, dtype=float64), 0, MPI.DOUBLE], root=0)
                group_temp.Free(); comm_temp.Free()
        f2.close()
    else:
        if rank_cart in range(num_col):
            comm_row.Scatterv([None, None, None, None],
                              [A_part, int(M_part) * int(N_part), MPI.DOUBLE], root=0)
        for m in range(1, num_row):
            group_temp = group.Range_incl([(0, 0, 1), (m * num_col, (m + 1) * num_col - 1, 1)])
            comm_temp = comm_cart.Create(group_temp)
            if rank_cart in range(m * num_col, (m + 1) * num_col):
                comm_temp.Scatterv([None, None, None, None],
                                   [A_part, int(M_part) * int(N_part), MPI.DOUBLE], root=0)
                comm_temp.Free()
            group_temp.Free()

    if rank_cart == 0:
        b = empty(int(M), dtype=float64)
        f3 = open(bDataPath, 'r')
        for j in range(int(M)):
            b[j] = float64(f3.readline())
        f3.close()
    else:
        b = None

    b_part = empty(int(M_part), dtype=float64)

    if rank_cart in range(0, numprocs, num_col):
        comm_col.Scatterv([b, rcounts_M, displs_M, MPI.DOUBLE],
                          [b_part, int(M_part), MPI.DOUBLE], root=0)

    comm_row.Bcast([b_part, int(M_part), MPI.DOUBLE], root=0)

    if rank_cart == 0:
        x = zeros(int(N), dtype=float64)
    else:
        x = None

    x_part = empty(int(N_part), dtype=float64)

    if rank_cart in range(num_col):
        comm_row.Scatterv([x, rcounts_N, displs_N, MPI.DOUBLE],
                          [x_part, int(N_part), MPI.DOUBLE], root=0)

    comm_col.Bcast([x_part, int(N_part), MPI.DOUBLE], root=0)

    x_part = conjugate_gradient_method(A_part, b_part, x_part, N_part, M_part,
                                       N, comm_cart, num_row, num_col)

    if rank_cart in range(num_col):
        comm_row.Gatherv([x_part, int(N_part), MPI.DOUBLE],
                         [x, rcounts_N, displs_N, MPI.DOUBLE], root=0)

    if rank == 0:
        t1 = MPI.Wtime()
        elapsed = t1 - t0
        csv_file = "lr11-3.csv"
        need_header = not os.path.exists(csv_file)
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            if need_header:
                writer.writerow(["numprocs", "N", "M", "time"])
            writer.writerow([numprocs, N, M, elapsed])
        print(f"nprocs={numprocs}, time={elapsed:.6f} s (written to {csv_file})")
