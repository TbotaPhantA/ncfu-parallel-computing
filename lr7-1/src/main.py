import csv
import os
from numpy import array, empty, float64, zeros, random
from threadpoolctl import threadpool_limits
from mpi4py import MPI

with threadpool_limits(limits=1):
    t0 = MPI.Wtime()

    def consecutive_tridiagonal_matrix_algorithm(a, b, c, d):
        N = len(d)
        
        x = empty(N, dtype=float64)

        for n in range(1, N) :
            coef = a[n]/b[n-1]
            b[n] = b[n] - coef*c[n-1]
            d[n] = d[n] - coef*d[n-1]
            
        for n in range(N-2, -1, -1) :
            coef = c[n]/b[n+1]
            d[n] = d[n] - coef*d[n+1]
            
        for n in range(N) :
            x[n] = d[n]/b[n]
        
        return x

    def data_prep(N) :
        a = empty(N, dtype=float64)
        b = empty(N, dtype=float64)
        c = empty(N, dtype=float64)
        d = empty(N, dtype=float64)
        for n in range(N) :
            a[n] = random.random_sample(1)
            b[n] = random.random_sample(1)
            c[n] = random.random_sample(1)
            d[n] = random.random_sample(1)
        return a, b, c, d

    N = 1000000

    a, b, c, d = data_prep(N)

    x = consecutive_tridiagonal_matrix_algorithm(a, b, c, d)

    t1 = MPI.Wtime()
    elapsed = t1 - t0
    csv_file = "lr7-1.csv"
    need_header = not os.path.exists(csv_file)
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        if need_header:
            writer.writerow(["numprocs", "N", "time"])
        writer.writerow([1, N, elapsed])
    print(f"nprocs={1}, time={elapsed:.6f} s (written to {csv_file})")

