from mpi4py import MPI
import threading
import os
import sys

def worker(thread_id, num_threads, rank, size, lock):
    with lock:
        print(f"MPI process {rank}/{size}, OpenMP thread {thread_id}/{num_threads}", flush=True)

def main():
    required = MPI.THREAD_FUNNELED

    if not MPI.Is_initialized():
        provided = MPI.Init_thread(required)
    else:
        provided = MPI.Query_thread()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    try:
        num_threads = int(os.environ.get("OMP_NUM_THREADS", "4"))
    except ValueError:
        num_threads = 4

    lock = threading.Lock()
    threads = []
    for tid in range(num_threads):
        t = threading.Thread(target=worker, args=(tid, num_threads, rank, size, lock))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    if not MPI.Is_finalized():
        MPI.Finalize()

if __name__ == "__main__":
    main()
