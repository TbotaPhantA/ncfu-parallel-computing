from mpi4py import MPI
import cupy as cp
import numpy as np
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print(f"MPI process {rank}/{size} initialized", flush=True)

try:
    ngpus = cp.cuda.runtime.getDeviceCount()
except Exception:
    ngpus = 1

if ngpus <= 0:
    if rank == 0:
        print("No CUDA devices found. Exiting.", file=sys.stderr)
    sys.exit(1)

device_id = rank % ngpus
cp.cuda.Device(device_id).use()
print(f"Rank {rank} using GPU device {device_id}", flush=True)

kernel_code = r'''
extern "C" __global__ void hello_cuda(int rank, int* device_data) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid == 0) {
        printf("Hello from GPU, MPI rank: %d, thread %d\n", rank, tid);
        device_data[tid] = rank * 1000 + tid;
    }
}
'''

hello_kernel = cp.RawKernel(kernel_code, 'hello_cuda')

device_data = cp.zeros(1, dtype=cp.int32)

hello_kernel((1,), (1,), (np.int32(rank), device_data))

cp.cuda.Device().synchronize()

host_data = int(device_data.get()[0])

print(f"MPI rank {rank} received from GPU: {host_data}", flush=True)

gathered = comm.gather(host_data, root=0)

if rank == 0:
    print("Collected data from all GPUs:", end=" ")
    for v in gathered:
        print(v, end=" ")
    print()