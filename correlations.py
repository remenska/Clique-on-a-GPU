#!/usr/bin/env python

import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
import time
from numba import jit

from pycuda.compiler import SourceModule
import pycuda.driver

mod = SourceModule("""
#include <inttypes.h>    
#ifndef block_size_x
    #define block_size_x 8
#endif
#ifndef block_size_y
    #define block_size_y 16
#endif

__global__ void quadratic_difference(bool *correlations, int N, int sliding_window_width, float *x, float *y, float *z, float *ct)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int l = i + j + 1;

    __shared__ float base_hits[4][block_size_x];

    if (i >= N || j >= sliding_window_width) return;

    const unsigned long pos = j * (uint64_t)N + (uint64_t)i;

    if (l >= N){
      return;
    }

    if (threadIdx.y == 0 && i < N){
      base_hits[0][threadIdx.x] = x[i];
      base_hits[1][threadIdx.x] = y[i];
      base_hits[2][threadIdx.x] = z[i];
      base_hits[3][threadIdx.x] = ct[i];
    }

    __shared__ float surrounding_hits[4][block_size_x + block_size_y - 1];

    if (threadIdx.x == 0 && l < N){
      surrounding_hits[0][threadIdx.y] = x[l];
      surrounding_hits[1][threadIdx.y] = y[l];
      surrounding_hits[2][threadIdx.y] = z[l];
      surrounding_hits[3][threadIdx.y] = ct[l];
    }

    if (threadIdx.x == block_size_x - 1 && l < N){
      surrounding_hits[0][threadIdx.x + threadIdx.y] = x[l];
      surrounding_hits[1][threadIdx.x + threadIdx.y] = y[l];
      surrounding_hits[2][threadIdx.x + threadIdx.y] = z[l];
      surrounding_hits[3][threadIdx.x + threadIdx.y] = ct[l];
    }

    __syncthreads();

    if (i < N && j < sliding_window_width && l < N){
      float diffx  = base_hits[0][threadIdx.x] - surrounding_hits[0][threadIdx.x + threadIdx.y];
      float diffy  = base_hits[1][threadIdx.x] - surrounding_hits[1][threadIdx.x + threadIdx.y];
      float diffz  = base_hits[2][threadIdx.x] - surrounding_hits[2][threadIdx.x + threadIdx.y];
      float diffct = base_hits[3][threadIdx.x] - surrounding_hits[3][threadIdx.x + threadIdx.y];

      if (diffct * diffct < diffx * diffx + diffy * diffy + diffz * diffz){ 
        correlations[pos] = 1;
      }
    }



}
""")

quadratic_difference= mod.get_function("quadratic_difference")

N = np.int32(4.5e6)

# try:
#     x = np.load("x.npy")
#     y = np.load("y.npy")
#     z = np.load("z.npy")
#     ct = np.load("ct.npy")

#     assert x.size == N

# except (FileNotFoundError, AssertionError):
x = np.random.normal(0.2, 0.1, N).astype(np.float32)
y = np.random.normal(0.2, 0.1, N).astype(np.float32)
z = np.random.normal(0.2, 0.1, N).astype(np.float32)
ct = 1000*np.random.normal(0.5, 0.06, N).astype(np.float32)

#np.save("x.npy", x)
#np.save("y.npy", y)
#np.save("z.npy", z)
#np.save("ct.npy", ct)

start_malloc = time.time()

pycuda.driver.start_profiler()
x_gpu = drv.mem_alloc(x.nbytes)
y_gpu = drv.mem_alloc(y.nbytes)
z_gpu = drv.mem_alloc(z.nbytes)
ct_gpu = drv.mem_alloc(ct.nbytes)

end_malloc = time.time()

print()
print('Memory allocation on device took {0:.2e}s.'.format(end_malloc -start_malloc))

start_transfer = time.time()

drv.memcpy_htod(x_gpu, x)
drv.memcpy_htod(y_gpu, y)
drv.memcpy_htod(z_gpu, z)
drv.memcpy_htod(ct_gpu, ct)

end_transfer = time.time()

print()
print('Data transfer from host to device took {0:.2e}s.'.format(end_transfer -start_transfer))

# The number of consecutive hits corresponding to the light crossing time of the detector (1km/c).
N_light_crossing     = 1500
# This used to be 2 * N_light_crossing, but caused redundant calculations.
sliding_window_width = np.int32(N_light_crossing)
# problem_size = N * sliding_window_width

correlations = np.zeros((sliding_window_width, N), 'b')
print()
print("Number of bytes needed for the correlation matrix = {0:.3e} ".format(correlations.nbytes))
correlations_gpu = drv.mem_alloc(correlations.nbytes)
drv.memcpy_htod(correlations_gpu, correlations)
# block_size_x = int(np.sqrt(block_size))
block_size_x = 8
# block_size_y = int(np.sqrt(block_size))
block_size_y = 16

# block_size = block_size_x * block_size_y

gridx = int(np.ceil(correlations.shape[0]/block_size_x))
gridy = int(np.ceil(correlations.shape[1]/block_size_y))

# create two timers so we can speed-test each approach
start = drv.Event()
end = drv.Event()

pycuda.autoinit.context.synchronize()

start.record() # start timing
quadratic_difference(
        correlations_gpu, np.int32(correlations.shape[0]), np.int32(correlations.shape[1]), x_gpu, y_gpu, z_gpu, ct_gpu, 
        block=(block_size_x, block_size_y, 1), grid=(gridx, gridy))

pycuda.autoinit.context.synchronize()

end.record() # end timing
# calculate the run length
end.synchronize()

secs = start.time_till(end)*1e-3

print()
print('Time taken for GPU computations is {0:.2e}s.'.format(secs))

start_transfer = time.time()

drv.memcpy_dtoh(correlations, correlations_gpu)

pycuda.driver.stop_profiler()
end_transfer = time.time()

print()
print('Data transfer from device to host took {0:.2e}s.'.format(end_transfer -start_transfer))

print()
print('correlations = ', correlations)
#np.save("correlations.npy", correlations)
# Speed up the CPU processing.
@jit
def correlations_cpu(check, x, y, z, ct):
    for i in range(check.shape[0]):
        for j in range(i + 1, i + check.shape[1] + 1):
            if j < check.shape[0]:
                if (ct[i]-ct[j])**2 < (x[i]-x[j])**2  + (y[i] - y[j])**2 + (z[i] - z[j])**2:
                   check[i, j - i - 1] = 1
    return check

# try:
#     check = np.load("check.npy")

    # assert N == check.shape[0] and sliding_window_width == check.shape[1]

# except (FileNotFoundError, AssertionError):
start_cpu_computations = time.time()   

check = np.zeros_like(correlations)
# Checkif output is correct.
check = correlations_cpu(check, x, y, z, ct)

end_cpu_computations = time.time()   
print()
print('Time taken for cpu computations is {0:.2e}s.'.format(end_cpu_computations - start_cpu_computations)) 

#np.save("check.npy", check)

print()
print()
print('check = ', check)
print()
print('check.max() = {0}'.format(check.max()))
print()

sum_abs = np.sum(np.abs(check - correlations)) 
print('This should be close to zero: {0}'.format(sum_abs))
print()

if sum_abs > 0: 
    print('Index or indices where the difference is nonzero: ', (check-correlations).nonzero())
    print()
    print('check - correlations = ', check - correlations)
print("Percentage hits = {0} %".format(100 * np.sum(correlations / (correlations.shape[0] * correlations.shape[1]))))
