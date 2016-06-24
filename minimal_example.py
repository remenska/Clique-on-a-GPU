import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from numba import jit
import pycuda.tools
from pycuda.compiler import SourceModule
import pycuda.driver

mod = SourceModule("""
#ifndef block_size_x
    #define block_size_x 2
#endif
#ifndef block_size_y
    #define block_size_y 2
#endif

#include <stdio.h>

__global__ void quadratic_difference(bool *correlations, int N, int sliding_window_width)
{
    unsigned long i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long j = blockIdx.y * blockDim.y + threadIdx.y;

    int l = i + j + 1;

    __shared__ float base_hits[4][block_size_x];

    if (i >= N || j >= sliding_window_width) return;

    const unsigned long pos = i * sliding_window_width + j;

    if (l >= N){
      correlations[pos] = 0;
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
      else{
        correlations[pos] = 0;
      }
    }



}
""")

# pycuda.tools.make_default_context()

quadratic_difference= mod.get_function("quadratic_difference")

N = 1600000
sliding_window_width = 1500

correlations = np.zeros(N * sliding_window_width).astype(np.bool)

correlations_gpu = drv.mem_alloc(correlations.nbytes)
drv.memcpy_htod(correlations_gpu, correlations)
block_size_x = 2
block_size_y = 2

gridx = int(np.ceil(N/block_size_x))
gridy = int(np.ceil(sliding_window_width/block_size_y))

pycuda.autoinit.context.synchronize()

quadratic_difference(
        correlations_gpu, np.int32(N), np.int32(sliding_window_width),
        block=(block_size_x, block_size_y, 1), grid=(gridx, gridy))

pycuda.autoinit.context.synchronize()

#correlations_done = np.empty_like(correlations)
drv.memcpy_dtoh(correlations, correlations_gpu)
#pycuda.autoinit.context.synchronize()

print('correlations = ', correlations)

print("Number hits = {0}".format(np.sum(correlations)))
print("Number of bytes transfered back/forth = %s " % correlations.nbytes)
