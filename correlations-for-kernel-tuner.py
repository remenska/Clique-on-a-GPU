#!/usr/bin/env python
"""This is the minimal example from the README"""

import numpy as np
from kernel_tuner import tune_kernel

kernel_string = """
#ifndef block_size_x
    #define block_size_x 2
#endif
#ifndef block_size_y
    #define block_size_y 32
#endif

__global__ void quadratic_difference(bool *correlations, int N, int sliding_window_width, float *x, float *y, float *z, float *ct)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int l = i + j + 1;

    __shared__ float base_hits[4][block_size_x];

    if (i >= N || j >= sliding_window_width) return;

    int pos = i * sliding_window_width + j;

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
"""

N                    =    30000
N_light_crossing     =    1500
sliding_window_width =    N_light_crossing

problem_size = (N, sliding_window_width)


x = np.random.random(N).astype(np.float32)
y = np.random.random(N).astype(np.float32)
z = np.random.random(N).astype(np.float32)
ct = np.random.random(N).astype(np.float32)

correlations = np.zeros((N, sliding_window_width), 'b')

args = [correlations, np.int32(N), np.int32(sliding_window_width), x, y, z, ct]

tune_params = dict()
tune_params["block_size_x"] = [2**i for i in range(11)]
tune_params["block_size_y"] = [2**i for i in range(11)]

grid_div_x = ["block_size_x"]
grid_div_y = ["block_size_y"]

restrict = ["block_size_x * block_size_y <= 1024"]

tune_kernel("quadratic_difference", kernel_string, problem_size, args, tune_params, grid_div_x=grid_div_x, grid_div_y=grid_div_y, restrictions=restrict, verbose = True) 
