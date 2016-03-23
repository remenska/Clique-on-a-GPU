import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
import time

from pycuda.compiler import SourceModule


mod = SourceModule("""
__global__ void quadratic_difference(int *dest, int N, float *x, float *y, float *z, float *ct)
{
    // const int i = threadIdx.x;
    // const int j = threadIdx.y;
    // const int N = 1500;

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int pos = i * N + j;

    // if(i >= N || j >= N) return;

    dest[pos] = powf(x[i] - x[j], 2) + powf(y[i] - y[j], 2) + powf(z[i] - z[j], 2) - powf(ct[i] - ct[j], 2) ;
}
""")

quadratic_difference= mod.get_function("quadratic_difference")

N = 1500

x = np.random.randn(N).astype(np.float32)
y = np.random.randn(N).astype(np.float32)
z = np.random.randn(N).astype(np.float32)
ct = np.random.randn(N).astype(np.float32)

dest = np.identity(N, 'int')

block_size = 1024
block_size_x = int(np.sqrt(block_size))
block_size_y = int(np.sqrt(block_size))

# We need N threads for i and N threads for j in quadratic_difference.
problem_size = int(np.ceil(N * (N - 1)/2))

gridx = int(np.ceil(problem_size/block_size_x))
gridy = int(np.ceil(problem_size/block_size_y))

# create two timers so we can speed-test each approach
start = drv.Event()
end = drv.Event()

start.record() # start timing

quadratic_difference(
        drv.Out(dest), np.int32(N), drv.In(x), drv.In(y), drv.In(z), drv.In(ct), 
        block=(block_size_x, block_size_y, 1), grid=(gridx, gridy))

end.record() # end timing
# calculate the run length
end.synchronize()
secs = start.time_till(end)*1e-3

print()
print('Times taken is {0:.2e}s.'.format(secs))
print()
print(dest)

