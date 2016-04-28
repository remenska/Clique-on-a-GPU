import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
import time

from pycuda.compiler import SourceModule


mod = SourceModule("""
__global__ void quadratic_difference(bool *correlations, int N, int N_lightcrossing, int sliding_window_width, float *x, float *y, float *z, float *ct)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    //  We want j to iterate over values near i, from i - N_light_crossing to i + N_light_crossing.
    //  blockIdx.y * blockDim.y + threadIdx.y should take values from 0 to and possibly including 2 * N_lightcrossing.
    int j = i + blockIdx.y * blockDim.y + threadIdx.y - N_lightcrossing;

    if (i >= N || j < 0 || j >= N) return;

    unsigned int pos1 = i * sliding_window_width + j - i;
    // unsigned int pos2 = j * sliding_window_width + i;

    if (j==i){
      correlations[pos1] = 1;
      // correlations[pos2] = 1;
      return;
    }

    float diffct = ct[i] - ct[j];
    float diffx  = x[i] - x[j];
    float diffy  = y[i] - y[j];
    float diffz  = z[i] - z[j];


    if (diffct * diffct < diffx * diffx + diffy * diffy + diffz * diffz){ 
      correlations[pos1] = 1;
      // correlations[pos2] = 1;
    }
    else{
      correlations[pos1] = 0;
      // correlations[pos2] = 0;
   }

}
""")

quadratic_difference= mod.get_function("quadratic_difference")

N = 30000

x = np.random.random(N).astype(np.float32)
y = np.random.random(N).astype(np.float32)
z = np.random.random(N).astype(np.float32)
ct = np.random.random(N).astype(np.float32)

start_transfer = time.time()

x_gpu = drv.mem_alloc(x.nbytes)
y_gpu = drv.mem_alloc(y.nbytes)
z_gpu = drv.mem_alloc(z.nbytes)
ct_gpu = drv.mem_alloc(ct.nbytes)

drv.memcpy_htod(x_gpu, x)
drv.memcpy_htod(y_gpu, y)
drv.memcpy_htod(z_gpu, z)
drv.memcpy_htod(ct_gpu, ct)

end_transfer = time.time()

print()
print('Data transfer from host to device plus memory allocation on device took {0:.2e}s.'.format(end_transfer -start_transfer))

# The number of consecutive hits corresponding to the light crossing time of the detector (1km/c).
N_light_crossing = 1500
# We have a sliding window with size 2 * N_light_crossing to consider for correlations.
sliding_window_width = 2 * N_light_crossing
# problem_size = N * sliding_window_width

correlations = np.empty((N, sliding_window_width), 'b')
print()
print("Number of bytes needed for the correlation matrix = {0:.3e} ".format(correlations.nbytes))
correlations_gpu = drv.mem_alloc(correlations.nbytes)

block_size = 1024
block_size_x = int(np.sqrt(block_size))
block_size_y = int(np.sqrt(block_size))

gridx = int(np.ceil(correlations.shape[0]/block_size_x))
gridy = int(np.ceil(correlations.shape[1]/block_size_y))

# create two timers so we can speed-test each approach
start = drv.Event()
end = drv.Event()

pycuda.autoinit.context.synchronize()

start.record() # start timing

quadratic_difference(
        correlations_gpu, np.int32(correlations.shape[0]), np.int32(correlations.shape[1]/2), np.int32(correlations.shape[1]), x_gpu, y_gpu, z_gpu, ct_gpu, 
        block=(block_size_x, block_size_y, 1), grid=(gridx, gridy))

pycuda.autoinit.context.synchronize()

end.record() # end timing
# calculate the run length
end.synchronize()
secs = start.time_till(end)*1e-3

print()
print('Time taken for computations is {0:.2e}s.'.format(secs))

start_transfer = time.time()

drv.memcpy_dtoh(correlations, correlations_gpu)

end_transfer = time.time()

print()
print('Data transfer from device to host took {0:.2e}s.'.format(end_transfer -start_transfer))

check = np.zeros(correlations.shape, correlations.dtype)

# Checkif output is correct.
for i in range(check.shape[0]):
    for j in range(i - int(check.shape[1]/2), i + int(check.shape[1]/2)):
        if (j < check.shape[0]) and (j >= 0):
            if (ct[i]-ct[j])**2 < (x[i]-x[j])**2  + (y[i] - y[j])**2 + (z[i] - z[j])**2:
                check[i, j - i +  int(check.shape[1]/2)] = 1

print()
print()
print('check = ', check)
print()
print('check.max() = {0}'.format(check.max()))
print()
print('This should be close to zero: {0}'.format(np.max(np.abs(check - correlations))))
print()
print('check - correlations = ', check -correlations)
