import numpy as np
from numba import cuda, f4, void, boolean
import time

@cuda.jit(void(boolean[:,:], f4[:], f4[:], f4[:], f4[:]))
def quadratic_difference(correlations, x, y, z, ct):

    i, j = cuda.grid(2)

    n, m = correlations.shape

    l = i + j - int(m/2)
 
    if i < n and j < m and l >= 0 and l < n:
        diffct = ct[i] - ct[l]
        diffx  = x[i] - x[l]
        diffy  = y[i] - y[l]
        diffz  = z[i] - z[l]
        if diffct * diffct < diffx * diffx + diffy * diffy + diffz * diffz:
            correlations[i, j] = 1

def main():
    start_computations = cuda.event(timing = True)
    end_computations   = cuda.event(timing = True)

    N = 30000

    x = np.random.random(N).astype(np.float32)
    y = np.random.random(N).astype(np.float32)
    z = np.random.random(N).astype(np.float32)
    ct = np.random.random(N).astype(np.float32)

    start_transfer = time.time()

    x_gpu = cuda.to_device(x)
    y_gpu = cuda.to_device(y)
    z_gpu = cuda.to_device(z)
    ct_gpu = cuda.to_device(ct)

    end_transfer = time.time()

    print()
    print('Data transfer from host to device plus memory allocation on device took {0:.2e}s.'.format(end_transfer - start_transfer))

    # The number of consecutive hits corresponding to the light crossing time of the detector (1km/c).
    N_light_crossing = 1500
    # We have a sliding window with size 2 * N_light_crossing to consider for correlations.
    sliding_window_width = 2 * N_light_crossing
    # problem_size = N * sliding_window_width

    correlations = np.zeros((N, sliding_window_width), 'b')
    print()
    print("Number of bytes needed for the correlation matrix = {0:.3e} ".format(correlations.nbytes))

    correlations_gpu = cuda.to_device(correlations)

    block_size = 1024
    block_size_x = int(np.sqrt(block_size))
    block_size_y = int(np.sqrt(block_size))

    gridx = int(np.ceil(correlations.shape[0]/block_size_x))
    gridy = int(np.ceil(correlations.shape[1]/block_size_y))

    start_computations.record()

    quadratic_difference[(gridx, gridy), (block_size_x, block_size_y)](correlations_gpu, x_gpu, y_gpu, z_gpu, ct_gpu)

    end_computations.record()

    end_computations.synchronize()

    print()
    print('Time taken for computations is {0:.3e}s.'.format(1e-3 * start_computations.elapsed_time(end_computations)))

    start_transfer = time.time()

    correlations_gpu.copy_to_host(correlations)

    end_transfer = time.time()

    print()
    print('Data transfer from device to host took {0:.2e}s.'.format(end_transfer -start_transfer))

    print()
    print('correlations = ', correlations)
   
    check = np.zeros_like(correlations)
   
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
    print('This should be close to zero: {0}'.format(np.sum(np.abs(check - correlations))))
    print()
    print('check - correlations = ', check -correlations)

if __name__ == '__main__':
    main()
