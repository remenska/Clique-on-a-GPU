import numpy as np
from numba import cuda, f4, void, boolean, jit
import time

# block_size = 1024
# block_size_x = int(np.sqrt(block_size))
# block_size_y = int(np.sqrt(block_size))
block_size_x = 3
block_size_y = 3

surrounding_hits_length = block_size_x + block_size_y - 1

@cuda.jit(void(boolean[:,:], f4[:], f4[:], f4[:], f4[:]))
def quadratic_difference(correlations, x, y, z, ct):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    
    i, j = cuda.grid(2)

    n, m = correlations.shape

    # l = i + j - int(m/2)
 
    l = i + j

    # Suppose the thread block size = 1024 and we have square blocks, i.e. cuda.blockDim.x = cuda.blockDim.y,
    # than we have to copy 64 values to shared memory.
    # I'll separate the base_hits (values of i) and surrounding_hits (values of l).
    base_hits = cuda.shared.array((4, block_size_x), dtype=f4)

    if ty == 0 and i < n:
        base_hits[0, tx] = x[i]
        base_hits[1, tx] = y[i]
        base_hits[2, tx] = z[i]
        base_hits[3, tx] = ct[i]

    surrounding_hits = cuda.shared.array((4, surrounding_hits_length), dtype=f4)

    if tx == 0 and l < n:
        surrounding_hits[0, ty] = x[l]
        surrounding_hits[1, ty] = y[l]
        surrounding_hits[2, ty] = z[l]
        surrounding_hits[3, ty] = ct[l]

    if tx == block_size_x - 1 and l < n:
        surrounding_hits[0, tx + ty] = x[l]
        surrounding_hits[1, tx + ty] = y[l]
        surrounding_hits[2, tx + ty] = z[l]
        surrounding_hits[3, tx + ty] = ct[l]

    cuda.syncthreads()

    if i < n and j < m and l < n:
        diffx  = base_hits[0, tx] - surrounding_hits[0, tx + ty]
        diffy  = base_hits[1, tx] - surrounding_hits[1, tx + ty]
        diffz  = base_hits[2, tx] - surrounding_hits[2, tx + ty]
        diffct = base_hits[3, tx] - surrounding_hits[3, tx + ty]

        if diffct * diffct < diffx * diffx + diffy * diffy + diffz * diffz:
            correlations[i, j] = 1

def main():
    start_computations = cuda.event(timing = True)
    end_computations   = cuda.event(timing = True)

    N = 81

    # try:
    #     x = np.load("x.npy")
    #     y = np.load("y.npy")
    #     z = np.load("z.npy")
    #     ct = np.load("ct.npy")

    #     assert x.size == N

    # except (FileNotFoundError, AssertionError):
    x = np.random.random(N).astype(np.float32)
    y = np.random.random(N).astype(np.float32)
    z = np.random.random(N).astype(np.float32)
    #ct = np.linspace(0, 0.1, N)
    ct = np.random.random(N).astype(np.float32)

    np.save("x.npy", x)
    np.save("y.npy", y)
    np.save("z.npy", z)
    np.save("ct.npy", ct)

    start_transfer = time.time()

    x_gpu = cuda.to_device(x)
    y_gpu = cuda.to_device(y)
    z_gpu = cuda.to_device(z)
    ct_gpu = cuda.to_device(ct)

    end_transfer = time.time()

    print()
    print('Data transfer from host to device plus memory allocation on device took {0:.2e}s.'.format(end_transfer - start_transfer))

    # The number of consecutive hits corresponding to the light crossing time of the detector (1km/c).
    N_light_crossing = 27

    # This used to be 2 * N_light_crossing, but caused redundant calculations.
    sliding_window_width =  N_light_crossing
    # problem_size = N * sliding_window_width

    correlations = np.zeros((N, sliding_window_width), 'b')
    print()
    print("Number of bytes needed for the correlation matrix = {0:.3e} ".format(correlations.nbytes))

    correlations_gpu = cuda.to_device(correlations)

    gridx = int(np.ceil(correlations.shape[0]/block_size_x))
    gridy = int(np.ceil(correlations.shape[1]/block_size_y))

    start_computations.record()

    quadratic_difference[(gridx, gridy), (block_size_x, block_size_y)](correlations_gpu, x_gpu, y_gpu, z_gpu, ct_gpu)

    end_computations.record()

    end_computations.synchronize()

    print()
    print('Time taken for gpu computations is {0:.2e}s.'.format(1e-3 * start_computations.elapsed_time(end_computations)))

    start_transfer = time.time()

    correlations_gpu.copy_to_host(correlations)

    end_transfer = time.time()

    print()
    print('Data transfer from device to host took {0:.2e}s.'.format(end_transfer -start_transfer))

    print()
    print('correlations = ', correlations)
 
    # Speed up the CPU processing.
    @jit
    def correlations_cpu(check, x, y, z, ct):
        for i in range(check.shape[0]):
            for j in range(i, i + check.shape[1]):
                if j < check.shape[0]:
                    if (ct[i]-ct[j])**2 < (x[i]-x[j])**2  + (y[i] - y[j])**2 + (z[i] - z[j])**2:
                        check[i, j - i] = 1
        return check
    
    try:
        check = np.load("check.npy")

        assert N == check.shape[0] and sliding_window_width == check.shape[1]

    except (FileNotFoundError, AssertionError):
            start_cpu_computations = time.time()   
            
            check = np.zeros_like(correlations)
            # Checkif output is correct.
            check = correlations_cpu(check, x, y, z, ct)

            end_cpu_computations = time.time()   
            
            print()
            print('Time taken for cpu computations is {0:.2e}s.'.format(end_cpu_computations - start_cpu_computations)) 

            np.save("check.npy", check) 

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
        print('check - correlations = ', check -correlations)

if __name__ == '__main__':
    main()
