import numpy as np
from numba import cuda, f4, void, boolean
import time

block_size = 9
block_size_x = int(np.sqrt(block_size))
block_size_y = int(np.sqrt(block_size))

@cuda.jit(void(boolean[:,:], f4[:], f4[:], f4[:], f4[:]))
def quadratic_difference(correlations, x, y, z, ct):
    tx = cuda.threadIdx.x  # numbers associated with each thread within a block
    ty = cuda.threadIdx.y  # numbers associated with each thread within a block

    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y

    bwx = cuda.blockDim.x
    bwy = cuda.blockDim.y

    
    i, jj = cuda.grid(2) # global position of the thread

    n, m = correlations.shape  # n = N, m = window size

    # l = i + j - int(m/2)
 
    #l = i + j

    # Suppose the thread block size = 1024 and we have square blocks, i.e. cuda.blockDim.x = cuda.blockDim.y,
    # than we have to copy 64 values to shared memory.
    # I'll separate the base_hits (values of i) and surrounding_hits (values of l).
    base_hits = cuda.shared.array((4, block_size_x), dtype=f4)

    # if ty == 0 and i < n:

    if i == tx + bx * bwx and ty == 0 and i < n: 
        base_hits[0, tx] = x[i]
        base_hits[1, tx] = y[i]
        base_hits[2, tx] = z[i]
        base_hits[3, tx] = ct[i]


    surrounding_hits = cuda.shared.array((4, block_size_y), dtype=f4)

    if jj == ty + by*bwy and tx == 0 and jj <m:
    # if tx ==0 and j < m
        surrounding_hits[0, ty] = x[jj]
        surrounding_hits[1, ty] = y[jj]
        surrounding_hits[2, ty] = z[jj]
        surrounding_hits[3, ty] = ct[jj]

    cuda.syncthreads()
    #if tx == 2 and bx == 0 and ty == 3 and by == 1:
    #    from pdb import set_trace
    #    set_trace()
    #if i < n and j < m and l >= 0 and l < n:
    #if i < n and j < m:
    if i == ( tx + bx * bwx ) and jj == ( ty + by * bwy ) and  i < n and jj < m and jj>i:
        diffx  = base_hits[0, tx] - surrounding_hits[0, ty]
        diffy  = base_hits[1, tx] - surrounding_hits[1, ty]
        diffz  = base_hits[2, tx] - surrounding_hits[2, ty]
        diffct = base_hits[3, tx] - surrounding_hits[3, ty]

        if diffct * diffct <= diffx * diffx + diffy * diffy + diffz * diffz:
            if jj>i:
                correlations[i, jj] = 1

def main():
    #start_computations = cuda.event(timing = True)
    #end_computations   = cuda.event(timing = True)

    N = 81

    x = np.random.random(N).astype(np.float32)
    y = np.random.random(N).astype(np.float32)
    z = np.random.random(N).astype(np.float32)
    ct = np.random.random(N).astype(np.float32)

    # pickle the data for reuse
    np.save("./x.pkl", x)
    np.save("./y.pkl", y)
    np.save("./z.pkl", z)       
    np.save("./ct.pkl", ct)
    #start_transfer = time.time()

    x_gpu = cuda.to_device(x)
    y_gpu = cuda.to_device(y)
    z_gpu = cuda.to_device(z)
    ct_gpu = cuda.to_device(ct)

    #end_transfer = time.time()

    print()
    #print('Data transfer from host to device plus memory allocation on device took {0:.2e}s.'.format(end_transfer - start_transfer))

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

    #start_computations.record()

    quadratic_difference[(gridx, gridy), (block_size_x, block_size_y)](correlations_gpu, x_gpu, y_gpu, z_gpu, ct_gpu)

    #end_computations.record()

    ##end_computations.synchronize()

    print()
    #print('Time taken for computations is {0:.3e}s.'.format(1e-3 * start_computations.elapsed_time(end_computations)))

    #start_transfer = time.time()

    correlations_gpu.copy_to_host(correlations)

    #end_transfer = time.time()

    print()
    #print('Data transfer from device to host took {0:.2e}s.'.format(end_transfer -start_transfer))

    print()
    print('correlations = ', correlations)
   
    check = np.zeros_like(correlations)
   
    # Checkif output is correct.
    for i in range(check.shape[0]):
        for j in range(i+1, check.shape[1]+1):
            if (ct[i]-ct[j])**2 < (x[i]-x[j])**2  + (y[i] - y[j])**2 + (z[i] - z[j])**2:
                check[i, j - i -1] = 1


    np.save("./correlations.pkl", correlations)
    np.save("./check.pkl", check)
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
