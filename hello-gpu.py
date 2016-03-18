import pycuda.autoinit
import pycuda.driver as drv
import numpy

from pycuda.compiler import SourceModule

mod = SourceModule("""
__global__ void multiply_them(float *dest, float *a, float *b)
{
  const int i = threadIdx.x;
  dest[i] = a[i] * b[i];
}
""")

multiply_them = mod.get_function("multiply_them")

N = 400

x = numpy.random.randn(N).astype(numpy.float32)
y = numpy.random.randn(N).astype(numpy.float32)
z = numpy.random.randn(N).astype(numpy.float32)
ct = numpy.random.randn(N).astype(numpy.float32)

dest = numpy.zeros_like(z)
multiply_them(
        drv.Out(dest), drv.In(x), drv.In(y),
        block=(N,1,1), grid=(1,1))

print(dest-x*y)
