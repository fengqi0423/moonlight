import numpy
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

mod = SourceModule(
    """
    __global__ void matrix_multiply(float *a, float *b, float *r, int m, int n, int k){
       
        int per_thread_m = m / (gridDim.x * blockDim.x);
        int per_thread_n = n / (gridDim.y * blockDim.y);

        if(m % (gridDim.x * blockDim.x) > 0.5){
            per_thread_m = per_thread_m + 1;
        }
        if(n % (gridDim.y * blockDim.y) > 0.5){
            per_thread_n = per_thread_n + 1;
        }

        int idx     = blockIdx.x * blockDim.x + threadIdx.x;
        int idy     = blockIdx.y * blockDim.y + threadIdx.y;
        int start_m = per_thread_m * idx;
        int start_n = per_thread_n * idy;
        int end_m   = start_m + per_thread_m;
        int end_n   = start_n + per_thread_n;

        int i, j, kk;
        float sum=0;
        for(i=start_m; i<end_m && i<m; i++){
            for(j=start_n; j<end_n && j<n; j++, sum=0.0){
                for(kk=0; kk < k; kk++){
                    sum = sum + a[k*i+kk] * b[j+kk*n];
                }
                r[i*n+j] = sum;
            }
        }
    }
    """
    )

def matrix_multiply_gpu(a, b, block=(5,5,1), grid=(200,200)):
    """
    a is a m-by-k dimensional 2d numpy array
    b is a k-by-n dimensional 2d numpy array
    returns a m-by-n dimensional 2d numpy array that is the product of a and b
    """
    m, k1 = a.shape
    k2, n = b.shape
    assert k1==k2, "Wokao, matrics dimesions do not match."

    if a.dtype != numpy.float32:
        a = a.astype(numpy.float32)
    if b.dtype != numpy.float32:
        b = b.astype(numpy.float32)

    r = numpy.zeros((m, n), dtype=numpy.float32)
    
    a_gpu = cuda.mem_alloc(a.nbytes)
    b_gpu = cuda.mem_alloc(b.nbytes)
    r_gpu = cuda.mem_alloc(m*n*4)

    cuda.memcpy_htod(a_gpu, numpy.array(a))
    cuda.memcpy_htod(b_gpu, numpy.array(b))

    func = mod.get_function("matrix_multiply")
    func(
        a_gpu, 
        b_gpu, 
        r_gpu, 
        numpy.int32(m), 
        numpy.int32(n), 
        numpy.int32(k1), 
        grid=grid, 
        block=block,
        )
    cuda.Context.synchronize()
    cuda.memcpy_dtoh(r, r_gpu)

    a_gpu.free()
    b_gpu.free()
    r_gpu.free()
    
    if numpy.isnan(r).any():
        print "a",a
        print "b",b
        print "r",r
        exit(1)
    return r


if __name__ == "__main__":
    import time
    import sys

    dim1 = int(sys.argv[1])
    dim2 = int(sys.argv[2])
    dim3 = int(sys.argv[3])
    a = numpy.random.randn(dim1, dim2).astype(numpy.float32)
    b = numpy.random.randn(dim2, dim3).astype(numpy.float32)
#    a = numpy.ones((dim1, dim2), dtype=numpy.float32)
#    b = numpy.ones((dim2, dim1), dtype=numpy.float32)

    t0 = time.time() 
    r = matrix_multiply_gpu(a, b, (25, 1, 1), (100, 1))
    t1 = time.time()
#    print r
    print t1-t0

    t0 = time.time()
    r = numpy.dot(a, b)
    t1 = time.time()
#    print r
    print t1-t0
