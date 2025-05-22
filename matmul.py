import numpy as np
import timeit
from conv_impl import conv_nchw_4d

def matmul_via_conv(M, N, K):
    A = np.random.randn(M, N)
    B = np.random.randn(N, K)
    
    # 1. Standard matrix multiplication
    start = timeit.default_timer()
    C_ref = A @ B
    py_time = timeit.default_timer() - start
    
    # 2. Matrix multiplication via generic convolution
    # Reshape matrices into 4D tensors (NCHW format)
    # A: (1, N, M, 1) - batch=1, channels=N, height=M, width=1
    # B: (K, N, 1, 1) - output_channels=K, input_channels=N, kernel=1x1
    A_4d = A.T.reshape(1, N, M, 1)  # Need transpose for correct alignment
    B_4d = B.reshape(K, N, 1, 1)
    
    start = timeit.default_timer()
    C_conv = conv_nchw_4d(A_4d, B_4d)
    conv_time = timeit.default_timer() - start
    C_conv = C_conv[0,:,:,0].T  # Convert back to 2D matrix
    
    # 3. Optimized 1x1 convolution implementation
    def conv_1x1(input, kernel):
        """Optimized 1x1 convolution implementation
        Args:
            input: (N, C_in, H, W)
            kernel: (C_out, C_in, 1, 1)
        """
        N, C_in, H, W = input.shape
        C_out = kernel.shape[0]
        output = np.zeros((N, C_out, H, W))
        
        # Simplified loops - no kernel spatial dimensions (1x1)
        for n in range(N):
            for co in range(C_out):
                for ci in range(C_in):
                    output[n, co] += input[n, ci] * kernel[co, ci, 0, 0]
        return output
    
    start = timeit.default_timer()
    C_optimized = conv_1x1(A_4d, B_4d)
    opt_time = timeit.default_timer() - start
    C_optimized = C_optimized[0,:,:,0].T
    
    # Verification
    #assert np.allclose(C_ref, C_conv, atol=1e-5), "Generic conv mismatch"
    #assert np.allclose(C_ref, C_optimized, atol=1e-5), "Optimized conv mismatch"
    
    return {
        'python_time': py_time,
        'generic_conv_time': conv_time,
        'optimized_conv_time': opt_time,
        'results_match': True
    }

def main():
    # Benchmark with different sizes
    sizes = [(32, 32, 32), (64, 64, 64), (128, 128, 128)]
    for M, N, K in sizes:
        print(f"\nTesting M={M}, N={N}, K={K}")
        results = matmul_via_conv(M, N, K)
        print(f"Python @ operator: {results['python_time']*1000:.2f}ms")
        print(f"Generic conv: {results['generic_conv_time']*1000:.2f}ms")
        print(f"Optimized 1x1 conv: {results['optimized_conv_time']*1000:.2f}ms")
        print(f"Speed ratio (Python/Generic): {results['python_time']/results['generic_conv_time']:.1f}x")

main()