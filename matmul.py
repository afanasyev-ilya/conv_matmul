import numpy as np
import timeit
from conv_impl import conv_nchw_4d, conv_1x1

def matmul_via_conv(M, N, K):
    A = np.random.randn(M, N).astype(np.float32)
    B = np.random.randn(N, K).astype(np.float32)
    
    # 1. Standard matrix multiplication
    start = timeit.default_timer()
    C_ref = A @ B
    matmul_time = timeit.default_timer() - start
    
    # 2. Matrix multiplication via generic convolution
    # Reshape matrices into 4D tensors (NCHW format)
    # A: (1, N, M, 1) - batch=1, channels=N, height=M, width=1
    # B: (K, N, 1, 1) - output_channels=K, input_channels=N, kernel=1x1
    A_4d = A.T.reshape(1, N, M, 1).astype(np.float32)
    B_4d = B.T.reshape(K, N, 1, 1).astype(np.float32)
    
    start = timeit.default_timer()
    C_conv = conv_nchw_4d(A_4d, B_4d)
    conv_time = timeit.default_timer() - start
    C_conv = C_conv[0,:,:,0].T  # Convert back to 2D matrix
    
    start = timeit.default_timer()
    C_optimized = conv_1x1(A_4d, B_4d)
    conv_1x1_time = timeit.default_timer() - start
    C_optimized = C_optimized[0,:,:,0].T
    
    # Verification
    assert np.allclose(C_ref, C_conv, atol=1e-5), "Generic conv mismatch"
    assert np.allclose(C_ref, C_optimized, atol=1e-5), "Optimized conv mismatch"
    
    return {
        'matmul_time': matmul_time,
        'conv_generic_time': conv_time,
        'conv_1x1_time': conv_1x1_time,
        'results_match': True
    }

def main():
    # Benchmark with different sizes
    sizes = [(3, 3, 3), (64, 64, 64), (128, 128, 128)]
    for M, N, K in sizes:
        print(f"\nTesting M={M}, N={N}, K={K}")
        results = matmul_via_conv(M, N, K)
        print(f"Matmul: {results['matmul_time']*1000:.2f}ms")
        print(f"Conv generic: {results['conv_generic_time']*1000:.2f}ms")
        print(f"Conv 1x1: {results['conv_1x1_time']*1000:.2f}ms")
        print(f"Speed ratio (Conv/Matmul): {results['conv_1x1_time']/results['matmul_time']:.2f}x")

main()