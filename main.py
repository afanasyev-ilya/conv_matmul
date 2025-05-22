import numpy as np
from skimage import data
from skimage.color import rgb2gray
from skimage.transform import resize
import matplotlib
matplotlib.use('Agg')  # Switch to non-GUI backend
import matplotlib.pyplot as plt
import time
import os

from enum import Enum

class FilterType(Enum):
    EDGES = 1
    BLUR = 2

class ComputeType(Enum):
    NAIVE_CONV = 1
    MATMUL = 2


def remove_png_files():
    # Get current directory
    current_dir = os.getcwd()
    
    # Find all .png files
    png_files = [f for f in os.listdir(current_dir) if f.lower().endswith('.png')]
    
    if not png_files:
        print("No .png files found in current directory")
        return
    
    # Remove each file and print what was removed
    for file in png_files:
        try:
            os.remove(file)
            print(f"Removed: {file}")
        except Exception as e:
            print(f"Error removing {file}: {str(e)}")
    
    print(f"\nTotal removed: {len(png_files)} .png files")

def load_kernel(type):
    print("kernel info: ")
    if type == FilterType.EDGES: 
        # Define kernel
        kernel = np.array([[[[-1, -1, -1],
                                [-1,  8, -1],
                                [-1, -1, -1]]]], dtype=np.float32)
        
    elif type == FilterType.BLUR:
        # Create RGB blur kernel (3x3) - processes each channel independently
        kernel = np.zeros((3, 3, 3, 3), dtype=np.float32)  # [C_out, C_in, H, W]
        blur = np.ones((3, 3)) / 9.0  # 3x3 box blur
        for co in range(3):
            kernel[co, co] = blur  # Each output channel blurs its corresponding input channel 
    else:
        print("Error: unknow kernel")
        exit(1)
    print(kernel.shape)
    return kernel


def conv_nchw_4d(input, kernel):
    N, C_in, H, W = input.shape
    C_out, C_in, K_h, K_w = kernel.shape
    """
    Python implementation of 4D convolution matching the C version
    """
    H_out = H - K_h + 1
    W_out = W - K_w + 1
    output = np.zeros((N, C_out, H_out, W_out), dtype=np.float32)
    for n in range(N):
        for co in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    sum_val = 0.0
                    for ci in range(C_in):
                        for kh in range(K_h):
                            for kw in range(K_w):
                                sum_val += input[n, ci, i + kh, j + kw] * kernel[co, ci, kh, kw]
                    output[n, co, i, j] = sum_val
    return output

def conv_im2col(input, kernel):
    """
    Optimized convolution using im2col and matrix multiplication
    with detailed memory analysis and intermediate shape printing
    
    Args:
        input: Input tensor in NCHW format (Batch, Channels, Height, Width)
        kernel: Convolution kernel in OIHW format (OutChannels, InChannels, K_h, K_w)
    
    Returns:
        Output tensor in NCHW format
    """
    # ---------------------------
    # Step 0: Extract dimensions
    # ---------------------------
    N, C_in, H, W = input.shape
    C_out, C_in_k, K_h, K_w = kernel.shape
    assert C_in == C_in_k, "Input channels must match kernel's input channels"
    
    # Calculate output dimensions
    H_out = H - K_h + 1
    W_out = W - K_w + 1
    
    print("\n=== Memory Analysis ===")
    print(f"Input shape: {input.shape} | Size: {input.nbytes/1024:.2f} KB")
    print(f"Kernel shape: {kernel.shape} | Size: {kernel.nbytes/1024:.2f} KB")

    # --------------------------------------------------
    # Step 1: im2col transformation (most memory intensive)
    # --------------------------------------------------
    # Each column in 'cols' represents a flattened receptive field
    # Shape: (C_in*K_h*K_w, N*H_out*W_out)
    # Memory: (3*3*3) * (1*2*2) * 4 bytes = 108 * 4 = 432 bytes (for 64x64 input)
    cols = np.zeros((C_in * K_h * K_w, N * H_out * W_out), dtype=input.dtype)
    
    # Debug: Show im2col matrix dimensions
    print(f"\n[im2col] Creating column matrix...")
    print(f"Column matrix shape: {cols.shape}")
    print(f"Column matrix size: {cols.nbytes/1024:.2f} KB")
    
    # Populate columns with image patches
    for idx in range(N):
        for h in range(H_out):
            for w in range(W_out):
                # Extract patch from input tensor
                patch = input[idx, :, h:h+K_h, w:w+K_w]
                
                # Calculate column index: [N][H][W] -> linear index
                col_idx = idx*H_out*W_out + h*W_out + w
                
                # Flatten patch and store as column
                cols[:, col_idx] = patch.flatten()

    # --------------------------------------------------
    # Step 2: Kernel reshaping
    # --------------------------------------------------
    # Reshape kernel to (C_out, C_in*K_h*K_w) into 2D matrix
    # This allows matrix multiplication with columns matrix
    kernel_reshaped = kernel.reshape(C_out, C_in*K_h*K_w)
    
    print(f"\n[Kernel] Reshaped kernel: {kernel_reshaped.shape}")
    print(f"Reshaped kernel size: {kernel_reshaped.nbytes/1024:.2f} KB")

    # --------------------------------------------------
    # Step 3: Matrix multiplication
    # --------------------------------------------------
    # Output shape: (C_out, N*H_out*W_out)
    # This single matmul replaces all nested loops
    print("\n[Matmul] Performing kernel @ columns...")
    print(f"Matrix multiply: {kernel_reshaped.shape} @ {cols.shape}")
    output_flat = kernel_reshaped @ cols  # Most computationally intensive part

    # --------------------------------------------------
    # Step 4: Output reshaping
    # --------------------------------------------------
    # Convert back to NCHW format:
    # 1. Reshape to (C_out, N, H_out, W_out)
    # 2. Transpose to (N, C_out, H_out, W_out)
    output = output_flat.reshape(C_out, N, H_out, W_out).transpose(1, 0, 2, 3)
    
    # Final memory stats
    print(f"\n[Output] Final shape: {output.shape}")
    print(f"Output size: {output.nbytes/1024:.2f} KB")
    print("=======================")
    
    return output

def preprocess_image(image, filter): 
    """Load and resize image with 3 channels (convert grayscale to RGB)"""
    # Convert grayscale to RGB if needed (blur)
    if filter == FilterType.BLUR and len(image.shape) == 2:
        image = np.stack([image]*3, axis=-1)
        
    # convert RGB to grayscale if needed
    if filter == FilterType.EDGES and len(image.shape) == 3:
        image = rgb2gray(image)
        image = np.expand_dims(image, axis=-1)

    if len(image.shape) == 3:
        image = image.transpose(2, 0, 1).astype(np.float32)  # CHW format
        
    return image
    


def main(filter_type, compute_type):
    # load images
    print("Loading images...")
    images = [data.astronaut(), data.astronaut()]
    print("Images loaded successfully")

    for i in range(len(images)):
        images[i] = preprocess_image(images[i], filter_type)
    print(f"Image shapes after resize: {images[0].shape}")
        
    # Create input tensor
    input_nchw = np.stack(images)[:, :, :]
    print(f"Input tensor shape: {input_nchw.shape}")

    # Define kernel
    kernel = load_kernel(filter_type)

    # Run convolution
    print("Running convolution...")
    t_st = time.time()
    if compute_type == ComputeType.NAIVE_CONV:
        output = conv_nchw_4d(input_nchw, kernel)
    elif compute_type == ComputeType.MATMUL:
        output = conv_im2col(input_nchw, kernel)
    else:
        print("Error: incorrect compute type")
        exit(1)
    t_end = time.time()
    print("CONV TIME: ", t_end-t_st, " s")
    print("Convolution completed successfully")
    print(f"Output shape: {output.shape}")

    def save_image(img, filename):
        img = (img - img.min()) / (img.max() - img.min())
        plt.imsave(filename, img, cmap='gray')

    for i in range(0, len(images)):
        save_image(input_nchw[0, 0], 'original_' + str(i) + '.png')
    
    for i in range(0, len(images)):
        save_image(output[i, 0],"result_" + str(i) + '.png')

    print("Results saved")


remove_png_files()
main(FilterType.BLUR, ComputeType.MATMUL)