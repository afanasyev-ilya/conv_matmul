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
from conv_impl import conv_nchw_4d, conv_im2col, winograd_conv3x3
import argparse


class FilterType(Enum):
    EDGES = 1
    BLUR = 2

class ComputeType(Enum):
    DIRECT_CONV = 1
    MATMUL = 2
    WINOGRAD_CONV = 3


def remove_png_files():
    # Get current directory
    current_dir = os.getcwd()
    images_dir = os.path.join(current_dir, "images")
    
    # Find all .png files
    png_files = [f for f in os.listdir(images_dir) if f.lower().endswith('.png')]
    
    if not png_files:
        print("No .png files found in images_dir")
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
    


def filter(filter_type, compute_type):
    os.makedirs("./images", exist_ok=True)

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
    if compute_type == ComputeType.DIRECT_CONV:
        output = conv_nchw_4d(input_nchw, kernel)
    elif compute_type == ComputeType.MATMUL:
        output = conv_im2col(input_nchw, kernel)
    elif compute_type == ComputeType.WINOGRAD_CONV:
        output = winograd_conv3x3(input_nchw, kernel)
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
        save_image(input_nchw[0, 0], 'images/original_' + str(i) + '.png')
    
    for i in range(0, len(images)):
        save_image(output[i, 0],"images/result_" + str(i) + '.png')

    print("Results saved")


# ================= Command-line Handling ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Convolution Methods')
    parser.add_argument('--filter', type=str, required=True, 
                        choices=['EDGES', 'BLUR'],
                        help='Filter type: EDGES or BLUR')
    parser.add_argument('--compute', type=str, required=True,
                        choices=['DIRECT_CONV', 'MATMUL', 'WINOGRAD_CONV'],
                        help='Compute method: DIRECT_CONV, MATMUL, or WINOGRAD_CONV')
    
    args = parser.parse_args()
    
    # Convert strings to Enum values
    filter_type = FilterType[args.filter]
    compute_type = ComputeType[args.compute]
    
    remove_png_files()
    filter(filter_type, compute_type)