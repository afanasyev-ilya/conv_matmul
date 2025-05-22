import numpy as np
from skimage import data
from skimage.color import rgb2gray
from skimage.transform import resize
import matplotlib
matplotlib.use('Agg')  # Switch to non-GUI backend
import matplotlib.pyplot as plt

def load_kernel(type = "edge"):
    print("kernel info: ")
    if type == "edge": 
        # Define kernel
        kernel = np.array([[[[-1, -1, -1],
                                [-1,  8, -1],
                                [-1, -1, -1]]]], dtype=np.float32)
        
    elif type == "blur":
        # Create RGB blur kernel (3x3) - processes each channel independently
        kernel = np.zeros((3, 3, 3, 3), dtype=np.float32)  # [C_out, C_in, H, W]
        blur = np.ones((3, 3)) / 9.0  # 3x3 box blur
        for co in range(3):
            kernel[co, co] = blur  # Each output channel blurs its corresponding input channel 
    else:
        print("Error: unknow kernel")
        exit(1)
    print(kernel.shape)
    print(kernel)
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

def preprocess_image(image): 
    """Load and resize image with 3 channels (convert grayscale to RGB)"""
    # Convert grayscale to RGB if needed
    if len(image.shape) == 2:
        image = np.stack([image]*3, axis=-1)
    
    return image.transpose(2, 0, 1).astype(np.float32)  # CHW format


def main(mode = "blur"):
    # Test image loading first
    try:
        print("Loading images...")
        images = [data.astronaut(), data.astronaut()]
        print("Images loaded successfully")
        
        for i in range(len(images)):
            images[i] = preprocess_image(images[i])
        print(f"Image shapes after resize: {images[0].shape}")
        
    except Exception as e:
        print(f"Error in image loading: {str(e)}")
        exit(1)

    # Create input tensor
    try:
        input_nchw = np.stack(images)[:, :, :]
        print(f"Input tensor shape: {input_nchw.shape}")
    except Exception as e:
        print(f"Error creating input tensor: {str(e)}")
        exit(1)

    # Define kernel
    kernel = load_kernel(mode)

    # Run convolution
    print("Running convolution...")
    output = conv_nchw_4d(input_nchw, kernel)
    print("Convolution completed successfully")
    print(f"Output shape: {output.shape}")

    def save_image(img, filename):
        img = (img - img.min()) / (img.max() - img.min())
        plt.imsave(filename, img, cmap='gray')

    for i in range(0, len(images)):
        save_image(input_nchw[0, 0], 'original_' + str(i) + '.png')
    
    for i in range(0, len(images)):
        save_image(output[i, 0], mode + str(i) + '.png')

    print("Results saved")

main()