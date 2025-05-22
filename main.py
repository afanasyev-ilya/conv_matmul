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


def main(mode = "edge"):
    # Test image loading first
    try:
        print("Loading images...")
        image1 = data.astronaut()
        print("!!!!!!!!!! ", image1.shape)
        #image2 = data.camera()
        image2 = image1
        print("Images loaded successfully")
        
        image1 = preprocess_image(image1)
        image2 = preprocess_image(image2)
        print(f"Image shapes after resize: {image1.shape}, {image2.shape}")
        
    except Exception as e:
        print(f"Error in image loading: {str(e)}")
        exit(1)

    # Create input tensor
    try:
        input_nchw = np.stack([image1, image2])[:, :, :]
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

    # Save results to files instead of showing interactively
    def save_image(img, filename):
        img = (img - img.min()) / (img.max() - img.min())
        plt.imsave(filename, img, cmap='gray')

    save_image(input_nchw[0, 0], 'original1.png')
    save_image(input_nchw[1, 0], 'original2.png')
    save_image(output[0, 0], mode + '1.png')
    save_image(output[1, 0], mode + '2.png')

    print("Results saved")

main()