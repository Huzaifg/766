import numpy as np
from PIL import Image
import scipy.ndimage as ndi
from skimage import filters, feature
import matplotlib.pyplot as plt

def gaussian_kernel(size, sigma=1):
    """Returns a 2D Gaussian kernel array."""
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    kernel_1D = np.exp(-kernel_1D**2 / (2 * sigma**2))

    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
    kernel_2D *= 1.0 / kernel_2D.sum()

    kernel_2D = kernel_2D[:, :, np.newaxis]
 
    return kernel_2D

def hw3_walkthrough1():
    #----------------------- 
    # Image processing: convolution, Gaussian smoothing
    #-----------------------
    # Image credit: http://commons.wikimedia.org/wiki/File:Beautiful-pink-flower_-_West_Virginia_-_ForestWander.jpg

    img = Image.open('data/flower.png')
    img = np.array(img)

    fig, axs = plt.subplots(4, figsize=(8,20))
    for ax in axs:
        ax.axis(False)
    axs[0].imshow(img)
    axs[0].set_title('Original')

    # Here we will demonstrate apply Gaussian blur with three different sigmas.
    # Initialize the sigma list
    sigma_list = [2, 4, 8]

    # In the following loop, each iteration applies a Gaussian blur with a
    # different sigma
    for i, sigma in enumerate(sigma_list):
        # Get the Gaussian kernel
        # Rule of thumb: set kernal size k ~= 2*pi*sigma
        kernel_size = np.ceil(2*np.pi*sigma).astype(int)
        kernel = gaussian_kernel(kernel_size, sigma=sigma)

        # Perform convolution 
        blur_img = ndi.convolve(img, kernel)
        
        # Display the result
        axs[i+1].imshow(blur_img)
        axs[i+1].set_title(f'$\sigma$ = {sigma}')

    fig.savefig('outputs/blur_flowers.png')
    # plt.show()

    #----------------------- 
    # Edge detection
    #-----------------------

    fig, axs = plt.subplots(2, 2)
    img = Image.open('data/hello.png')
    axs[0, 0].imshow(img)
    axs[0, 0].set_title('Color Image')

    # Convert the image to grayscale
    gray_img = img.convert('L')
    axs[0, 1].imshow(gray_img, cmap='gray')
    axs[0, 1].set_title('Grayscale Image')
    
    img = np.array(img)
    gray_img = np.array(gray_img)

    # Sobel edge detection
    thresh = 0.12
    edge_img = filters.sobel(gray_img) > thresh
    axs[1, 0].imshow(edge_img, cmap='gray')
    axs[1, 0].set_title('Sobel Edge Detection')

    # Canny edge detection
    edge_img = feature.canny(gray_img, sigma=2.5, low_threshold=0.2, high_threshold=0.3)
    axs[1,1].imshow(edge_img, cmap='gray')
    axs[1,1].set_title('Canny Edge Detection')

    plt.savefig('outputs/hello_edges.png')
