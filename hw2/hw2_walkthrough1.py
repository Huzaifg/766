from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import dilation, erosion
from skimage.filters import threshold_otsu


def hw2_walkthrough1():
    # -----------------
    # Convert a grayscale image to a binary image
    # -----------------
    img = Image.open('data/coins.png')
    img = img.convert('L')  # Convert the image to grayscale
    img = np.array(img)

    # -----------------
    # plot the histogram of the image
    # -----------------
    fig, ax = plt.subplots(1, 1)
    ax.hist(img.ravel(), bins=256, range=(0, 255), fc='k', ec='k')
    ax.set_title('Histogram')
    ax.set_xlim([0, 255])
    plt.show()

    # After plotting the bins were printed 
    # Using the plot and the bins, we can see that the lowest bins are around 90
    hist_data, bin_edges = np.histogram(img, bins=256, range=(0, 255))
    sorted_indices = np.argsort(hist_data)
    lowest_bins = bin_edges[sorted_indices[:10]]
    print(lowest_bins)

    # Convert the image into a binary image by applying a threshold
    threshold = 90
    bw_img = img > threshold

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title('Original Image')

    ax[1].imshow(bw_img, cmap='gray')
    ax[1].set_title('Binary Image')

    fig.savefig('outputs/binary_coins.png')
    plt.show()

    # -----------------
    # Remove noises in the binary image
    # -----------------
    # Clean the image (you may notice some holes in the coins) by using
    # dilation and then erosion

    # Specify the size of the structuring element for erosion/dilation
    k = 27
    selem = np.ones((k, k))

    fig, ax = plt.subplots(1, 2)
    try:
        processed_img = dilation(bw_img, selem=selem)
    except:
        processed_img = dilation(bw_img, footprint=selem)
    ax[0].imshow(processed_img, cmap='gray')
    ax[0].set_title('After Dilation')

    # Apply erosion then dilation once to remove the noises
    try:
        processed_img = erosion(processed_img, selem=selem)
    except:
        processed_img = erosion(processed_img, footprint=selem)
    ax[1].imshow(processed_img, cmap='gray')
    ax[1].set_title('After Erosion')

    plt.savefig('outputs/noise_removal_coins.png')
    plt.show()

    # -----------------
    # Remove the rices
    # -----------------
    # Apply erosion then dilation once to remove the rices

    # Specify the size of the structuring element for erosion/dilation
    k = 27
    selem = np.ones((k, k))

    fig, ax = plt.subplots(1, 2)
    try:
        processed_img = erosion(processed_img, selem=selem)
    except:
        processed_img = erosion(processed_img, footprint=selem)
    ax[0].imshow(processed_img, cmap='gray')
    ax[0].set_title('After Erosion')

    try:
        processed_img = dilation(processed_img, selem=selem)
    except:
        processed_img = dilation(processed_img, footprint=selem)
    ax[1].imshow(processed_img, cmap='gray')
    ax[1].set_title('After Dilation')

    fig.savefig('outputs/morphological_operations_coins.png')
    plt.show()
