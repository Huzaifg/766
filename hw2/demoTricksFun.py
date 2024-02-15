import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import matplotlib.cm as cm
from skimage.color import label2rgb
from skimage.color import rgb2gray

def demo():
    # This demo shows you the following tricks:
    # - how to 'color' a labeled image to make the labeled image easier to see,
    # - how to use plot and line to annotate an image,
    # - how to save an annotated image.
    #
    # When you use plot or add a patch to annotate a displayed image, the
    # annotations "float" on top of the displayed image. To save both the
    # annotations and the displayed image into a new image, use the trick shown
    # here.

    # Open a new figure
    fig, ax = plt.subplots(3, 1)

    # Load the image
    threeboxes = Image.open('data/labeled_three_boxes.png')
    ax[0].imshow(threeboxes, cmap='gray')

    # 'color' the labeled image
    colors = np.array(cm.get_cmap('Set1').colors)
    rgb_img = label2rgb(np.array(threeboxes), image=None, colors=colors, bg_label=0, bg_color=(0,0,0))

    # Display the image
    ax[1].imshow(rgb_img)

    # Now let's try annotate the image
    # Convert the image to grayscale
    gray_img = rgb2gray(rgb_img)
    ax[2].imshow(gray_img, cmap='gray')

    # Draw dots on the image
    dots = [[100,80],[200,80],[300,80]]
    for dot in dots:
        ax[2].plot(dot[0], dot[1], 'rs', markerfacecolor='w')

    # Draw four lines on the image
    loopxy = [[31, 130], [31, 31], [390, 31], [390, 130], [31, 130]]
    for i in range(1, 5):
        ax[2].add_patch(patches.Polygon(loopxy[i-1:i+1], closed=None, fill=None, edgecolor='g', linewidth=2))

    # Save the figure with the annotations
    plt.tight_layout()
    fig.savefig('outputs/annotated_img.png', pad_inches=0)

    # Load the saved image to check
    annotated_img = Image.open('outputs/annotated_img.png')
    annotated_img.show()

    # Close the figure
    plt.close(fig)
