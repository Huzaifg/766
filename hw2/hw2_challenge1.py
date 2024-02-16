import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.patches as patches
matplotlib.use('TkAgg')

from PIL import Image
import numpy as np
from skimage.measure import label


def generateLabeledImage(gray_img: np.ndarray, threshold: float) -> np.ndarray:
    '''
    Generates a labeled image from a grayscale image by assigning unique labels to each connected component.
    Arguments:
        gray_img: grayscale image.
        threshold: threshold for the grayscale image.
    Returns:
        labeled_img: the labeled image.
    '''
    # Convert the image into a binary image by applying a threshold
    bw_img = gray_img > threshold

    # Segment binary image
    labeled_image, num_labels = label(bw_img, background=0, connectivity=1, return_num=True)
    print(f"Number of objects: {num_labels}")
    return labeled_image

    

def compute2DProperties(orig_img: np.ndarray, labeled_img: np.ndarray) ->  np.ndarray:
    '''
    Compute the 2D properties of each object in labeled image.
    Arguments:
        orig_img: the original image.
        labeled_img: the labeled image.
    Returns:
        obj_db: the object database, where each row contains the properties
            of one object.
    '''


    # Find number of objects in the labeled image 
    num_objects = np.max(labeled_img)
    # Initialize the numpy array
    obj_db = np.zeros((num_objects,6))
    # Loop through each object
    for i in range(1, num_objects + 1):
        obj_db[i-1][0] = i #Store the label

        # Find all the pixels that belong to the object
        pixels = np.where(labeled_img == i)
        
        # Calculate the area as the number of pixels belonging to the object
        area = len(pixels[0])

        # Find CG along x and y
        # Since pixes[0] provides the y pixels along which we have bij = 1,
        # we can find the sum of all the pixels and divide by the area to get the CG
        y_cg = int(np.sum(pixels[0]) / area)
        x_cg = int(np.sum(pixels[1]) / area)
        
        # Store row and column position of the center
        obj_db[i-1][1] = y_cg
        obj_db[i-1][2] = x_cg

        # Find second moments
        a = np.sum(np.square(pixels[1] - x_cg))
        b = 2*np.sum((pixels[1] - x_cg) * (pixels[0] - y_cg))
        c = np.sum(np.square(pixels[0] - y_cg))
        # Find theta
        theta = 0.5 * np.arctan2(b, a - c)

        # Evaluate minimum E
        E_min = a*np.sin(theta)**2 - b*np.sin(theta)*np.cos(theta) + c*np.cos(theta)**2

        # Store
        obj_db[i-1][3] = E_min

        # Store theta in degrees
        obj_db[i-1][4] = np.degrees(theta)

        # Evaluate roundness
        theta_max = theta + np.pi/2
        E_max = a*np.sin(theta_max)**2 - b*np.sin(theta_max)*np.cos(theta_max) + c*np.cos(theta_max)**2
        roundness = E_min / E_max
        # store roundness
        obj_db[i-1][5] = roundness
    return obj_db

    

def recognizeObjects(orig_img: np.ndarray, labeled_img: np.ndarray, obj_db: np.ndarray, output_fn: str):
    '''
    Recognize the objects in the labeled image and save recognized objects to output_fn
    Arguments:
        orig_img: the original image.
        labeled_img: the labeled image.
        obj_db: the object database, where each row contains the properties 
            of one object.
        output_fn: filename for saving output image with the objects recognized.
    '''
    fig, ax = plt.subplots()
    plt.axis(False)
    ax.imshow(orig_img, cmap='gray')
    # Get the number of objects from the labelled image
    num_objects = np.max(labeled_img)
    for i in range(1,num_objects+1):
        # Get the object data base for each object in the labelled image that we are testing
        obj_db_lab = compute2DProperties(orig_img, labeled_img)

        # Compare if the roundness matches anything in obj_db
        for j in range(obj_db.shape[0]):
            if np.abs(obj_db_lab[i-1][5] - obj_db[j][5]) < 0.03:
                # plot the position
                box_size = 4
                rect = patches.Rectangle((obj_db_lab[i-1][2] - box_size, obj_db_lab[i-1][1] - box_size), 2*box_size, 2*box_size, linewidth=2, edgecolor='g', facecolor='none')

                ax.add_patch(rect)
                # plot the orientation
                scaling_factor = 20  # Adjust the scaling factor to make the line longer
                x = obj_db_lab[i-1][2] + scaling_factor * np.cos(np.radians(obj_db_lab[i-1][4]))
                y = obj_db_lab[i-1][1] + scaling_factor * np.sin(np.radians(obj_db_lab[i-1][4]))
                ax.plot([obj_db_lab[i-1][2], x], [obj_db_lab[i-1][1], y], 'r-')
    plt.savefig(output_fn)
    plt.show()



def generateImgHist(img_list: list):
    """
    Helper function to plot the histograms and print the 20 lowest bins and 20 highest bins
    Arguments:
        img_list: list of image names
    """
    for i in range(len(img_list)):
        orig_img = Image.open(f"data/{img_list[i]}")
        orig_img = np.array(orig_img.convert('L')) / 255.
        fig, ax = plt.subplots(1, 1)
        ax.hist(orig_img.ravel(), bins=256, range=(0, 1), fc='k', ec='k')
        ax.set_title('Histogram for Img ' + img_list[i])
        ax.set_xlim([0, 1])
        plt.show()
        hist_data, bin_edges = np.histogram(orig_img, bins=256, range=(0, 1))
        sorted_indices = np.argsort(hist_data)
        lowest_bins = bin_edges[sorted_indices[:20]]
        highest_bins = bin_edges[sorted_indices[-20:]]
        print(f"Lowest bins for {img_list[i]}")
        print(lowest_bins)
        print(f"Highest bins for {img_list[i]}")
        print(highest_bins)

    
def hw2_challenge1a():
    import matplotlib.cm as cm
    from skimage.color import label2rgb
    from hw2_challenge1 import generateLabeledImage
    img_list = ['two_objects.png', 'many_objects_1.png', 'many_objects_2.png']
    # Function to plot the histograms and print the 10 lowest bins
    # Then one bin is selected as the threshold by looking at the plots
    # Manually
    generateImgHist(img_list)
    threshold_list = [0.46, 0.46, 0.46]   # You need to find the right thresholds
    
    for i in range(len(img_list)):
        orig_img = Image.open(f"data/{img_list[i]}")
        orig_img = np.array(orig_img.convert('L')) / 255.
        labeled_img = generateLabeledImage(orig_img, threshold_list[i])
        Image.fromarray(labeled_img.astype(np.uint8)).save(
            f'outputs/labeled_{img_list[i]}')
        
        cmap = np.array(cm.get_cmap('Set1').colors)
        rgb_img = label2rgb(labeled_img, colors=cmap, bg_label=0)
        Image.fromarray((rgb_img * 255).astype(np.uint8)).save(
            f'outputs/rgb_labeled_{img_list[i]}')

def hw2_challenge1b():
    labeled_two_obj = Image.open('outputs/labeled_two_objects.png')
    # labeled_two_obj = Image.open('outputs/labeled_many_objects_1.png')

    labeled_two_obj = np.array(labeled_two_obj)
    orig_img = Image.open('data/two_objects.png')
    # orig_img = Image.open('data/many_objects_1.png')

    orig_img = np.array(orig_img.convert('L')) / 255.
    obj_db  = compute2DProperties(orig_img, labeled_two_obj)
    np.save('outputs/obj_db.npy', obj_db)
    # np.save('outputs/obj_db_many_objects_1.npy', obj_db)
    print(obj_db)

    fig, ax = plt.subplots()
    plt.axis(False)
    ax.imshow(orig_img, cmap='gray')
    print(obj_db.shape[0])
    for i in range(obj_db.shape[0]):
        box_size = 4
        # plot the position
        rect = patches.Rectangle((obj_db[i][2] - box_size, obj_db[i][1] - box_size), 2*box_size, 2*box_size, linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
        # plot the orientation
        scaling_factor = 20  # Adjust the scaling factor to make the line longer
        x = obj_db[i][2] + scaling_factor * np.cos(np.radians(obj_db[i][4]))
        y = obj_db[i][1] + scaling_factor * np.sin(np.radians(obj_db[i][4]))
        ax.plot([obj_db[i][2], x], [obj_db[i][1], y], 'r-')


    plt.savefig('outputs/two_objects_properties.png')
    # plt.savefig('outputs/many_objects_1_properties.png')
    plt.show()


def hw2_challenge1c():
    # two_object database
    obj_db = np.load('outputs/obj_db.npy')
    img_list = ['many_objects_1.png', 'many_objects_2.png']

    for i in range(len(img_list)):
        labeled_img = Image.open(f'outputs/labeled_{img_list[i]}')
        labeled_img = np.array(labeled_img)
        orig_img = Image.open(f"data/{img_list[i]}")
        orig_img = np.array(orig_img.convert('L')) / 255.

        recognizeObjects(orig_img, labeled_img, obj_db,
                         f'outputs/testing1c_{img_list[i]}')
        
    # Now load the many_objects_1 database
    obj_db = np.load('outputs/obj_db_many_objects_1.npy')
    # Other images
    img_list = ['two_objects.png', 'many_objects_2.png']
    for i in range(len(img_list)):
        labeled_img = Image.open(f'outputs/labeled_{img_list[i]}')
        labeled_img = np.array(labeled_img)
        orig_img = Image.open(f"data/{img_list[i]}")
        orig_img = np.array(orig_img.convert('L')) / 255.

        recognizeObjects(orig_img, labeled_img, obj_db,
                         f'outputs/testing1cAdd_{img_list[i]}')
    
