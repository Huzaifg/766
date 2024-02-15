import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

from PIL import Image
import numpy as np


def generateLabeledImage(gray_img: np.ndarray, threshold: float) -> np.ndarray:
    '''
    Generates a labeled image from a grayscale image by assigning unique labels to each connected component.
    Arguments:
        gray_img: grayscale image.
        threshold: threshold for the grayscale image.
    Returns:
        labeled_img: the labeled image.
    '''
    raise NotImplementedError

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
    raise NotImplementedError

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
    raise NotImplementedError


def hw2_challenge1a():
    import matplotlib.cm as cm
    from skimage.color import label2rgb
    from hw2_challenge1 import generateLabeledImage
    img_list = ['two_objects.png', 'many_objects_1.png', 'many_objects_2.png']
    threshold_list = [0.5, 0.5, 0.5]   # You need to find the right thresholds

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
    labeled_two_obj = np.array(labeled_two_obj)
    orig_img = Image.open('data/two_objects.png')
    orig_img = np.array(orig_img.convert('L')) / 255.
    obj_db  = compute2DProperties(orig_img, labeled_two_obj)
    np.save('outputs/obj_db.npy', obj_db)
    print(obj_db)
    
    # TODO: Plot the position and orientation of the objects
    # Use a dot or star to annotate the position and a short line segment originating from the dot for orientation
    # Refer to demoTricksFun.py for examples to draw dots and lines. 
    fig, ax = plt.subplots()
    plt.axis(False)
    ax.imshow(orig_img, cmap='gray')
    # for i in range(obj_db.shape[0]):
        # plot the position
        # plot the orientation
    plt.savefig('outputs/two_objects_properties.png')
    plt.show()

def hw2_challenge1c():
    obj_db = np.load('outputs/obj_db.npy')
    img_list = ['many_objects_1.png', 'many_objects_2.png']

    for i in range(len(img_list)):
        labeled_img = Image.open(f'outputs/labeled_{img_list[i]}')
        labeled_img = np.array(labeled_img)
        orig_img = Image.open(f"data/{img_list[i]}")
        orig_img = np.array(orig_img.convert('L')) / 255.

        recognizeObjects(orig_img, labeled_img, obj_db,
                         f'outputs/testing1c_{img_list[i]}')
