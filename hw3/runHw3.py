import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from skimage import filters, feature
from PIL import Image
import argparse
import numpy as np
from runTests import run_tests


def runHw3():
    # runHw3 is the "main" interface that lets you execute all the 
    # walkthroughs and challenges in this homework. It lists a set of 
    # functions corresponding to the problems that need to be solved.
    #
    # Note that this file also serves as the specifications for the functions 
    # you are asked to implement. In some cases, your submissions will be 
    # auto-graded.  Thus, it is critical that you adhere to all the specified 
    # function signatures.
    #
    # Before your submission, make sure you can run runHw3('all') 
    # without any error.
    #
    # Usage:
    # python runHw3.py                  : list all the registered functions
    # python runHw3.py 'function_name'  : execute a specific test
    # python runHw3.py 'all'            : execute all the registered functions
    parser = argparse.ArgumentParser(
        description='Execute a specific test or all tests.')
    parser.add_argument(
        'function_name', type=str, nargs='?', default='all',
        help='Name of the function to test or "all" to execute all the registered functions')
    args = parser.parse_args()

    # Call test harness
    fun_handles = {'honesty': honesty, 
                   'walkthrough1': walkthrough1,
                   'challenge1a': challenge1a,
                   'challenge1b': challenge1b,
                   'challenge1c': challenge1c,
                   'challenge1d': challenge1d}
    run_tests(args.function_name, fun_handles)


# Academic Honesty Policy
def honesty():
    from signAcademicHonestyPolicy import sign_academic_honesty_policy
    # Type your full name and uni (both in string) to state your agreement 
    # to the Code of Academic Integrity.
    sign_academic_honesty_policy('Peter Parker', 'pp117')


# Test for Walkthrough 1: Image processing
def walkthrough1():
    from hw3_walkthrough1 import hw3_walkthrough1
    hw3_walkthrough1()


# Tests for Challenge 1: Hough transform
def challenge1a():
    img_list = ['hough_1.png', 'hough_2.png', 'hough_3.png']
    for i, fn in enumerate(img_list):
        img = Image.open(f"data/{fn}")
        gray_img = img.convert('L')
        gray_img = np.array(gray_img)

        # Apply edge detection to grayscale image
        thresh = 0.035
        edge_img = filters.sobel(gray_img) > thresh
        # edge_img = feature.canny(gray_img, sigma=1, low_threshold=0.1, high_threshold=0.2)

        # Save the edge detected image
        edge_img = Image.fromarray((edge_img * 255).astype(np.uint8))
        edge_img.save(f'outputs/edge_{fn}')

def challenge1b():
    from hw3_challenge1 import generateHoughAccumulator

    img_list = ['hough_1.png', 'hough_2.png', 'hough_3.png']

    # rb = 768 * 2
    rb = 1000
    tb = 1000

    for i, fn in enumerate(img_list):
        # Load the edge image from challenge1a
        img = Image.open(f'outputs/edge_{fn}')
        img = np.array(img.convert('L'))  # Convert the image to grayscale
        hough_accumulator = generateHoughAccumulator(img, tb, rb)
        hough_accumulator = Image.fromarray(hough_accumulator.astype(np.uint8))
        hough_accumulator.save(f'outputs/accumulator_{fn}')


def challenge1c():
    from hw3_challenge1 import lineFinder

    img_list = ['hough_1.png', 'hough_2.png', 'hough_3.png']
    # img_list = ['hough_1.png']

    # hough_threshold = [250, 250, 250]
    hough_threshold = [90, 160, 130]

    for i, fn in enumerate(img_list):
        orig_img = Image.open(f"data/{fn}")
        orig_img = np.array(orig_img.convert('L'))  

        hough_img = Image.open(f'outputs/accumulator_{fn}')
        hough_img = np.array(hough_img.convert('L'))

        line_img = lineFinder(orig_img, hough_img, hough_threshold[i])
        line_img.save(f'outputs/line_{fn}')

def challenge1d():
    from hw3_challenge1 import lineSegmentFinder
    img_list = ['hough_1.png', 'hough_2.png', 'hough_3.png']

    hough_threshold = [90, 160, 130]

    for i, fn in enumerate(img_list):
        orig_img = Image.open(f"data/{fn}")
        orig_img = np.array(orig_img.convert('L'))  # Convert the image to grayscale

        edge_img = Image.open(f'outputs/edge_{fn}')
        edge_img = np.array(edge_img.convert('L'))  # Convert the image to grayscale
        
        hough_img = Image.open(f'outputs/accumulator_{fn}')
        hough_img = np.array(hough_img.convert('L'))  # Convert the image to grayscale

        line_segement_img = lineSegmentFinder(orig_img, edge_img, hough_img, hough_threshold[i])
        line_segement_img.save(f'outputs/croppedline_{fn}')

if __name__ == '__main__':
    runHw3()