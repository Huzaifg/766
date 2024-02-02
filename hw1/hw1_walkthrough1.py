import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from PIL import Image


def hw1_walkthrough1():
    ######################################################################
    # (1) Help and basics
    # In Python, comments are made with the "#" symbol.
    # For help, use the help() function or look up the documentation online.
    # Python statements can be continued to the next line using "\"
    # Python does not display the result unless you use the print() function.
    # Python indexing starts from 0, not 1.

    ######################################################################
    # (2) Objects in Python -- the basic objects in Python are numbers, strings, lists, and dictionaries...
    N = 5  # a scalar
    v = [1, 0, 0]  # a list
    v = [1, 2, 3]  # a list
    v = np.array(v).T  # transpose a list
    v = [i for i in range(1, 4, 1)]  # a list in a specified range
    v = np.pi * np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4]) / 4  # a list in a specified range
    v = []  # empty list

    m = np.array([[1, 2, 3], [4, 5, 6]])  # a matrix
    m = np.zeros((2, 3))  # a matrix of zeros
    v = np.ones((1, 3))  # a matrix of ones
    m = np.eye(3)  # identity matrix
    v = np.random.rand(3, 1)  # random matrix with values in [0,1]

    # To load data from a file, use numpy's np.loadtxt or np.genfromtxt functions.

    v = [1, 2, 3]  # access a list element
    v[2]  # list[index]

    m = np.array([[1, 2, 3], [4, 5, 6]])
    m[0, 2]  # access a matrix element
    m[1, :]  # access a matrix row
    m[:, 0]  # access a matrix column

    m.shape  # size of a matrix
    m.shape[0]  # number rows
    m.shape[1]  # number of columns


    ######################################################################
    #### (3) Simple operations on vectors and matrices

    # (A) Pointwise (element by element) Operations:

    # addition of vectors/matrices and multiplication by a scalar
    # are done "element by element"
    a = np.array([1, 2, 3, 4])  # vector
    b = np.array([5, 6, 7, 8])  # vector
    M = np.array([[1.3, 2.1, 3.6], [4.1, 5.6, 6.2]])  # matrix

    2 * a  # scalar multiplication
    a / 4  # scalar multiplication
    a + b  # pointwise vector addition
    a - b  # pointwise vector subtraction
    a ** 2  # pointwise vector squaring
    a * b  # pointwise vector multiplication
    a / b  # pointwise vector division

    np.log(M)  # pointwise arithmetic operation
    np.round(M)  # pointwise arithmetic operation

    # (B) Vector Operations (no for loops needed)
    # Built-in numpy functions operate on vectors, if a matrix is given,
    # then the function operates on each column of the matrix

    a = np.array([1, 4, 6, 3])  # vector
    np.sum(a)  # sum of vector elements
    np.mean(a)  # mean of vector elements
    np.var(a)  # variance
    np.std(a)  # standard deviation
    np.max(a)  # maximum

    a = np.array([[1, 2, 3], [4, 5, 6]])  # matrix
    a.flatten()  # vectorized version of the matrix
    np.mean(a, axis=0)  # mean of each column
    np.max(a, axis=0)  # max of each column
    np.max(a)  # to obtain max of matrix

    # (C) Matrix Operations:
    a = np.random.rand(3)  # 3 vector
    b = np.random.rand(3)  # 3 vector

    np.dot(a, b)  # dot product or inner product (scalar)
    np.outer(a, b)  # outer product (3x3 matrix)

    a = np.random.rand(3, 2)  # 3x2 matrix
    b = np.random.rand(2, 4)  # 2x4 matrix
    a@b  # Matrix multiplication: 3x4 matrix

    a = np.array([[1, 2], [3, 4], [5, 6]])  # 3 x 2 matrix
    b = np.array([[5, 6, 7]])  # 1 x 3 vector
    np.dot(b, a)      # matrix multiply
    np.dot(a.T, b.T)  # matrix multiply

    ######################################################################
    # (5) Relations and control statements

    # Example: given a vector v, create a new vector with values equal to
    # v if they are greater than 0, and equal to 0 if they less than or
    # equal to 0.

    v = np.array([3, 5, -2, 5, -1, 0])  # 1: FOR LOOPS
    u = np.zeros(v.shape)  # initialize
    for i in range(v.size):  # v.size is the number of elements
        if v[i] > 0:
            u[i] = v[i]
    print(u)

    v = np.array([3, 5, -2, 5, -1, 0])  # 2: NO FOR LOOPS
    u2 = np.zeros(v.shape)  # initialize
    ind = np.where(v > 0)  # index into >0 elements
    u2[ind] = v[ind]
    print(u2)

    v = np.array([3, 5, -2, 5, -1, 0])  # 2: NO FOR LOOPS (binary mask)
    u2 = np.zeros(v.shape)  # initialize
    ind = (v > 0)  # True if v[i]>0, False otherwise
    u2[ind] = v[ind]
    print(u2)

    ######################################################################
    # (6) Creating functions:
    # Functions in python are defined using a `def` statement. For example:
    def threshold(v):
        u = np.zeros(v.shape)  # initialize
        u = v * (v>0).astype(float)
        return u

    v = np.array([3, 5, -2, 5, -1, 0])
    print(threshold(v))  # call from command line


    ######################################################################
    # (7) Plotting
    import matplotlib.pyplot as plt

    x = np.array([0, 1, 2, 3, 4])  # basic plotting
    plt.figure()
    plt.plot(x)
    plt.plot(x, 2*x)
    plt.axis([0, 8, 0, 8])
    plt.show()

    x = np.pi * np.arange(-24, 25) / 24
    plt.figure()
    plt.plot(x, np.sin(x))
    plt.xlabel('radians')
    plt.ylabel('sin value')
    plt.title('dummy')
    plt.text(0, 0, 'Hello World!')  # manually specify the position
    plt.show()

    plt.figure()  # multiple functions in separate graphs
    plt.subplot(1, 2, 1)
    plt.plot(x, np.sin(x))
    plt.axis('square')
    plt.subplot(1, 2, 2)
    plt.plot(x, 2 * np.cos(x))
    plt.axis('square')
    plt.show()

    plt.figure()  # multiple functions in single graph
    plt.plot(x, np.sin(x))
    plt.plot(x, 2 * np.cos(x), '--')  # hold on is not needed in Python
    plt.legend(['sin', 'cos'])
    plt.show()

    plt.figure()  # matrices as images
    m = np.random.rand(64, 64)
    plt.imshow(m, cmap='gray')
    plt.axis('off')
    plt.show()

    ######################################################################
    # (8) Working with images using the Python Imaging Library (PIL)
    from PIL import Image

    # loading an image
    I = Image.open('data/nyc.png')

    # display it
    plt.figure()
    plt.imshow(I)
    plt.axis('off') # turn off axis
    plt.show()

    # convert it to grayscale
    I2 = I.convert('L')

    # scale data to use full colormap for values between 0 and 255
    plt.figure()
    plt.imshow(I2, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.show()

    # Crop the image
    I2 = I.crop((100, 100, 400, 300))  # crop

    # convert cropped image to grayscale
    I2 = I2.convert('L')

    # scale data to use full colormap between min and max values in I2
    plt.figure()
    plt.imshow(I2, cmap='gray')
    plt.colorbar()  # turn on color bar
    plt.show()

    # resize by 50% using bilinear interpolation
    I3 = I2.resize((I2.size[0]//2, I2.size[1]//2), Image.BILINEAR)

    # rotate 45 degrees and crop to original size
    I3 = I2.rotate(45, Image.BILINEAR, 1)  # 1 means expand to fit

    # convert from uint8 PIL image into a numpy array to allow math operations
    I3 = np.array(I3, dtype=float)

    # display squared image (pixel-wise)
    plt.figure()
    plt.imshow(I3**2, cmap='gray')
    plt.show()
