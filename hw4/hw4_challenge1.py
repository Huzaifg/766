from PIL import Image
import numpy as np
from typing import Union, Tuple, List
from PIL import ImageDraw
from math import floor
from scipy import ndimage
from helpers import genSIFTMatches
import matplotlib.pyplot as plt

def computeHomography(src_pts_nx2: np.ndarray, dest_pts_nx2: np.ndarray) -> np.ndarray:
    '''
    Compute the homography matrix.
    Arguments:
        src_pts_nx2: the coordinates of the source points (nx2 numpy array).
        dest_pts_nx2: the coordinates of the destination points (nx2 numpy array).
    Returns:
        H_3x3: the homography matrix (3x3 numpy array).
    '''
    # First we generate matrix A
    A = np.zeros((2*src_pts_nx2.shape[0], 9))

    for i in range(src_pts_nx2.shape[0]):
        xs, ys = src_pts_nx2[i]
        xd, yd = dest_pts_nx2[i]
        A[2*i] = np.array([xs, ys, 1, 0, 0, 0, -xd*xs, -ys*xd, -xd])
        A[2*i+1] = np.array([0, 0, 0, xs, ys, 1, -xs*yd, -yd*ys, -yd])

    #  Homography ends up being the equivalent of solving an eigen value problem
    eigen_values, eigen_vectors = np.linalg.eig(A.T @ A)

    # Find eigen_vector corresponding to the smallest eigen_value
    H_3x3 = eigen_vectors[:, np.argmin(eigen_values)].reshape(3, 3)

    return H_3x3


def applyHomography(H_3x3: np.ndarray, src_pts_nx2: np.ndarray) ->  np.ndarray:
    '''
    Apply the homography matrix to the source points.
    Arguments:
        H_3x3: the homography matrix (3x3 numpy array).
        src_pts_nx2: the coordinates of the source points (nx2 numpy array).
    Returns:
        dest_pts_nx2: the coordinates of the destination points (nx2 numpy array).
    '''
    # Add a column of ones to the source points
    src_pts_nx3 = np.hstack((src_pts_nx2, np.ones((src_pts_nx2.shape[0], 1))))

    # Apply the homography
    dest_pts_nx3 = src_pts_nx3 @ H_3x3.T

    # Normalize the coordinates
    dest_pts_nx2 = dest_pts_nx3[:, :2] / dest_pts_nx3[:, 2].reshape(-1, 1)

    return dest_pts_nx2



def showCorrespondence(img1: Image.Image, img2: Image.Image, pts1_nx2: np.ndarray, pts2_nx2: np.ndarray) -> Image.Image:
    '''
    Show the correspondences between the two images.
    Arguments:
        img1: the first image.
        img2: the second image.
        pts1_nx2: the coordinates of the points in the first image (nx2 numpy array).
        pts2_nx2: the coordinates of the points in the second image (nx2 numpy array).
    Returns:
        result: image depicting the correspondences.
    '''
    
    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]

    # Add a white space gap
    middle_space = 0
    width = width1 + middle_space + width2
    height = max(height1, height2)

    # What offset is the source image at
    img1_offset = (0, 0)
    img2_offset = (width1 + middle_space, floor((height - height2) / 2))
    # Create a new image with the combined width
    combined_img = Image.new('RGB', (width, height))
    combined_img.paste(Image.fromarray(img1), img1_offset)
    combined_img.paste(Image.fromarray(img2), img2_offset)

    # Transalation required to match 0,0
    trans_x = img2_offset[0] - img1_offset[0]
    trans_y = img2_offset[1] - img1_offset[1]
    draw = ImageDraw.Draw(combined_img)
    for i in range(len(pts1_nx2)):
        pt1 = pts1_nx2[i]
        pt2 = pts2_nx2[i]    
        pt2 = (pt2[0] + trans_x, pt2[1] + trans_y)  # Adjust for placement of second image
        draw.line((pt1[0], pt1[1], pt2[0], pt2[1]), fill='red', width=2)

    return combined_img



# function [mask, result_img] = backwardWarpImg(src_img, resultToSrc_H, dest_canvas_width_height)

def backwardWarpImg(src_img: Image.Image, destToSrc_H: np.ndarray, canvas_shape: Union[Tuple, List]) -> Tuple[Image.Image, Image.Image]:
    '''
    Backward warp the source image to the destination canvas based on the
    homography given by destToSrc_H. 
    Arguments:
        src_img: the source image.
        destToSrc_H: the homography that maps points from the destination
            canvas to the source image.
        canvas_shape: shape of the destination canvas (height, width).
    Returns:
        dest_img: the warped source image.
        dest_mask: a mask indicating sourced pixels. pixels within the
            source image are 1, pixels outside are 0.
    '''
    
    # Apply inverse homography to the destination canvas
    dest_img = np.zeros((canvas_shape[0], canvas_shape[1], 3))
    print(dest_img.shape)
    dest_mask = np.zeros((canvas_shape[0], canvas_shape[1]), dtype=np.bool_)

    for i in range(canvas_shape[0]):
        for j in range(canvas_shape[1]):
            # Apply inverse homography
            src_x, src_y, src_w = destToSrc_H @ np.array([j, i, 1])
            src_x /= src_w
            src_y /= src_w

            src_x = floor(src_x)
            src_y = floor(src_y)



            # Check if all 4 points are within the source image

            if src_x < 0 or src_x >= src_img.shape[1] or src_y < 0 or src_y >= src_img.shape[0]:
                continue
            else:
                dest_img[i, j,:] = src_img[src_y, src_x,:]
                dest_mask[i, j] = True
    
    return dest_mask, dest_img 



def blendImagePair(img1: np.ndarray, mask1: np.ndarray, img2: np.ndarray, mask2: np.ndarray, mode: str) -> np.ndarray:
    binary_mask1 = mask1.astype(bool)
    binary_mask2 = mask2.astype(bool)
    
    if mode == "overlay":
        # Where mask2 is True, replace img1's pixels with img2's
        result = np.where(binary_mask2[..., None], img2, img1)
    elif mode == "blend":
        weight_s = ndimage.distance_transform_edt(binary_mask1)
        weight_d = ndimage.distance_transform_edt(binary_mask2)

        result = (img1 * weight_s[..., None] + img2 * weight_d[..., None]) / (weight_s[...,None] + weight_d[..., None] + 1e-6)
        return result
    else:
        raise ValueError("Unsupported blending mode. Choose 'overlay' or 'blend'.")

    return result.astype(img1.dtype)


def runRANSAC(src_pt: np.ndarray, dest_pt: np.ndarray, ransac_n: int, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Run the RANSAC algorithm to find the inliers between the source and
    destination points.
    Arguments:
        src_pt: the coordinates of the source points (nx2 numpy array).
        dest_pt: the coordinates of the destination points (nx2 numpy array).
        ransac_n: the number of iterations to run RANSAC.
        eps: the threshold for considering a point to be an inlier.
    Returns:
        inliers_id: the indices of the inliers (kx1 numpy array).
        H: the homography matrix (3x3 numpy array).
    '''
    # Initialize H
    best_H = np.zeros((3, 3))
    best_len = 0
    best_inliers_id = np.array([])
    for i in range(ransac_n):
        # Randomly Sample 4 indicies to sample from src_pt and dest_pt
        ind_rn = np.random.choice(src_pt.shape[0], 4, replace=False)

        # Form smaller list of points
        src_pt_rn = src_pt[ind_rn]
        dest_pt_rn = dest_pt[ind_rn]

        # Compute homography
        H = computeHomography(src_pt_rn, dest_pt_rn)

        # Apply homography
        dest_pt_est = applyHomography(H, src_pt)

        # Compute distance
        dist = np.sqrt(np.sum((dest_pt - dest_pt_est)**2, axis=1))

        # Find inliers
        inliers_id = np.where(dist < eps)[0]
        lent = len(inliers_id)

        # Check if this is the best homography
        if lent > best_len:
            best_len = lent
            best_inliers_id = inliers_id
            best_H = H


    return best_inliers_id, best_H


def compute_bounding_box(img1, img2, H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Corners of image 1
    img1_corners = np.array([[0, 0], [w1, 0], [w1, h1], [0, h1]])

    # Transform corners of image 1 using the homography
    warped_img1_corners = applyHomography(H, img1_corners)

    # Corners of image 2
    img2_corners = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]])

    # Find min/max x and y across all corners
    all_corners = np.vstack((warped_img1_corners, img2_corners))  
    min_x, min_y = np.min(all_corners, axis=0)
    max_x, max_y = np.max(all_corners, axis=0)

    return np.array([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]])

def stitchImg(*args: np.ndarray) -> np.ndarray:
    '''
    Stitch a list of images.
    Arguments:
        args: a variable number of input images.
    Returns:
        stitched_img: the stitched image.
    '''
    base_img = args[0]  # Start with the first image as the base

    for i in range(1, len(args)):
        img_to_stitch = args[i]

        # 1. Find Corresponding Keypoints
        src_pts, dest_pts = genSIFTMatches(np.array(base_img), np.array(img_to_stitch))

        # 2. RANSAC to Estimate Homography
        inliers_id, H = runRANSAC(src_pts, dest_pts, ransac_n=20, eps=0.5)

        # 3. Compute Bounding Box with Negative Coordinates
        # corners = compute_bounding_box(base_img, img_to_stitch, H)
        # translation_offset = (min(corners[:, 0]) * (-1), min(corners[:, 1]) * (-1))

        # # 4. Adjust Homography with Translation
        # H_adjusted = np.eye(3)  # Identity matrix
        # H_adjusted[:2, 2] = translation_offset
        # H = H_adjusted @ H  # Modify the original homography

        
        # 5. Warp and Blend
        mask, warped = backwardWarpImg(img_to_stitch, np.linalg.inv(H), base_img.shape[:2])
        base_img = blendImagePair(base_img, mask, warped, np.zeros_like(mask), mode='blend')

    return base_img
    