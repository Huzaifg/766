from PIL import Image, ImageDraw
import numpy as np

def apply_threshold_and_non_maximum_suppression(hough_img, hough_threshold):
    # Thresholding
    strong_hough_img = np.where(hough_img >= hough_threshold, hough_img, 0)

    # Non-maximum suppression
    rho_bins, theta_bins = strong_hough_img.shape
    for i in range(rho_bins):
        for j in range(theta_bins):
            cur_val = strong_hough_img[i, j]
            if cur_val == 0:  # Skip if already zero 
                continue

            # Create neighborhood window 
            a_vals = np.arange(-7, 8) 
            b_vals = np.arange(-7, 8) 
            a_grid, b_grid = np.meshgrid(a_vals, b_vals)

            # Apply boundary conditions and skip center element 
            mask = (
                (i + a_grid < rho_bins) & 
                (j + b_grid < theta_bins) & 
                (i + a_grid >= 0) & 
                (j + b_grid >= 0) & 
                (a_grid != 0) & 
                (b_grid != 0)
            )

            # Check for larger values in the neighborhood
            if np.any(strong_hough_img[i + a_grid[mask], j + b_grid[mask]] > cur_val):
                strong_hough_img[i, j] = 0

    return strong_hough_img

def generateHoughAccumulator(edge_image: np.ndarray, theta_num_bins: int, rho_num_bins: int) -> np.ndarray:
    '''
    Generate the Hough accumulator array.
    Arguments:
        edge_image: the edge image.
        theta_num_bins: the number of bins in the theta dimension.
        rho_num_bins: the number of bins in the rho dimension.
    Returns:
        hough_accumulator: the Hough accumulator array.
    '''
 
    H, W = edge_image.shape
    hough_img = np.zeros((rho_num_bins, theta_num_bins))
 
    # Coordinate system centered at the image center
    y_edge, x_edge = np.nonzero(edge_image)
    
    # Calculate rho and theta for the edge pixels
    diagonal = np.floor(np.hypot(H, W))
    theta_vals = np.linspace(0, np.pi, theta_num_bins)
 
    for i in range(len(x_edge)):
        x = x_edge[i]
        y = y_edge[i]
 
        # Calculate rho values for each theta for this particular edge (x, y)
        rhos = x * np.cos(theta_vals) + y * np.sin(theta_vals)
 
        # Map rho to the index 
        # Thanks ChatGPT
        rho_indices = np.floor((rhos + diagonal) / (2 * diagonal) * (rho_num_bins)).astype(int)
 
        for j in range(theta_num_bins):
            rho_index = rho_indices[j]
            hough_img[rho_index, j] += 1
    
    hough_img = hough_img / np.max(hough_img) * 255
    return hough_img

def suppress_neigh(hough_space, neighborhood_dim=10):
    # Create a copy to avoid modifying the original image
    suppressed_space = np.copy(hough_space)
    h_space_height, h_space_width = hough_space.shape
 
    # Define half the size for neighborhood iteration
    half_dim = neighborhood_dim // 2
 
    for h_rho in range(half_dim, h_space_height - half_dim):
        for h_theta in range(half_dim, h_space_width - half_dim):
            # Find the max value in the neighborhood
            local_peak = np.amax(hough_space[h_rho-half_dim:h_rho+half_dim+1, h_theta-half_dim:h_theta+half_dim+1])
            # Suppress non-maximum values
            if hough_space[h_rho, h_theta] != local_peak:
                suppressed_space[h_rho, h_theta] = 0
    return suppressed_space
 
def lineFinder(image_original: np.ndarray, hough_space: np.ndarray, threshold_hough: float):
    '''
    Find the lines in the image using the Hough transform.
    Arguments:
        image_original: the original image as a NumPy array.
        hough_space: the Hough image as a NumPy array.
        threshold_hough: the threshold for the Hough accumulator array.
    Returns: 
        image_with_lines: PIL image with lines drawn.
    '''
    num_rho, num_theta = hough_space.shape
    diag_length = np.floor(np.hypot(image_original.shape[0], image_original.shape[1]))
 

    values_rho = np.linspace(-diag_length, diag_length, num_rho)
    values_theta = np.linspace(0, np.pi, num_theta)
 
    # Suppress competing lines
    hough_suppressed = suppress_neigh(hough_space, neighborhood_dim=15)
 

    hough_strong = np.where(hough_suppressed >= threshold_hough, hough_suppressed, 0)

    image_with_lines = Image.fromarray(image_original.astype(np.uint8)).convert('RGB')
    drawer = ImageDraw.Draw(image_with_lines)
 
    # Thanks ChatGPT
    for idx_rho, idx_theta in np.argwhere(hough_strong):
        rho_val = values_rho[idx_rho]
        theta_val = values_theta[idx_theta]
        cos_theta = np.cos(theta_val)
        sin_theta = np.sin(theta_val)
        x_center = cos_theta * rho_val
        y_center = sin_theta * rho_val
        point1 = (int(x_center + 1000*(-sin_theta)), int(y_center + 1000*(cos_theta)))
        point2 = (int(x_center - 1000*(-sin_theta)), int(y_center - 1000*(cos_theta)))
        drawer.line([point1, point2], fill=(255, 0, 0), width=2)
 
    return image_with_lines



def lineSegmentFinder(orig_img: np.ndarray, edge_img: np.ndarray, hough_img: np.ndarray, hough_threshold: float):
    '''
    Find the line segments in the image.
    Arguments:
        orig_img: the original image.
        edge_img: the edge image.
        hough_img: the Hough image.
        hough_threshold: the threshold for the Hough accumulator array.
    Returns:
        line_segement_img: PIL image with line segments drawn.
    '''

    # Lets just do lineFinder and get the lines
    num_rho, num_theta = hough_img.shape
    diag_length = np.floor(np.hypot(orig_img.shape[0], orig_img.shape[1]))
 

    values_rho = np.linspace(-diag_length, diag_length, num_rho)
    values_theta = np.linspace(0, np.pi, num_theta)
 
    # Suppress competing lines
    hough_suppressed = suppress_neigh(hough_img, neighborhood_dim=15)
 

    hough_strong = np.where(hough_suppressed >= hough_threshold, hough_suppressed, 0)

    image_with_lines = Image.fromarray(orig_img.astype(np.uint8)).convert('RGB')
    drawer = ImageDraw.Draw(image_with_lines)
    
    point_list = []
    # Thanks ChatGPT
    for idx_rho, idx_theta in np.argwhere(hough_strong):
        rho_val = values_rho[idx_rho]
        theta_val = values_theta[idx_theta]
        cos_theta = np.cos(theta_val)
        sin_theta = np.sin(theta_val)
        x_center = cos_theta * rho_val
        y_center = sin_theta * rho_val
        point1 = (int(x_center + 1000*(-sin_theta)), int(y_center + 1000*(cos_theta)))
        point2 = (int(x_center - 1000*(-sin_theta)), int(y_center - 1000*(cos_theta)))

        # This time however store all the points
        point_list.append((point1, point2))
    

    # Now keep moving along these lines and when the edge_img pixel becomes 0, stop and draw the line
    for point1, point2 in point_list:

        # Get the slope and intercept
        m = np.inf if point2[0] == point1[0] else (point2[1] - point1[1]) / (point2[0] - point1[0])
        c = point1[1] - m * point1[0] if m != np.inf else point1[0]
        
        # Now move along the line and check the edge_img pixel
        edge_point_counter = 0
        first_line_drawn = False
        prev_pt = None
        prev_point_corrupted = False
        for x in range(point1[0], point2[0]):
            #make sure that x is within the image
            if x < 0 or x >= edge_img.shape[1]:
                continue
            y = int(m*x + c)
            #make sure that y is within the image
            if y < 0 or y >= edge_img.shape[0]:
                continue


            cur_pt = (x, y)

            # Check if the current point is on the edge
            if (edge_img[y, x] != 0) and (cur_pt != None):
                # if this is the first point on the edg, don't do anything
                # Just initialize the prev_pt
                if(edge_point_counter == 0):
                    prev_pt = (x, y)
                    edge_point_counter += 1

                # Draw line from last point to current point if previous point was not corrupted - if the previous point was corrupted that means that if we 
                # draw a line from the previous point to the current point, it will be a line that is not on the edge
                if not prev_point_corrupted:
                    first_line_drawn = True
                    drawer.line([prev_pt, cur_pt], fill=(255, 0, 0), width=2)
                    prev_pt = cur_pt
                # If we have entered here again, it means we are back on the edge,
                # so uncorrput previous point and set it to current point
                prev_point_corrupted = False
                prev_pt = cur_pt
            # This means we are not on an edge any more, that means drawing a line from the previous point will make us draw a line that is not on the edge
            else:
                # We only corrupt the prev point if the first line is drawn
                if first_line_drawn:
                    prev_point_corrupted = True
                    continue

    
    return image_with_lines


    
    

