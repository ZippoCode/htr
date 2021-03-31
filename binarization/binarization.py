import cv2
import sys
import numpy as np
import operator
import matplotlib.pyplot as plt
from skimage.filters import *

from scipy.spatial import ConvexHull
from scipy.interpolate import griddata, NearestNDInterpolator
from numpy import linalg as LNG


def grayscale_conversion(image: np.ndarray, num_clusters=2):
    """
    Convert an image in BGR color space into a grayscale image
    :param
        image: numpy.ndarray with shape (H, W, C)
            Image into BGR color space
        num_clusters: int (default = 2)
            Number of clusters used for K-Means
    :return:
        numpy.ndarray with shape (H, W)
            Image into grayscale image
    """
    if len(image.shape) != 3:
        sys.exit('[ERROR]')

    # Convert RGB image into YPQ
    image = np.float32(image)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Swap BGR to RGB

    for i in range(image_rgb.shape[2]):
        image_rgb[:, :, i] *= 1.0 / image_rgb[:, :, i].max()
    image_r = image_rgb.reshape((image_rgb.shape[0] * image_rgb.shape[1], 3))
    matrix = np.array([[0.2989, 0.5870, 0.1140],
                       [0.5000, 0.5000, -1.0000],
                       [1.0000, -1.0000, 00000]], dtype=np.float32)
    image_lpq = np.matmul(image_r, matrix.T).reshape(image_rgb.shape)

    # Find bigger cluster (K-MEANS)
    points = image_lpq.reshape(-1, 3)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, cluster_centers = cv2.kmeans(points, num_clusters, None, criteria, 10, flags)
    unique, counts = np.unique(labels, return_counts=True)
    dict_labels = dict(zip(unique, counts))
    label_bigger_cluster, value_bigger_cluster = max(dict_labels.items(), key=operator.itemgetter(1))
    y_b, p_b, q_b = cluster_centers[label_bigger_cluster]
    y, p, q = np.transpose(image_lpq, axes=[2, 0, 1])

    # Revert value 0 (white) and 1 (black)
    grayscale_image = np.sqrt((y - y_b) ** 2 + (p - p_b) ** 2 + (q - q_b) ** 2)
    grayscale_image = np.abs(grayscale_image - grayscale_image.max()) + grayscale_image.min()
    grayscale_image *= 1.0 / grayscale_image.max()
    return grayscale_image


def lowlights_map(image: np.ndarray, alpha_0=0.1, alpha_1=1, gamma=0.001):
    # IMAGE TO POINT-SET TRANSFORMATION (STAGE 1)
    x = (np.arange(image.shape[1]) - image.shape[1] / 2) / max(image.shape)
    y = (np.arange(image.shape[0]) - image.shape[0] / 2) / max(image.shape)
    x, y = np.meshgrid(x, y)
    z = 1

    # Equation
    r = alpha_1 * image + alpha_0
    theta = np.arctan2(y, x)
    phi = np.arctan2(np.sqrt(x ** 2 + y ** 2), z)
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    # DETECTION VISIBLE/OCCLUDING POINTS (STAGE 2)
    transformation_points = np.dstack([x, y, z])
    norm = LNG.norm(transformation_points, axis=2, keepdims=False)
    for i in range(transformation_points.shape[2]):  # Point transformation
        transformation_points[:, :, i] = (transformation_points[:, :, i] / norm) * np.power(norm, gamma)

    # Convex Hull Constructor
    # Create set of point with viewpoint
    pts = transformation_points.reshape(-1, 3)
    c = np.zeros(shape=(1, 3))
    pts = np.concatenate((c, pts))
    hull = ConvexHull(pts)
    values_vp = pts[hull.simplices[:, 0]]
    print(f"Found {len(values_vp)} visible points")

    # Make smoothed image
    smoothed_image = np.zeros(image.shape, dtype=np.float32)
    mask = np.isin(transformation_points.reshape(-1, 3), values_vp)
    ax_one = mask[:, 0].reshape(image.shape)
    ax_two = mask[:, 1].reshape(image.shape)
    ax_three = mask[:, 2].reshape(image.shape)
    mask_3d = np.dstack((ax_one, ax_two, ax_three))
    coordinates = np.where(np.all(mask_3d, axis=2))
    smoothed_image[coordinates[0], coordinates[1]] = image[coordinates[0], coordinates[1]]

    # Nearest ND Interpolate for not visible point
    mask = smoothed_image != 0
    xx, yy = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    xy_points = np.vstack((np.ravel(xx[mask]), np.ravel(yy[mask]))).T
    data_values = np.ravel(smoothed_image[mask])
    grid = griddata(xy_points, data_values, (np.ravel(xx), np.ravel(yy)), method='linear')
    grid = grid.reshape(xx.shape)
    grid[np.isnan(grid)] = 1.0
    zero_coordinates = np.where(smoothed_image == 0)
    smoothed_image[zero_coordinates[0], zero_coordinates[1]] = grid[zero_coordinates[0], zero_coordinates[1]]

    # Last step - Normalization
    final = image - smoothed_image
    final = cv2.normalize(src=final, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return final


def binarization_blackbox(image: np.ndarray):
    if len(image.shape) > 2:
        return

    image = cv2.medianBlur(image.astype(np.float32), ksize=3)  # Noise Removal

    # First
    local_thresh = image > threshold_local(image, block_size=35, offset=5)
    sauvola_thresh = image > threshold_sauvola(image, window_size=15, k=0.1, r=20)
    niblack_thresh = image > threshold_niblack(image, window_size=51, k=0.5)
    # print(np.count_nonzero(local_thresh), np.count_nonzero(sauvola_thresh), np.count_nonzero(niblack_thresh))
    or_image = np.logical_or(np.logical_or(local_thresh, sauvola_thresh), niblack_thresh)
    invert_or_image = np.bitwise_not(or_image)

    # Plot Threshold Image
    # fig, axs = plt.subplots(2, 2)
    # axs[0, 0].imshow(image, cmap='gray')
    # axs[0, 1].imshow(local_thresh, cmap='gray')
    # axs[1, 0].imshow(sauvola_thresh, cmap='gray')
    # axs[1, 1].imshow(niblack_thresh, cmap='gray')
    # plt.show()
    # plt.close(fig)

    # Second
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    grad_x = cv2.Sobel(image, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(image, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    localization_image = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    mean = localization_image.mean()
    localization_image[localization_image > (mean + 50)] = 255
    localization_image[localization_image <= (mean + 50)] = 0

    # ss = sobel(image)
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.hist(np.ravel(localization_image), 256, [0, 256])
    # ax2.hist(np.ravel(ss), 256, [0, 256])
    # plt.show()
    # localization_image = sobel_image.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    dilation = cv2.dilate(localization_image, kernel, iterations=2)

    # Plot EDGE DETECTION AND DILATION
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    # ax1.imshow(image, cmap='gray')
    # ax2.imshow(localization_image, cmap='gray')
    # ax3.imshow(dilation, cmap='gray')
    # plt.show()
    # plt.close(fig)

    # Combine
    final = dilation * invert_or_image
    final = 255 - final

    # fig, axs = plt.subplots(2, 2)
    # axs[0, 0].imshow(image, cmap='gray')
    # axs[0, 1].imshow(or_image, cmap='gray')
    # axs[1, 0].imshow(dilation, cmap='gray')
    # axs[1, 1].imshow(final, cmap='gray')
    # plt.show()
    # plt.close(fig)

    return final


def quadrant_binarization(image):
    image_top_left = lowlights_map(image[0:image.shape[0] // 2, 0:image.shape[1] // 2])
    image_top_right = lowlights_map(image[0:image.shape[0] // 2, image.shape[1] // 2:image.shape[1]])
    image_bottom_left = lowlights_map(image[image.shape[0] // 2:image.shape[0], 0:image.shape[1] // 2])
    image_bottom_right = lowlights_map(image[image.shape[0] // 2:image.shape[0], image.shape[1] // 2:image.shape[1]])
    result = np.zeros(image.shape)
    result[0:image.shape[0] // 2, 0:image.shape[1] // 2] = image_top_left
    result[0:image.shape[0] // 2, image.shape[1] // 2:image.shape[1]] = image_top_right
    result[image.shape[0] // 2:image.shape[0], 0:image.shape[1] // 2] = image_bottom_left
    result[image.shape[0] // 2:image.shape[0], image.shape[1] // 2:image.shape[1]] = image_bottom_right
    return result
