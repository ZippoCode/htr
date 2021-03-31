import cv2
import os
import sys
import random
import numpy as np


def upload_files(source_path: str):
    """
        Return a list of image path
    :param source_path:  The folder which contains the files
    :return: list<str>
    """
    if not os.path.exists(source_path):
        sys.exit(f"[ERROR] Path \"{source_path}\" not found!")
    images = list()
    for file in os.listdir(path=source_path):
        if file.endswith('.png') or file.endswith('.jpg') or file.endswith('bmp'):
            images.append(source_path + file)
    print(f"[INFO] Found {len(images)} images")
    random.shuffle(images)
    return images


def pre_processing(image: np.ndarray):
    """
    :param image:
    :return:
    """
    _, image_thr = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    num_labels, labels_im = cv2.connectedComponents(image_thr)
    for label in range(1, num_labels):
        mask = np.array(labels_im, dtype=np.uint8)
        mask[labels_im == label] = 255
    contours, _ = cv2.findContours(image_thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area, index_max_contour = -1, -1
    for index, contour in enumerate(contours):
        area_contour = cv2.moments(contour)['m00']
        if area_contour > max_area:
            max_area = area_contour
            index_max_contour = index
    if index_max_contour < 0:
        return 0, 0, image.shape[0], image.shape[1]
    return cv2.boundingRect(contours[index_max_contour])
