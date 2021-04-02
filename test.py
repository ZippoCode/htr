import time
import cv2
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Custom importing
from binarization.binarization import grayscale_conversion, lowlights_map, binarization_blackbox
from binarization.read_write_image import upload_files, pre_processing


def store_image(image: np.ndarray, folder_name: str, path_name: str):
    if not os.path.exists(path_name):
        try:
            os.makedirs(path_name)
        except OSError:
            print(f"Creation of the directory {path_name} failed")
        else:
            print(f"Successfully created the directories {path_name}")
    image = image.astype(np.uint8)
    cv2.imwrite(f'{path_name}/{folder_name}', image)


year = 2018

# Setting input folders
path_source = f'data/dataset/DIBCO/{year}/'
path_images = upload_files(path_source)

# Setting output folders
grayfication_dest = f'output/DIBCO_{year}/Grayfication'
lowlightsMap_dest = f'output/DIBCO_{year}/Lowlights_map'
result_dest = f'output/DIBCO_{year}/Results'
folder_destination = [grayfication_dest, lowlightsMap_dest, result_dest]
types = ['grayfication', 'lowlights_map', 'bin']
for path in path_images:
    # tuning_parameters(path)

    t0 = time.perf_counter()

    # Load image and apply the Connected Components
    print(f"Elaborate image {path}")
    name_image = os.path.splitext(os.path.basename(path))[0]
    image_bw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # (x, y, w, h) = pre_processing(image_bw)
    # image_bw = image_bw[y:y + h, x:x + w]

    print(f'Image shape: {image_bw.shape}')

    # With grayscale custom conversion
    print("----- Test with grayscale and lowlights map -----")
    image_bgr = np.float32(cv2.imread(path, cv2.IMREAD_COLOR))
    # grayscale_image = grayscale_conversion(image_bgr[y:y + h, x:x + w])
    grayscale_image = grayscale_conversion(image_bgr)
    lm_with_gray = lowlights_map(grayscale_image)
    lm_with_gray *= 255.0
    bin_image = binarization_blackbox(lm_with_gray)
    grayscale_image *= 255.0
    images = [grayscale_image, grayscale_image, bin_image]
    for index in range(3):
        name = f'{name_image}_{types[index]}.bmp'
        store_image(images[index], name, folder_destination[index])

    # store_image(lm_with_gray, bin_name_grayficitation, lowlightsMap_dest)
    # store_image(result_with_gray, bin_name_grayficitation, result_dest)

    # Without grayscale custom conversion
    print("----- Test with only lowlights map -----")
    image_bw = np.float32(image_bw)
    image_bw *= 1 / image_bw.max()
    lm_without_gray = lowlights_map(image_bw)
    lm_without_gray *= 255.0
    result_without_gray = binarization_blackbox(lm_without_gray)
    store_image(lm_without_gray, f'{name_image}_without_grayfication.bmp', lowlightsMap_dest)
    store_image(result_without_gray, f'{name_image}_without_grayfication.bmp', result_dest)

    # Without pre-processing
    print("----- Test without any algorithms -----")
    image_bw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    result_simple = binarization_blackbox(image_bw)
    store_image(image_bw, f'{name_image}_normal_grayscale.bmp', grayfication_dest)
    store_image(result_simple, f"{name_image}_simple.bmp", result_dest)
    print('*' * 50)

    print(f'Time: {round(time.perf_counter() - t0, 3)}')
    print('\n')

    """
    # Quadrant Page
    quadrant_name = f"{os.path.splitext(os.path.basename(path))[0]}_quadrant.png"
    quadrant_page = quadrant_binarization(image)
    ret_q, thresh_image_quadrant = cv2.threshold(np.uint8(quadrant_page), 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(f'output/{quadrant_name}', thresh_image_quadrant)

    """
