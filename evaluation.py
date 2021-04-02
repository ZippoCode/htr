import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
# Custom importing
from binarization.read_write_image import upload_files

G123_LUT = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1,
                     0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0,
                     1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                     0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1,
                     0, 0, 0], dtype=np.bool)

G123P_LUT = np.array([0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
                      1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0,
                      0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0,
                      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1,
                      0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0], dtype=np.bool)


def evaluation(test: np.ndarray, ground_truth: np.ndarray):
    true_positives = np.bitwise_and(test, ground_truth)
    false_positives = np.bitwise_and(test, ~ground_truth)
    false_negatives = np.bitwise_and(~test, ground_truth)
    c_tp = np.sum(true_positives)
    c_fn = np.sum(false_positives)
    c_fp = np.sum(false_negatives)
    # print(f"C_TR={c_tp}, C_FP={c_fp}, C_FN= {c_fn}")

    rc = np.round(c_tp / (c_fn + c_tp), decimals=2)
    pr = np.round(c_tp / (c_fp + c_tp), decimals=2)
    fm = np.round(((2 * rc * pr) / (rc + pr)) * 100, decimals=2)

    #  Pseudo F-Measure
    mask = np.array([[8, 4, 2], [16, 0, 1], [32, 64, 128]], dtype=np.uint8)
    skel = ground_truth
    while True:
        before = np.sum(skel)
        for lut in [G123_LUT, G123P_LUT]:
            n = ndimage.correlate(skel, mask, mode='constant')
            d = np.take(lut, n)
            skel[d] = 0
        after = np.sum(skel)
        if before == after:
            break
    plt.imshow(skel, cmap='gray')
    plt.show()
    print(f"Recall: {rc}")
    print(f"Precition: {pr}")
    print(f"F-Measure: {fm}")
    return rc, pr, fm


test_images = upload_files('output/DIBCO_2018/Results/')
gt_image = upload_files('data/ground_truth/DIBCO/2018/')
test_images, gt_image = sorted(test_images), sorted(gt_image)
average_rc, average_pr, average_fm = 0, 0, 0
for path_test, path_gt in zip(test_images, gt_image):
    image_test = cv2.imread(path_test, cv2.IMREAD_GRAYSCALE)
    image_gt = cv2.imread(path_gt, cv2.IMREAD_GRAYSCALE)
    rc, pr, fm = evaluation(image_test == 0, image_gt == 0)
    print('*' * 32)
    average_rc += rc
    average_pr += pr
    average_fm += fm

print('#' * 25)
average_rc = average_rc / len(test_images)
average_pr = average_pr / len(test_images)
average_fm = average_fm / len(test_images)
print(f"Recall: {average_rc}, Precition: {average_pr}, F-Measure: {average_fm}")
