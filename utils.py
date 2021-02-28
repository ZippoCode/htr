import torch
import numpy as np


def get_patches(image, size_patches=256):
    """
        Convert a torch into a list of patches with shape (256x256)
    :return:
    """
    result = []
    h, w = image.shape[-2:]
    new_h, new_w = ((h // size_patches) + 1) * size_patches, ((w // size_patches) + 1) * size_patches
    image_padding = torch.zeros((1, 1, new_h, new_w))
    image_padding[:, :, :h, :w] = image.detach().clone()

    for i in range(0, new_h, size_patches):
        for j in range(0, new_w, size_patches):
            result.append(image_padding[:, :, i:i + size_patches, j:j + size_patches])
    return result


def merge_patches(patches_list, size, patches_size=256):
    image_result = torch.zeros((1, size[0], size[1]))
    index = 0
    for i in range(0, image_result.shape[1], patches_size):
        for j in range(0, image_result.shape[2], patches_size):
            image_result[:, i:i + patches_size, j:j + patches_size] = patches_list[index].detach().clone()
            index = index + 1
    return image_result
