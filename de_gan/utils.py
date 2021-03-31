import torch
from typing import List
from de_gan.networks import Generator


def get_patches(image: torch.Tensor, size_patches=256):
    """
        Convert a torch.Tensor into a list of patches with shape (256x256)
    :return: list(torch.Tensor)
    """
    result = []
    h, w = image.shape[-2:]
    new_h, new_w = ((h // size_patches) + 1) * size_patches, ((w // size_patches) + 1) * size_patches
    image_padding = torch.zeros((1, 1, new_h, new_w))
    image_padding[:, :, :h, :w] = image.detach().clone()
    for i in range(0, new_h, size_patches):
        for j in range(0, new_w, size_patches):
            result.append(image_padding[:, :, i:i + size_patches, j:j + size_patches])
    return result, (new_h, new_w)


def merge_patches(patches_list: List[torch.Tensor], image_size: tuple, patches_size=256):
    """

    :param patches_list: List[torch.Tensor] where a torch.Tensor has shape (1, 1, 256, 256)
    :param image_size: tuple
    :param patches_size:
    :return:
    """
    image_result = torch.zeros((1, 1, image_size[0], image_size[1]))
    index = 0
    for i in range(0, image_size[0], patches_size):
        for j in range(0, image_size[1], patches_size):
            image_result[:, :, i:i + patches_size, j:j + patches_size] = patches_list[index].detach().clone()
            index = index + 1
    return torch.FloatTensor(image_result)


def predict_image(image: torch.Tensor, generator: Generator, device: torch.device):
    if len(image.shape) != 4:
        print("[ERROR] The shape must be four!")
    h, w = image.shape[-2:]
    patches, (new_h, new_w) = get_patches(image)
    predicted_patches = list()
    for patch in patches:
        result_patch = generator(patch.to(device))
        predicted_patches.append(result_patch)
    predicted_image = merge_patches(predicted_patches, (new_h, new_w))
    predicted_image = predicted_image[:, :, :h, :w]
    return predicted_image
