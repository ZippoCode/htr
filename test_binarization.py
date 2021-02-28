import os
import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as v_utils
import matplotlib.pyplot as plt

# Custom importing
from networks import Generator
from utils import get_patches, merge_patches
from parameters import PATH_WEIGHTS_GENERATOR

torch.set_printoptions(precision=10)  # Print all decimals

path = 'dataset/'
num_gpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and num_gpu > 0) else "cpu")

generator = Generator()
generator = generator.to(device)
generator.state_dict(torch.load(PATH_WEIGHTS_GENERATOR, map_location=device))

dataset = os.listdir('test')

with torch.no_grad():
    for num_image, image in enumerate(dataset):
        print(f"Elaborated: {image}")
        path = f'test/{image}'
        image = Image.open(path).convert('L')
        torch_image = transforms.ToTensor()(image).to(device)
        h, w = torch_image.shape[-2:]
        new_h = ((torch_image.shape[1] // 256) + 1) * 256
        new_w = ((torch_image.shape[2] // 256) + 1) * 256
        patches = get_patches(torch_image)
        predicted_patches = list()
        for patch in patches:
            result_patch = generator(patch.to(device))
            predicted_patches.append(result_patch)

        predicted_image = merge_patches(predicted_patches, (new_h, new_w))
        predicted_image = predicted_image[:h, :w, :]
        v_utils.save_image(predicted_image, f"bin_{num_image}.png", normalize=True)

print("Test completed successfully!")
