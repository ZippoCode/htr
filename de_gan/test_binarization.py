import os
import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as v_utils

# Custom importing
from de_gan.networks import Generator
from utils import predict_image
from parameters import PATH_WEIGHTS_GENERATOR

torch.set_printoptions(precision=10)  # Print all decimals

path = '../data/dataset/'
num_gpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and num_gpu > 0) else "cpu")

generator = Generator()
generator = generator.to(device)
generator.load_state_dict(torch.load(PATH_WEIGHTS_GENERATOR, map_location=device))

dataset = os.listdir('../test')

with torch.no_grad():
    for num_image, image in enumerate(dataset):
        print(f"Elaborated: {image}")
        path = f'test/{image}'
        image = Image.open(path).convert('L')
        torch_image = transforms.ToTensor()(image).to(device).unsqueeze(0)
        predicted_image = predict_image(torch_image, generator, device)
        v_utils.save_image(predicted_image, os.path.join("../output", f"bin_{num_image}.png"), normalize=True)

print("Test completed successfully!")
