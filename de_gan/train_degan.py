import os
import re
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as v_utils
from torchvision.transforms import transforms
from PIL import Image

# Custom importing
from parameters import PATH_WEIGHTS_GENERATOR, PATH_WEIGHTS_DISCRIMINATOR
from de_gan.networks import Generator, Discriminator
from utils import predict_image


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(tensor=m.weight.data, mean=0.0, std=0.02)


def patches(degraded_image, clear_image, size_patches=256, stride=192):
    degraded_patches, clear_patches = [], []
    h, w = degraded_image.shape
    new_h, new_w = ((h // size_patches) + 1) * size_patches, ((w // size_patches) + 1) * size_patches
    degraded_padding, clear_padding = np.ones((new_h, new_w)), np.ones((new_h, new_w)) * 255.0
    degraded_padding[:h, :w], clear_padding[:h, :w] = degraded_image, clear_image

    for i in range(0, new_h - size_patches, stride):
        for j in range(0, new_w - size_patches, stride):
            degraded_patches.append(degraded_padding[i:i + size_patches, j:j + size_patches])
            clear_patches.append(clear_padding[i:i + size_patches, j:j + size_patches] / 255.0)
    return np.array(degraded_patches), np.array(clear_patches)


num_gpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available()) and num_gpu > 0 else "gpu")
num_epochs = 80

generator = Generator().to(device)
generator.apply(weights_init)
# generator.load_state_dict(torch.load(PATH_WEIGHTS_GENERATOR))
discriminator = Discriminator().to(device)
# discriminator.load_state_dict(torch.load(PATH_WEIGHTS_DISCRIMINATOR))

# Loss functions
criterion = nn.MSELoss().to(device)
criterionBCE = nn.BCELoss().to(device)
optimizerGenerator = torch.optim.Adam(params=generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
optimizerDiscriminator = torch.optim.Adam(params=discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))

batch_size = 4
train_set, gt_set = [], []

# Found files
for dir_path, _, filenames in os.walk('../data/dataset'):
    for file in filenames:
        train_set.append(os.path.abspath(os.path.join(dir_path, file)))
for dir_path, _, filenames in os.walk('../ground_truth'):
    for file in filenames:
        gt_set.append(os.path.abspath(os.path.join(dir_path, file)))
train_set.sort(key=natural_keys), gt_set.sort(key=natural_keys)

# Visualized output
d_x, d_g_z1, d_g_z2 = 0, 0, 0
G_losses = []
D_losses = []
error_discriminator = None
error_generator_one = None
error_generator_two = None

# Test image configuration
size_patches = 256
path = f'../test/1.bmp'
image = Image.open(path).convert('L')
test_image = transforms.ToTensor()(image).to(device).unsqueeze(0)

for epoch in range(num_epochs):
    for i in range(len(train_set)):
        degraded_image = Image.open(train_set[i]).convert('L')
        degraded_image.save('Current_Degraded_Image.png')
        degraded_image = plt.imread('Current_Degraded_Image.png')
        clear_image = Image.open(gt_set[i]).convert('L')
        clear_image.save('Current_Clear_Image.png')
        clear_image = plt.imread('Current_Clear_Image.png')
        degraded_batch, clear_batch = patches(degraded_image, clear_image)
        batch_count = degraded_batch.shape[0] // batch_size
        for n_batch in range(batch_count):
            seed = range(n_batch * batch_size, (n_batch * batch_size) + batch_size)
            n_degraded_batch = torch.from_numpy(degraded_batch[seed].reshape(batch_size, 1, 256, 256))
            n_degraded_batch = n_degraded_batch.float().to(device)
            n_clear_batch = torch.from_numpy(clear_batch[seed].reshape(batch_size, 1, 256, 256))
            n_clear_batch = n_clear_batch.float().to(device)

            n_valid_batch = torch.full((batch_size * 16 * 16,), fill_value=1., dtype=torch.float, device=device)
            n_fake_batch = torch.full((batch_size * 16 * 16,), fill_value=0., dtype=torch.float, device=device)

            ######################################################################
            # (1) Update D: Maximize log(D(I_W, I_GT)) + log(1 - D(I_W, G(I_W))
            ######################################################################
            # Training with real
            discriminator.zero_grad()
            output = discriminator(n_clear_batch, n_degraded_batch).view(-1)
            error_discriminator_real = criterion(output, n_valid_batch)
            error_discriminator_real.backward()
            d_x = output.mean().item()

            # Training with fake
            n_generated_batch = generator(n_degraded_batch)
            output = discriminator(n_generated_batch.detach(), n_degraded_batch).view(-1)
            error_discriminator_fake = criterion(output, n_fake_batch)
            error_discriminator_fake.backward()
            d_g_z1 = output.mean().item()
            error_discriminator = error_discriminator_real + error_discriminator_fake
            optimizerDiscriminator.step()

            ######################################################################
            # (1) Update G: Maximize log(1 - D(I_W, G(I_W))
            ######################################################################
            generator.zero_grad()
            output = discriminator(n_generated_batch, n_degraded_batch).view(-1)
            error_generator_one = criterion(output, n_valid_batch)
            error_generator_one.backward()
            d_g_z2 = output.mean().item()
            optimizerGenerator.step()

            #####################################################################
            # Addition Log Loss function
            #####################################################################
            # generator.zero_grad()
            # output_generator = generator(n_degraded_batch)
            # error_generator_two = criterion(output_generator, n_clear_batch)
            # error_generator_two.backward()
            # d_g_z3 = output.mean().item()
            # optimizerGenerator.step()

        if i % 50 == 0:
            epoch_text = f"[{epoch}/{num_epochs}][{i}/{len(train_set)}]"
            # loss_text = "\tLoss_D: %.4f\tLoss_G: %.4f / %.4f" % (
            # error_discriminator.item(), error_generator_one.item(), error_generator_two.item())
            loss_text = "\tLoss_D: %.4f\tLoss_G: %.4f" % (
                error_discriminator.item(), error_generator_one.item())
            error_text = f"\t\tD(I_W, I_GT): %.4f\tD(I_W, G(I_W)) %.4f / %.4f" % (d_x, d_g_z1, d_g_z2)
            print(epoch_text + loss_text + error_text)
            with torch.no_grad():
                bin_batch = predict_image(test_image, generator, device)
                v_utils.save_image(bin_batch, os.path.join("../output", "bin_sample.png"), normalize=True)
            torch.save(generator.state_dict(), PATH_WEIGHTS_GENERATOR)
            torch.save(discriminator.state_dict(), PATH_WEIGHTS_DISCRIMINATOR)
