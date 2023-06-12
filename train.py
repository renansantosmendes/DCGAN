import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

from models import Generator, Discriminator, weights_init

if __name__ == '__main__':
    manual_seed = 42
    print("Random Seed: ", manual_seed)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    data_root = "C:\\PUC\\2023-01\\gans-datasets"
    workers = 1
    batch_size = 128
    image_size = 64
    number_of_channels = 3

    latent_vector_size = 100
    generator_feature_maps_size = 64
    discriminator_feature_maps_size = 64
    num_epochs = 50
    lr = 0.0002
    beta1 = 0.5
    number_of_gpus = 1


    def check_is_valid_file(file):
        return file.endswith('.jpg')


    dataset = dset.ImageFolder(root=data_root,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]),
                               is_valid_file=check_is_valid_file)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)

    device = torch.device("cuda:0" if (torch.cuda.is_available() and number_of_gpus > 0) else "cpu")

    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))

    generator = Generator(number_of_gpus,
                          latent_vector_size,
                          generator_feature_maps_size,
                          number_of_channels).to(device)
    discriminator = Discriminator(number_of_gpus,
                                  number_of_channels,
                                  discriminator_feature_maps_size,).to(device)

    if (device.type == 'cuda') and (number_of_gpus > 1):
        generator = nn.DataParallel(generator, list(range(number_of_gpus)))

    if (device.type == 'cuda') and (number_of_gpus > 1):
        discriminator = nn.DataParallel(discriminator, list(range(number_of_gpus)))

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    print(generator)
    print(discriminator)

    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, latent_vector_size, 1, 1, device=device)

    real_label = 1
    fake_label = 0

    generator_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

    img_list = []
    generator_losses = []
    discriminator_losses = []
    iterations = 0

    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            discriminator.zero_grad()
            real_data = data[0].to(device)
            current_batch_size = real_data.size(0)
            label = torch.full((current_batch_size,), real_label, device=device)
            output = discriminator(real_data).view(-1)
            real_data_discriminator_error = criterion(output, label.float())
            real_data_discriminator_error.backward()

            noise = torch.randn(current_batch_size, latent_vector_size, 1, 1, device=device)
            fake = generator(noise)
            label.fill_(fake_label)
            output = discriminator(fake.detach()).view(-1)
            fake_data_discriminator_error = criterion(output, label.float())
            fake_data_discriminator_error.backward()

            discriminator_error = real_data_discriminator_error + fake_data_discriminator_error
            discriminator_optimizer.step()

            generator.zero_grad()
            label.fill_(real_label)
            output = discriminator(fake).view(-1)
            generator_error = criterion(output, label.float())
            generator_error.backward()
            generator_optimizer.step()

            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f' % (epoch,
                                                                      num_epochs,
                                                                      i,
                                                                      len(dataloader),
                                                                      discriminator_error.item(),
                                                                      generator_error.item()))

                generator_losses.append(generator_error.item())
                discriminator_losses.append(discriminator_error.item())

                if (iterations % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                    with torch.no_grad():
                        fake = generator(fixed_noise).detach().cpu()
                        img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

                        iterations += 1

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(generator_losses, label="G")
    plt.plot(discriminator_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    HTML(ani.to_jshtml())

    real_batch = next(iter(dataloader))

    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),
                            (1, 2, 0)))

    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    plt.show()