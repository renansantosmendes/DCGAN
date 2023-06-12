import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils

from data import create_dataloader, check_is_valid_file
from models import Generator, Discriminator, weights_init

manual_seed = 42
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
real_label = 1
fake_label = 0


def train(generator,
          discriminator,
          generator_optimizer,
          discriminator_optimizer,
          generator_losses,
          discriminator_losses,
          criterion,
          num_epochs,
          dataloader,
          device,
          iterations,
          img_list):

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


def create_models(number_of_gpus,
                  latent_vector_size,
                  generator_feature_maps_size,
                  discriminator_feature_maps_size,
                  number_of_channels,
                  device):
    generator = Generator(number_of_gpus,
                          latent_vector_size,
                          generator_feature_maps_size,
                          number_of_channels).to(device)
    discriminator = Discriminator(number_of_gpus,
                                  number_of_channels,
                                  discriminator_feature_maps_size).to(device)
    if (device.type == 'cuda') and (number_of_gpus > 1):
        generator = nn.DataParallel(generator,
                                    list(range(number_of_gpus)))
    if (device.type == 'cuda') and (number_of_gpus > 1):
        discriminator = nn.DataParallel(discriminator,
                                        list(range(number_of_gpus)))
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    return generator, discriminator


def get_available_device():
    return torch.device("cuda:0" if (torch.cuda.is_available() and number_of_gpus > 0) else "cpu")


def create_optimizers(generator,
                      discriminator,
                      lr,
                      beta1):
    generator_optimizer = optim.Adam(generator.parameters(),
                                     lr=lr,
                                     betas=(beta1, 0.999))
    discriminator_optimizer = optim.Adam(discriminator.parameters(),
                                         lr=lr,
                                         betas=(beta1, 0.999))
    return generator_optimizer, discriminator_optimizer


if __name__ == '__main__':
    dataloader = create_dataloader(data_root=data_root,
                                   image_size=image_size,
                                   is_valid_file=check_is_valid_file,
                                   batch_size=batch_size,
                                   workers=workers)

    device = get_available_device()
    generator, discriminator = create_models(number_of_gpus=number_of_gpus,
                                             latent_vector_size=latent_vector_size,
                                             generator_feature_maps_size=generator_feature_maps_size,
                                             discriminator_feature_maps_size=discriminator_feature_maps_size,
                                             number_of_channels=number_of_channels,
                                             device=device)

    print(generator)
    print(discriminator)

    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, latent_vector_size, 1, 1, device=device)
    generator_optimizer, discriminator_optimizer = create_optimizers(generator=generator,
                                                                     discriminator=discriminator,
                                                                     lr=lr,
                                                                     beta1=beta1)
    img_list = []
    generator_losses = []
    discriminator_losses = []
    iterations = 0

    train(generator=generator,
          discriminator=discriminator,
          generator_optimizer=generator_optimizer,
          discriminator_optimizer=discriminator_optimizer,
          generator_losses=generator_losses,
          discriminator_losses=discriminator_losses,
          criterion=criterion,
          num_epochs=num_epochs,
          dataloader=dataloader,
          device=device,
          iterations=iterations,
          img_list=img_list)
