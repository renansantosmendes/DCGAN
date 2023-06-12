import random
import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, number_of_gpus, latent_vector_size, generator_feature_maps_size, number_of_channels):
        super(Generator, self).__init__()
        self.number_of_gpus = number_of_gpus
        self.layers = nn.Sequential(

            nn.ConvTranspose2d(latent_vector_size, generator_feature_maps_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(generator_feature_maps_size * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(generator_feature_maps_size * 8, generator_feature_maps_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(generator_feature_maps_size * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(generator_feature_maps_size * 4, generator_feature_maps_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(generator_feature_maps_size * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(generator_feature_maps_size * 2, generator_feature_maps_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(generator_feature_maps_size),
            nn.ReLU(True),

            nn.ConvTranspose2d(generator_feature_maps_size, number_of_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.layers(input)


class Discriminator(nn.Module):
    def __init__(self, number_of_gpus, number_of_channels, discriminator_feature_maps_size,):
        super(Discriminator, self).__init__()
        self.number_of_gpus = number_of_gpus
        self.layers = nn.Sequential(
            nn.Conv2d(number_of_channels, discriminator_feature_maps_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(discriminator_feature_maps_size, discriminator_feature_maps_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(discriminator_feature_maps_size * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(discriminator_feature_maps_size * 2, discriminator_feature_maps_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(discriminator_feature_maps_size * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(discriminator_feature_maps_size * 4, discriminator_feature_maps_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(discriminator_feature_maps_size * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(discriminator_feature_maps_size * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.layers(input)
