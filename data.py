from torchvision import datasets
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def check_is_valid_file(file):
    return file.endswith('.jpg')


def create_dataloader(data_root,
                      image_size,
                      is_valid_file,
                      batch_size,
                      workers):
    dataset = ImageFolder(root=data_root,
                          transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                          is_valid_file=is_valid_file)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=workers)
    return dataloader
