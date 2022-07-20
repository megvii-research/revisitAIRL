import os
from torchvision import datasets


class TwoCropsTransform:
    def __init__(self, transform):
        if not isinstance(transform, list):
            self.transform = [transform, transform]
        else:
            self.transform = transform

    def __call__(self, x):
        return [trans(x) for trans in self.transform]


def build_dataset(transform, data_path, is_train, two_aug=True):
    if two_aug:
        transform = TwoCropsTransform(transform)
    if not os.path.exists(data_path):
        raise ValueError("data_path {} is not valid.".format(data_path))

    data_path = os.path.join(data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(data_path, transform=transform)
    return dataset
