from torchvision.datasets import ImageFolder

def get_dataset(root_dir, transform):
    dataset = ImageFolder(root = root_dir, transform = transform)
    return dataset