import torchvision
import torch

def load_dataset():
    data_path = ''
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=torchvision.transforms.ToTensor()
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        num_workers=0,
        shuffle=True
    )
    return train_loader

for batch_idx, (data, target) in enumerate(load_dataset()):
    #train network