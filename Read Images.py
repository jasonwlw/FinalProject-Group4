import torchvision
import torch

def load_dataset():
    train_path = ''
    train_dataset = torchvision.datasets.ImageFolder(
        root=train_path,
        transform=torchvision.transforms.ToTensor()
    )

    valid_path = ''
    valid_dataset = torchvision.datasets.ImageFolder(
        root=valid_path,
        transform=torchvision.transforms.ToTensor()
    )
    test_path = ''
    test_dataset = torchvision.datasets.ImageFolder(
        root=test_path,
        transform=torchvision.transforms.ToTensor()
    )
    return train_dataset,valid_dataset,test_dataset
def get_loaders(train_dataset,valid_dataset,test_dataset):
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        num_workers=0,
        shuffle=True
        )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=64,
        num_workers=0,
        shuffle=True
        )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,
        num_workers=0,
        shuffle=True
        )
    return train_loader,valid_loader,test_loader
#
# for batch_idx, (data, target) in enumerate(load_dataset()):
#     #train network



train_dataset,valid_dataset,test_dataset = load_dataset()
train_loader,valid_loader,test_loader = get_loaders(train_dataset,valid_dataset,test_dataset)
