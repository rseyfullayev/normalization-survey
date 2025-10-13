import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms as T


MEAN = (0.5071, 0.4867, 0.4408)
STD  = (0.2675, 0.2565, 0.2761)


def vector_norm(x: torch.Tensor, ord: str = "l2", eps: float = 1e-6) -> torch.Tensor:

    if ord not in ["l1", "l2", "linf"]:
        return x

    v = x.view(-1)
    if ord == "l1":
        s = v.abs().sum()
    elif ord == "l2":
        s = torch.sqrt((v * v).sum())
    elif ord == "linf":
        s = v.abs().max()

    s = s.clamp(min=eps)
    v = v / s
    return v.view_as(x)

class PerSampleNormalize(nn.Module):

    def __init__(self, ord: str = "l2"):
        super().__init__()
        self.ord = ord

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return vector_norm(x, ord=self.ord)


def make_transform(img_size=32, norm_type="none"):

    transform_list = [
        T.RandomCrop(img_size, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor()
    ]


    if norm_type in ["l1", "l2", "linf"]:
        transform_list.append(PerSampleNormalize(ord=norm_type))


    transform_list.append(T.Normalize(mean=MEAN, std=STD))
    return T.Compose(transform_list)

def make_test_transform(norm_type="none"):
    transform_list = [T.ToTensor()]
    if norm_type in ["l1", "l2", "linf"]:
        transform_list.append(PerSampleNormalize(ord=norm_type))
    transform_list.append(T.Normalize(mean=MEAN, std=STD))
    return T.Compose(transform_list)


def loadData(batch=128, valid=5000, workers=2, seed=42, norm_type="none"):

    train_tf = make_transform(norm_type=norm_type)
    test_tf = make_test_transform(norm_type=norm_type)

    trainset = datasets.CIFAR100(root="data", train=True, download=True, transform=train_tf)
    testset  = datasets.CIFAR100(root="data", train=False, download=True, transform=test_tf)

    gen = torch.Generator().manual_seed(seed)
    train_len = len(trainset) - valid
    train_subset, valid_subset = random_split(trainset, [train_len, valid], generator=gen)

    train_loader = DataLoader(train_subset, batch_size=batch, shuffle=True, num_workers=workers)
    val_loader = DataLoader(valid_subset, batch_size=batch, shuffle=False, num_workers=workers)
    test_loader = DataLoader(testset, batch_size=batch, shuffle=False, num_workers=workers)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":

    for norm_type in ["none", "l1", "l2", "linf"]:
        print(f"\ncifar-100 loaders with {norm_type.upper()} normalization")
        train_loader, val_loader, test_loader = loadData(norm_type=norm_type)
        images, labels = next(iter(train_loader))
        print("batch shape:", images.shape, "| labels:", labels.shape)
        print("first image mean:", images[0].mean().item())
