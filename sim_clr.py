
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100

from src.ssl.augment import simclr_transforms
from src.ssl.two_view_dataset import TwoViewWrapper
from src.ssl.simclr import SimCLR, train_simclr
from src.models.backbones import resnet18_backbone
from src.models.heads import ProjectionMLP
from src.losses.nt_xent import NTXentLoss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)


    base = CIFAR100(root="./data", train=True, download=True, transform=None)
    t = simclr_transforms(img_size=32)
    ds = TwoViewWrapper(base, t)
    loader = DataLoader(ds, batch_size=256, shuffle=True, num_workers=4, drop_last=True)


    encoder = resnet18_backbone()
    proj = ProjectionMLP(in_dim=100, hid_dim=512, out_dim=128)
    model = SimCLR(encoder, proj).to(device)

    loss_fn = NTXentLoss(temperature=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)


    train_simclr(model, loader, loss_fn, optimizer, device, epochs=20)


    torch.save(model.encoder.state_dict(), "checkpoints/resnet18_simclr_encoder.pth")
    print("Saved backbone to checkpoints/resnet18_simclr_encoder.pth")

if __name__ == "__main__":
    main()