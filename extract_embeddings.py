
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torchvision import transforms

from src.models.backbones import resnet18_backbone

@torch.no_grad()
def run_split(model, dataset, device, batch=256):
    loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=4)
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device)
        h = model(x)
        feats.append(h.cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labels)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    base_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                             std=[0.2675, 0.2565, 0.2761]),
    ])
    val = CIFAR100(root="./data", train=False, download=True, transform=base_tf)

    encoder = resnet18_backbone().to(device)
    encoder.load_state_dict(torch.load("checkpoints/resnet18_simclr_encoder.pth", map_location=device))
    encoder.eval()

    feats, labels = run_split(encoder, val, device)
    np.save("embeddings/cifar100_simclr_val_feats.npy", feats)
    np.save("embeddings/cifar100_simclr_val_labels.npy", labels)
    print("Saved embeddings to embeddings/*.npy")

if __name__ == "__main__":
    main()
