import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torchvision.models import resnet18, resnet34, resnet50
from src.datasets.cifar100 import SimCLRDataset


def get_norm_layer(layer, num_groups=32):

    if layer == "batch":
        return nn.BatchNorm2d
    elif layer == "group":
        return lambda num_channels: nn.GroupNorm(num_groups, num_channels)
    else:
        raise ValueError("wrong norm")


class Encoder(nn.Module):
    def __init__(self, model, layer_norm, num_groups=32):
        super().__init__()
        self.layer_norm = get_norm_layer(layer_norm, num_groups)
        self.backbone = self.choose_encoder(model)(norm_layer= layer_norm, num_classes = 0)

    def forward(self, input_data):
        h = self.backbone(input_data)
        return h
    def choose_encoder(self, model):
        if model == 'resnet18':
            return resnet18
        elif model == 'resnet34':
            return resnet34
        elif model == 'resnet50':
            return resnet50
        return None


class SimCLRModel(nn.Module):
    def __init__(self, backbone_model, layer_norm, out_dim=128):
        super().__init__()

        self.backbone = Encoder(backbone_model, layer_norm)
        self.feature_dim: int = self.backbone.backbone.fc.in_features

        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, out_dim)
        )
    def forward(self, x) -> torch.Tensor:
        h = self.backbone(x)
        z = self.projector(h)
        return f.normalize(z, dim=1)


def nt_xent_loss(z_i, z_j, temperature=0.5):
    batch_size = z_i.shape[0]
    device = z_i.device

    representations = torch.cat([z_i, z_j], dim=0)  # (2*B, dim)
    similarity_matrix = torch.mm(representations, representations.t())  # (2*B, 2*B)


    mask = torch.eye(2 * batch_size, device=device).bool()
    similarity_matrix = similarity_matrix.masked_fill(mask, -1e9)


    labels = torch.cat([torch.arange(batch_size, 2 * batch_size),
                        torch.arange(batch_size)])
    labels = labels.to(device)

    logits = similarity_matrix / temperature
    return f.cross_entropy(logits, labels)



def train_simclr(model_name, norm_type, epochs=10, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    dataset = SimCLRDataset("cifar100", size=224)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4
    )


    model = SimCLRModel(model_name, layer_norm=norm_type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)



    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for img_i, img_j in loader:
            img_i, img_j = img_i.to(device), img_j.to(device)

            optimizer.zero_grad()

            z_i = f.normalize(model(img_i), dim=1)
            z_j = f.normalize(model(img_j), dim=1)

            loss = nt_xent_loss(z_i, z_j)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss / len(loader):.4f}")

    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for img_i, img_j in loader:
            img_i, img_j = img_i.to(device), img_j.to(device)

            z_i = model(img_i)
            z_j = model(img_j)

            batch_features = torch.cat([z_i, z_j], dim=0).cpu()

            batch_labels = torch.arange(img_i.size(0))
            batch_labels = torch.cat([batch_labels, batch_labels], dim=0)

            features.append(batch_features)
            labels.append(batch_labels)

    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)

    return model.backbone, features, labels


def compute_intra_inter(features, labels):

    classes = labels.unique()
    centroids = {}

    for c in classes:
        mask = labels == c
        centroids[c.item()] = features[mask].mean(dim=0)

    intra = 0
    count = 0

    for i in range(len(features)):
        c = labels[i].item()
        centroid = centroids[c]

        intra += torch.norm(features[i] - centroid)
        count += 1

    intra /= count

    inter = 0
    pairs = 0

    class_list = list(centroids.keys())

    for i in range(len(class_list)):
        for j in range(i+1, len(class_list)):

            c1 = centroids[class_list[i]]
            c2 = centroids[class_list[j]]

            inter += torch.norm(c1 - c2)
            pairs += 1

    inter /= pairs

    return intra, inter







if __name__ == "__main__":

    trained_backbone, fet, lab = train_simclr("resnet50", norm_type="batch", epochs=5)

    intr, inte = compute_intra_inter(fet, lab)

    print(intr, inte)
    print("Training finished. Backbone is ready for extraction.")
