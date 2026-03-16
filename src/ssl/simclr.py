import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

def _flatten_feats(x):
    if x.dim() == 4:
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
    elif x.dim() > 2:
        x = x.view(x.size(0), -1)
    return x




class SimCLR(nn.Module):
    def __init__(self, encoder, projector):
        super().__init__()
        self.encoder = encoder
        self.projector = projector

    def forward(self, x):
        h = self.encoder(x)
        h = _flatten_feats(h)

        if h.dim() != 2:
            raise RuntimeError(f"Expected 2D features, but got shape {tuple(h.shape)}")
        try:
            z = self.projector(h)
        except RuntimeError as e:
            raise RuntimeError(
                f"Projection head input dim mismatch "
                f"Got features of dim {h.size(1)}; "
                f"dim did not match"
            ) from e
        z = F.normalize(z, dim=1)
        return h, z

def train_simclr(model, loader, loss_fn, optimizer, device, epochs=200):
        model.train()
        for epoch in range(1, epochs + 1):
            running, seen = 0.0, 0
            for (x1, x2), _ in tqdm(loader, desc=f"epoch {epoch}", ncols=80):

                if x1.size(0) != x2.size(0):

                    continue

                x1, x2 = x1.to(device), x2.to(device)
                _, z1 = model(x1)
                _, z2 = model(x2)

                z1 = F.normalize(z1, dim=1)
                z2 = F.normalize(z2, dim=1)

                loss = loss_fn(z1, z2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                b = x1.size(0)
                running += loss.item() * (2 * b)  # loss over 2B samples
                seen += 2 * b
            avg = running / max(seen, 1)
            print(f"epoch {epoch}: loss {avg:.4f}")
