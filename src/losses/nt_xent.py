import torch
import torch.nn.functional as f

class NTXentLoss(torch.nn.Module):

    def __init__(self, temperature = 0.5):
        super(NTXentLoss, self).__init__()
        self.tau = temperature



    def forward(self, z1,z2):

        B, D = z1.shape
        z = torch.cat([z1, z2], dim=0)
        sim = torch.matmul(z, z.T) / self.tau

        mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(mask, float('-inf'))


        pos_idx = torch.arange(B, device=z.device)
        positives = torch.cat([pos_idx + B, pos_idx], dim=0)  # (2B,)

        loss = f.cross_entropy(sim, positives)
        return loss


