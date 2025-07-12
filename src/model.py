import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import clip


class TopKSAE(nn.Module):

    def __init__(self,
                 input_d: int,
                 latent_n: int,
                 top_k: int,
                 lambda_: float
                 ) -> None:
        super(TopKSAE, self).__init__()

        self.top_k = top_k
        self.input_d = input_d
        self.latent_n = latent_n
        self.lambda_ = lambda_

        W_dec = torch.randn(input_d, latent_n) / math.sqrt(input_d)
        W_dec = F.normalize(W_dec, dim=0)
        self.W_dec = nn.Parameter(W_dec)
        self.W_enc = nn.Parameter(W_dec.T.clone())

        self.bias  = nn.Parameter(torch.zeros(input_d))

    def topk(self, z: torch.tensor) -> torch.tensor:
        _, idx = torch.topk(z, self.top_k, dim = -1)
        mask = torch.zeros_like(z).scatter_(-1, idx, 1.0)
        return z * mask

    def forward(self, x: torch.tensor) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        z = (x - self.bias) @ self.W_enc.T
        z = self.topk(F.relu(z))
        reconstruct = z @ self.W_dec.T + self.bias
        return x, z, reconstruct

    def encode(self, x: torch.tensor) -> torch.tensor:
        z = (x - self.b) @ self.W_enc.T
        z = self.topk(F.relu(z))
        return z

    def loss(self, x: torch.tensor) -> torch.tensor:
        x_in, z, reconstruct = self.forward(x)
        rec_mse_loss = F.mse_loss(reconstruct, x_in, reduction='sum')
        normed = self.W_dec.norm(dim=0)
        l1  = (z.abs() * normed).sum()  # reg
        return rec_mse_loss + self.lambda_ * l1

    @torch.no_grad()
    def renorm(self)-> None:
      self.W_dec.data[:] = F.normalize(self.W_dec.data, dim=0)

    @torch.no_grad()
    def init_median_bias(self,
                         vit: clip.model.CLIP,
                         dataloader: DataLoader,
                         latent_n: int
                         ) -> None:

        vit.eval()
        device = next(vit.parameters()).device

        features = []
        for i, batch in enumerate(dataloader):
            if i >= latent_n:
                break
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            images = images.to(device)
            f = vit.encode_image(images)
            features.append(f.cpu())
        if not features:
            raise ValueError(f"Smth went wrong... (latent_n={latent_n})")

        features = torch.cat(features, dim=0)
        N, d = features.shape
        median = features.mean(dim=0)

        eps = 1e-5
        max_iter = 500
        for _ in range(max_iter):
            diffs = features - median.unsqueeze(0)
            distances = diffs.norm(dim=1)

            mask = distances < eps
            if mask.any():
                median = features[mask][0]
                break

            inv = 1.0 / (distances + eps)
            w = inv / inv.sum()
            new_median = (w.unsqueeze(1) * features).sum(dim=0)
            if (new_median - median).norm().item() < eps:
                median = new_median
                break
            median = new_median
        median = median.to(device)

        # init bias
        self.bias.data.copy_(median)
