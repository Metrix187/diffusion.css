"""The denoiser: one small MLP per noise level, evaluated for all levels at once.

Level j = 0 is pure noise, level j = T is the clean image.
    x_j = alpha_j * x0 + sigma_j * eps,   alpha_j = sin(phi_j),  sigma_j = cos(phi_j)
Network j (1 <= j <= T-1) reads x_j and predicts x0.  Level 0 needs no network: the best guess from
pure noise is the dataset mean.

Shape of each network (chosen so the CSS stays small: the wide layers never mix colour channels):
    in    per channel c: 256 pixels -> G_IN units           3 * 256 * G_IN  weights
    mid   3*G_IN -> M -> 3*G_OUT   (fully connected)        cheap
    out   per channel c: G_OUT units -> 256 pixels          3 * 256 * G_OUT weights
"""
import math

import numpy as np
import torch
import torch.nn as nn


def schedule(T, lam_min=-6.0, lam_max=6.0):
    """T+1 levels: pure noise, then T-1 levels evenly spaced in log-SNR, then the clean image."""
    lam = np.linspace(lam_min, lam_max, T - 1)
    alpha = np.concatenate([[0.0], np.sqrt(1 / (1 + np.exp(-lam))), [1.0]])
    sigma = np.concatenate([[1.0], np.sqrt(1 / (1 + np.exp(lam))), [0.0]])
    return alpha, sigma


def ddim_coefs(alpha, sigma):
    """x_{j+1} = k1[j] * x_j + k2[j] * x0_hat_j   for j = 0..T-1"""
    T = len(alpha) - 1
    k1 = np.array([sigma[j + 1] / sigma[j] for j in range(T)])
    k2 = np.array([alpha[j + 1] - alpha[j] * sigma[j + 1] / sigma[j] for j in range(T)])
    return k1, k2


class Denoisers(nn.Module):
    def __init__(self, n_levels, g_in=12, mid=64, g_out=20, mean=None):
        super().__init__()
        L = n_levels                      # networks for levels 1..T-1
        self.L, self.g_in, self.mid, self.g_out = L, g_in, mid, g_out

        def u(*shape, fan):
            return nn.Parameter((torch.rand(*shape) * 2 - 1) / math.sqrt(fan))

        self.w_in = u(L, 3, g_in, 256, fan=256)
        self.b_in = nn.Parameter(torch.zeros(L, 3 * g_in))
        self.w_m1 = u(L, 3 * g_in, mid, fan=3 * g_in)
        self.b_m1 = nn.Parameter(torch.zeros(L, mid))
        self.w_m2 = u(L, mid, 3 * g_out, fan=mid)
        self.b_m2 = nn.Parameter(torch.zeros(L, 3 * g_out))
        self.w_out = u(L, 3, g_out, 256, fan=g_out)
        b = torch.zeros(L, 3, 256)
        if mean is not None:              # mean: (768,) laid out pixel*3 + channel
            b[:] = torch.as_tensor(mean, dtype=torch.float32).view(256, 3).t()
        self.b_out = nn.Parameter(b)

    def forward(self, x, wq=None):
        """x: (L, B, 768) -> x0_hat (L, B, 768).  wq optionally maps (name, tensor) -> tensor."""
        q = (lambda n, w: w) if wq is None else wq
        L, B, _ = x.shape
        xc = x.view(L, B, 256, 3).permute(0, 1, 3, 2)                      # L B 3 256
        h = torch.einsum('lbcp,lcgp->lbcg', xc, q('w_in', self.w_in)).reshape(L, B, -1)
        h = torch.relu(h + self.b_in[:, None])
        h = torch.relu(torch.einsum('lbi,lio->lbo', h, q('w_m1', self.w_m1)) + self.b_m1[:, None])
        h = torch.relu(torch.einsum('lbi,lio->lbo', h, q('w_m2', self.w_m2)) + self.b_m2[:, None])
        o = torch.einsum('lbcg,lcgp->lbcp', h.view(L, B, 3, self.g_out), q('w_out', self.w_out))
        o = o + self.b_out[:, None]
        return o.permute(0, 1, 3, 2).reshape(L, B, 768)

    def one(self, j, x, wq=None):
        """Run only network j (1-based level) on x: (B, 768)."""
        q = (lambda n, w: w) if wq is None else wq
        l = j - 1
        B = x.shape[0]
        xc = x.view(B, 256, 3).permute(0, 2, 1)
        h = torch.einsum('bcp,cgp->bcg', xc, q('w_in', self.w_in)[l]).reshape(B, -1)
        h = torch.relu(h + self.b_in[l])
        h = torch.relu(h @ q('w_m1', self.w_m1)[l] + self.b_m1[l])
        h = torch.relu(h @ q('w_m2', self.w_m2)[l] + self.b_m2[l])
        o = torch.einsum('bcg,cgp->bcp', h.view(B, 3, self.g_out), q('w_out', self.w_out)[l])
        o = o + self.b_out[l]
        return o.permute(0, 2, 1).reshape(B, 768)


@torch.no_grad()
def sample_float(model, mean, alpha, sigma, noise, keep=False):
    """Deterministic DDIM from x_0-level noise.  noise: (B, 768) tensor."""
    T = len(alpha) - 1
    k1, k2 = ddim_coefs(alpha, sigma)
    x = noise.clone()
    traj = [x.clone()]
    m = torch.as_tensor(mean, dtype=torch.float32)
    for j in range(T):
        x0 = m.expand_as(x) if j == 0 else model.one(j, x)
        x = float(k1[j]) * x + float(k2[j]) * x0
        if keep:
            traj.append(x.clone())
    return (x, traj) if keep else x
