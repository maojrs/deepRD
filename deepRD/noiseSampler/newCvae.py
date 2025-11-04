import numpy as np
import matplotlib.pyplot as plt
import math
import joblib
import torch
import torch.nn as nn
import torchvision
import deepRD.tools.trajectoryTools as trajectoryTools
from torch.utils.data import Dataset, DataLoader
from torch import optim
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(128,128)):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.SiLU(), nn.LayerNorm(h)]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

def reparam(mu, logvar):
    eps = torch.randn_like(mu)
    return mu + eps * torch.exp(0.5 * logvar)

class DiagGaussianHead(nn.Module):
    """Outputs (mu, log_sigma) for R^3."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = MLP(in_dim, out_dim, hidden=(128,128))
    def forward(self, x):
        out = self.mlp(x)
        mu, log_sig = out[..., :3], out[..., 3:]
        return mu, log_sig

# ---------- CVAE ----------
class CVAE(nn.Module):
    def __init__(self, idim=3, cdim=6, zdim=3):
        super().__init__()
        self.cdim, self.zdim = cdim, zdim
        self.encoder = MLP(idim + cdim, out_dim=2*zdim, hidden=(128,128))
        self.prior   = MLP(cdim, out_dim=2*zdim, hidden=(128,128))
        self.decoder = DiagGaussianHead(zdim + cdim, 2*idim)
        
    def attach_normalizers(self, scaler_rnext, scaler_v, scaler_r):
        """Attach normalization scalers for automatic preprocessing."""
        self.scaler_rnext = scaler_rnext
        self.scaler_v = scaler_v
        self.scaler_r = scaler_r

    def encode(self, r_next, c):
        q = self.encoder(torch.cat([r_next, c], dim=-1))
        q_mu, q_logv = q.split(self.zdim, dim=-1)
        return q_mu, q_logv

    def prior_params(self, c):
        p = self.prior(c)
        p_mu, p_logv = p.split(self.zdim, dim=-1)
        return p_mu, p_logv

    def decode(self, z, c):
        mu, log_sig = self.decoder(torch.cat([z, c], dim=-1))
        return mu, log_sig

    def forward(self, r_next, c):
        p_mu, p_logv = self.prior_params(c)
        q_mu, q_logv = self.encode(r_next, c)
        z = reparam(q_mu, q_logv)
        dec_out = self.decode(z, c)
        return dec_out, (q_mu, q_logv), (p_mu, p_logv)

    @torch.no_grad()
    def sample_torch(self, c, T=1.0):
        """
        Sampling from torch tensor, no (de)normalisation.
        """
        p_mu, p_logv = self.prior_params(c)
        z = reparam(p_mu, p_logv)  # sample from p(z|c)
        
        mu, log_sig = self.decode(z, c)
        r = mu + torch.exp(log_sig) * torch.randn_like(mu) * T
        return r
    
    @torch.no_grad()
    def sample(self, c_n_np, Tr=1.0, Tz=1.0, device=None):
        """
        Sample r_{n+1} in physical units given c = [v_n, r_n] as NumPy array. 
        Built in normalisation of input and denormalisation of output.

        Args:
            c_n_np (np.ndarray): shape (..., cdim)
            Tr (float): temperature scaling factor for stochasticity
            Tz (float): temperature scaling factor in latent space for stochasticity
            device (torch.device): GPU/CPU device to use (optional)

        Returns:
            np.ndarray: generated r_{n+1} in same physical scale as input
        """
        if device is None:
            device = next(self.parameters()).device

        # --- Normalize inputs ---
        v_n_np, r_n_np = c_n_np[...,:3], c_n_np[..., 3:]

        single_sample = False
        if r_n_np.ndim == 1:
            r_n_np = r_n_np.reshape(1, -1)
            single_sample = True
        if v_n_np.ndim == 1:
            v_n_np = v_n_np.reshape(1, -1)

        v_norm = self.scaler_v.transform(v_n_np)
        r_norm = self.scaler_r.transform(r_n_np)

        c = np.concatenate([v_norm, r_norm], axis=-1)

        # --- Convert to torch tensor ---
        c_t = torch.tensor(c, dtype=torch.float32, device=device)

        # --- Sample from conditional prior and decode ---
        p_mu, p_logv = self.prior_params(c_t)
        z = p_mu + torch.exp(0.5 * p_logv) * torch.randn_like(p_mu) * Tz
        mu_r, log_sig_r = self.decode(z, c_t)
        r_next_norm = mu_r + torch.exp(log_sig_r) * torch.randn_like(mu_r) * Tr

        # --- De-normalize to physical scale ---
        r_next_np = r_next_norm.cpu().numpy()
        r_next_phys = self.scaler_rnext.inverse_transform(r_next_np)

        if single_sample:
            r_next_phys = r_next_phys.squeeze()  # (3,)

        return r_next_phys