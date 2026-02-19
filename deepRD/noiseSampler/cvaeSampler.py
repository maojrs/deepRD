import numpy as np
import matplotlib.pyplot as plt
import math
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
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

def reparam(mu, logvar, Tz=1.0):
    """
    Reparameterization for a diagonal Gaussian with parameters (mu, logvar).
    logvar = log(variance), so std = exp(0.5 * logvar).
    """
    eps = torch.randn_like(mu)
    return mu + eps * torch.exp(0.5 * logvar) * Tz

class DiagGaussianHead(nn.Module):
    """Outputs (mu, log_sigma) for R^3."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        assert out_dim % 2 == 0, "out_dim must be even: 2 * D"
        self.D = out_dim//2
        self.mlp = MLP(in_dim, out_dim, hidden=(128,128))
    def forward(self, x):
        out = self.mlp(x)
        mu, log_sigma = out[..., :self.D], out[..., self.D:]
        return mu, log_sigma

# ---------- CVAE ----------
class CVAE(nn.Module):
    def __init__(self, zdim=3, system_type="bistable", cond_type="piri"):
        super().__init__()
        
        assert system_type in ("bistable", "dimer")
        self.system_type = system_type
        self.cond_type = cond_type
        self.zdim = zdim
        
        self.idim, self.cdim = self.assign_dims(system_type=system_type, cond_type=cond_type)
        
        # networks
        self.encoder = MLP(self.idim + self.cdim, out_dim=2*zdim, hidden=(128,128))
        self.prior   = MLP(self.cdim, out_dim=2*zdim, hidden=(128,128))
        self.decoder = DiagGaussianHead(zdim + self.cdim, 2*self.idim)
        
        # normalisers
        self.scaler_r = None
        self.scaler_c = None

    @staticmethod
    def assign_dims(system_type: str, cond_type: str) -> tuple[int, int]:
        # idim by system
        idim_map = {"bistable": 3, "dimer": 6}
        assert system_type in idim_map, f"Unknown system_type={system_type!r}"
        idim = idim_map[system_type]

        # cdim mapping by (system_type, cond_type)
        cdim_map = {
            "bistable": {
                "piri": 6,
                "piririm": 9,
                "pipimri": 9,
                "piririmrimm": 12,
                "pipimririm": 12,
            },
            "dimer": {
                "pidqiri": 13,
                "dqidpiri": 8,
                "dqidpiririm": 14,
                "pipimririm": 24,
                "pipimririmrimm": 30,
                "pipimdqiririm": 25,
                "pipimdpiririm": 25,
                "pipimdqidpiririm": 26,
            },
        }

        assert system_type in cdim_map, f"Missing cdim map for system_type={system_type!r}"
        assert cond_type in cdim_map[system_type], (
            f"Unsupported cond_type={cond_type!r} for system_type={system_type!r}. "
            f"Supported: {tuple(cdim_map[system_type].keys())}"
        )

        cdim = cdim_map[system_type][cond_type]
        return idim, cdim
        
    def attach_normalizers(self, scaler_r, scaler_c):
        """Attach normalization scalers for automatic preprocessing."""
        self.scaler_r = scaler_r
        self.scaler_c = scaler_c

    def set_temps(self, Tr=None, Tz=None):
        """Set global sampling temperatures. Call with no args to unset."""
        if Tr is None and Tz is None:
            # remove attributes if they exist
            for name in ("Tr", "Tz"):
                if hasattr(self, name):
                    delattr(self, name)
        else:
            if Tr is not None:
                self.Tr = Tr
            if Tz is not None:
                self.Tz = Tz

    def encode(self, r_next, c):
        q = self.encoder(torch.cat([r_next, c], dim=-1))
        q_mu, q_logvar = q.split(self.zdim, dim=-1)
        return q_mu, q_logvar

    def prior_params(self, c):
        p = self.prior(c)
        p_mu, p_logvar = p.split(self.zdim, dim=-1)
        return p_mu, p_logvar

    def decode(self, z, c):
        mu, log_sigma = self.decoder(torch.cat([z, c], dim=-1))
        return mu, log_sigma

    def forward(self, r_next, c):
        p_mu, p_logv = self.prior_params(c)
        q_mu, q_logv = self.encode(r_next, c)
        z = reparam(q_mu, q_logv)
        dec_out = self.decode(z, c)
        return dec_out, (q_mu, q_logv), (p_mu, p_logv)   
    
    # ----- sampling ----- #
    @torch.no_grad()
    def sample_torch(self, c, Tr=1.0, Tz=1.0):
        """
        Sampling from torch tensor, no (de)normalisation.
        """
        p_mu, p_logv = self.prior_params(c)
        z = reparam(p_mu, p_logv, Tz=Tz)  # sample from p(z|c)
        
        mu, log_sigma = self.decode(z, c)
        r = mu + torch.exp(log_sigma) * torch.randn_like(mu) * Tr
        return r
    
    @torch.no_grad()
    def sample(self, c_n_np, Tr=1.0, Tz=1.0, device=None):
        """
        Sample r_{n+1} in physical units given c_n_np in physical units as NumPy array. 
        Built in normalisation of input and denormalisation of output.

        Args:
            c_n_np (np.ndarray): shape (..., cdim)
            Tr (float): temperature scaling factor for stochasticity
            Tz (float): temperature scaling factor in latent space for stochasticity
            device (torch.device): GPU/CPU device to use (optional)
            
        Handles every conditioning type provided the corresponding scaler.

        Returns:
            np.ndarray: generated r_{n+1} in same physical scale as input
        """
        if self.scaler_c is None or self.scaler_r is None:
            raise ValueError("Call attach_normalizers(...) before sample().")

        if device is None:
            device = next(self.parameters()).device

        # overwrite with global temps if defined
        if hasattr(self, "Tr"):
            Tr = self.Tr
        if hasattr(self, "Tz"):
            Tz = self.Tz

        # ---- Reshape and normalise ---- #
        c_n_np = np.asarray(c_n_np, dtype=np.float32)
        single_sample = False
        if c_n_np.ndim == 1:
            c_n_np = c_n_np.reshape(1, -1)
            single_sample = True
            
        c_norm = self.scaler_c.transform(c_n_np)

        # --- Convert to torch tensor ---
        c_t = torch.from_numpy(c_norm).to(device=device)

        r_next_norm_t = self.sample_torch(c_t, Tr=Tr, Tz=Tz)

        # --- De-normalize to physical scale ---
        r_next_np = r_next_norm_t.cpu().numpy()
        r_next_phys = self.scaler_r.inverse_transform(r_next_np)

        if single_sample:
            r_next_phys = r_next_phys.squeeze()  # (3,)

        return r_next_phys