import torch
import torch.nn as nn
import math
import numpy as np
from typing import Dict


CONDITION_TO_MODEL_CFG: Dict[str, Dict[str, int]] = {
    "pidqiri": {"c_dim": 15, "num_blocks": 2},
    "piabsdqiri": {"c_dim": 13, "num_blocks": 2},
    "piabsdqiririm": {"c_dim": 19, "num_blocks": 2},
    "pipimabsdqiririm": {"c_dim": 25, "num_blocks": 4},
    "piabsdqiririmrimm": {"c_dim": 25, "num_blocks": 2},
}

# Normalization rules per condition:
# - tuple format: (slice(start, end), key)
# - key in {"v", "r", "rel_pos", "rel_dis"}
CONDITION_RULES: Dict[str, Dict[str, object]] = {
    "pidqiri": {
        "input_dim": 15,
        "clamp_start": 9,
        "norm_ops": [
            (slice(0, 3), "v"),
            (slice(3, 6), "v"),
            (slice(6, 9), "rel_pos"),
            (slice(9, 12), "r"),
            (slice(12, 15), "r"),
        ],
    },
    "piabsdqiri": {
        "input_dim": 13,
        "clamp_start": 7,
        "norm_ops": [
            (slice(0, 3), "v"),
            (slice(3, 6), "v"),
            (slice(6, 7), "rel_dis"),
            (slice(7, 10), "r"),
            (slice(10, 13), "r"),
        ],
    },
    "piabsdqiririm": {
        "input_dim": 19,
        "clamp_start": 7,
        "norm_ops": [
            (slice(0, 3), "v"),
            (slice(3, 6), "v"),
            (slice(6, 7), "rel_dis"),
            (slice(7, 10), "r"),
            (slice(10, 13), "r"),
            (slice(13, 16), "r"),
            (slice(16, 19), "r"),
        ],
    },
    "pipimabsdqiririm": {
        "input_dim": 25,
        "clamp_start": 13,
        "norm_ops": [
            (slice(0, 3), "v"),
            (slice(3, 6), "v"),
            (slice(6, 9), "v"),
            (slice(9, 12), "v"),
            (slice(12, 13), "rel_dis"),
            (slice(13, 16), "r"),
            (slice(16, 19), "r"),
            (slice(19, 22), "r"),
            (slice(22, 25), "r"),
        ],
    },
    "piabsdqiririmrimm": {
        "input_dim": 25,
        "clamp_start": 7,
        "norm_ops": [
            (slice(0, 3), "v"),
            (slice(3, 6), "v"),
            (slice(6, 7), "rel_dis"),
            (slice(7, 10), "r"),
            (slice(10, 13), "r"),
            (slice(13, 16), "r"),
            (slice(16, 19), "r"),
            (slice(19, 22), "r"),
            (slice(22, 25), "r"),
        ],
    },
}

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor, T: int = 100):
        if t.dim() == 1:
            t = t[:, None]
#         t = t / (T-1)  
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        args = t * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

# ---- 2) AdaLN residual block (unchanged) ----
class AdaLNBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.ff   = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )

    def forward(self, h, gamma, beta):
        h_norm = self.norm(h)
        h_mod  = h_norm * (1 + gamma) + beta  
        out    = self.ff(h_mod)
        return h + out  

# ---- 3) Encoder (unchanged) ----
class Encoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
    def forward(self, x):
        return self.net(x)

# ---- 4) Main network with Deep Injection architecture ----
# ---- 4) Corrected main network ----
class New_DenoiseNet(nn.Module):
    def __init__(
        self,
        x_dim: int = 6,
        c_dim: int =15,
        hidden_dim: int = 256,
        time_dim: int = 64,
        num_blocks: int = 2,
        out_dim: int = 6           
    ):
        super().__init__()
        self.enc_x = Encoder(x_dim, hidden_dim)
        self.enc_c = Encoder(c_dim, hidden_dim)

        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )

        self.context_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.blocks   = nn.ModuleList([AdaLNBlock(hidden_dim) for _ in range(num_blocks)])
        self.to_gamma = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_blocks)])
        self.to_beta  = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_blocks)])

        self.out_r1 = self._make_out_head(hidden_dim)
        self.out_r2 = self._make_out_head(hidden_dim)

    def _make_out_head(self, hidden_dim):
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 3)
        )

    def forward(self, x_input, t):
        # Input layout: [B, 6 + c_dim]
        x_t = x_input[:, 0:6]
        conds = x_input[:, 6:]

        hx = self.enc_x(x_t)
        hc = self.enc_c(conds)

        # Keep T consistent with training settings.
        t_feat = self.time_mlp(self.time_emb(t, T=40)) 

        combined_ctx = self.context_mlp(torch.cat([t_feat, hc], dim=-1))

        h = hx 
        for blk, to_g, to_b in zip(self.blocks, self.to_gamma, self.to_beta):
            gamma = to_g(combined_ctx)
            beta  = to_b(combined_ctx)
            h = blk(h, gamma, beta)

        eps_r1 = self.out_r1(h)
        eps_r2 = self.out_r2(h)
        
        return torch.cat([eps_r1, eps_r2], dim=-1)
def cosine_beta_schedule(T, s=0.008):
    steps = np.arange(T + 1, dtype=np.float64)
    alphas_cumprod = np.cos(((steps / T) + s) / (1 + s) * np.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # normalize to start at 1

    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, 1e-5, 0.999)  # Prevent extreme values.
    return torch.tensor(betas, dtype=torch.float32)

def prepare_diffusion_schedules(T, device, schedule_type='cosine'):
    if schedule_type == 'cosine':
        betas = cosine_beta_schedule(T).to(device)
    elif schedule_type == 'linear':
        betas = torch.linspace(1e-4, 0.02, T).to(device)
    else:
        raise ValueError(f"Unknown schedule_type: {schedule_type}")

    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1. - alphas_cumprod)
    }
    
def forward_process(r_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    sqrt_alpha = sqrt_alphas_cumprod[t].unsqueeze(-1)
    sqrt_one_minus = sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
    noise = torch.randn_like(r_0)
    r_t = sqrt_alpha * r_0 + sqrt_one_minus * noise
    return r_t, noise


def sample_from_model_vrr(model, input_tensor, T, schedule, device):
    model.eval()
    B = input_tensor.shape[0]
    x_cond = input_tensor 
    r_t = torch.randn(B, 6).to(device)

    betas = schedule["betas"].to(device)
    alphas = schedule["alphas"].to(device)
    alphas_cumprod = schedule["alphas_cumprod"].to(device)

    with torch.no_grad():  # Inference-only sampling.
        for t in reversed(range(T-1)):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            x_input = torch.cat([r_t, x_cond], dim=1)
            eps_pred = model(x_input, t_batch)

            sqrt_recip_alpha = torch.sqrt(1. / alphas[t])
            sqrt_one_minus_alpha_bar = torch.sqrt(1. - alphas_cumprod[t])

            r_0_est = (r_t - ((1 - alphas[t]) * eps_pred / sqrt_one_minus_alpha_bar)) * sqrt_recip_alpha

            if t > 0:
                noise = torch.randn_like(r_t)
                sigma_t = torch.sqrt(betas[t])
                r_t = r_0_est + sigma_t * noise
            else:
                r_t = r_0_est

    return r_t



class diffusionSampler:
    """
    Lightweight loader/sampler for inference-only molecular simulation.
    API kept compatible with existing usage:
      - diffusionSampler(model_path, conditionedOn, ...)
      - sample(conditionedVars)
    """

    def __init__(self, model_path, conditionedOn, device="cuda", normalize=True, norm_params=None, T=40):
        if conditionedOn not in CONDITION_TO_MODEL_CFG or conditionedOn not in CONDITION_RULES:
            raise ValueError(f"Unknown conditioned variables: {conditionedOn}")

        if device == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.conditionedOn = conditionedOn
        self.rule = CONDITION_RULES[conditionedOn]
        self.T = T
        self.normalize = normalize

        model_cfg = CONDITION_TO_MODEL_CFG[conditionedOn]
        self.model = New_DenoiseNet(
            x_dim=6,
            c_dim=model_cfg["c_dim"],
            hidden_dim=256,
            time_dim=64,
            num_blocks=model_cfg["num_blocks"],
            out_dim=6,
        )

        ckpt = torch.load(model_path, map_location=self.device)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.schedule = prepare_diffusion_schedules(T, self.device)
        self._init_norm_params(ckpt, norm_params)

    def _init_norm_params(self, ckpt, norm_params):
        self.r_std = None
        self.v_std = None
        self.rel_pos_mean = None
        self.rel_pos_std = None
        self.rel_dis_mean = None
        self.rel_dis_std = None

        if not self.normalize:
            return

        if norm_params is None and isinstance(ckpt, dict):
            norm_params = ckpt.get("norm_params")
        if norm_params is None:
            raise ValueError("normalize=True but no 'norm_params' provided and none found in checkpoint.")

        def _to_vec(x):
            return torch.as_tensor(x, dtype=torch.float32, device=self.device).flatten()

        required_keys = {"ri_std"}
        op_keys = {key for _, key in self.rule["norm_ops"]}
        if "v" in op_keys:
            required_keys.add("v_std")
        if "rel_pos" in op_keys:
            required_keys.update({"rel_pos_mean", "rel_pos_std"})
        if "rel_dis" in op_keys:
            required_keys.update({"rel_dis_mean", "rel_dis_std"})

        missing = [k for k in sorted(required_keys) if k not in norm_params]
        if missing:
            raise KeyError(f"Missing keys in norm_params: {missing}")

        self.r_std = _to_vec(norm_params["ri_std"])
        if "v_std" in norm_params:
            self.v_std = _to_vec(norm_params["v_std"])
        if "rel_pos_mean" in norm_params:
            self.rel_pos_mean = _to_vec(norm_params["rel_pos_mean"])
        if "rel_pos_std" in norm_params:
            self.rel_pos_std = _to_vec(norm_params["rel_pos_std"])
        if "rel_dis_mean" in norm_params:
            self.rel_dis_mean = _to_vec(norm_params["rel_dis_mean"])
        if "rel_dis_std" in norm_params:
            self.rel_dis_std = _to_vec(norm_params["rel_dis_std"])

    def _normalize_inputs(self, input_tensor: torch.Tensor) -> torch.Tensor:
        for slc, key in self.rule["norm_ops"]:
            if key == "v":
                input_tensor[slc] = input_tensor[slc] / self.v_std
            elif key == "r":
                input_tensor[slc] = input_tensor[slc] / self.r_std
            elif key == "rel_pos":
                input_tensor[slc] = (input_tensor[slc] - self.rel_pos_mean) / self.rel_pos_std
            elif key == "rel_dis":
                input_tensor[slc] = (input_tensor[slc] - self.rel_dis_mean) / self.rel_dis_std
        return input_tensor

    def sample(self, conditionedVars):
        input_tensor = torch.as_tensor(conditionedVars, dtype=torch.float32, device=self.device).flatten()

        expected_dim = int(self.rule["input_dim"])
        if input_tensor.numel() != expected_dim:
            raise ValueError(
                f"conditionedVars expected length {expected_dim}, got {input_tensor.numel()} "
                f"(condition={self.conditionedOn})"
            )

        clamp_start = int(self.rule["clamp_start"])
        input_tensor[clamp_start:] = torch.clamp(input_tensor[clamp_start:], -0.10, 0.10)

        if self.normalize:
            input_tensor = self._normalize_inputs(input_tensor)

        input_tensor = input_tensor.unsqueeze(0)
        r_sample = sample_from_model_vrr(self.model, input_tensor, self.T, self.schedule, self.device).squeeze(0)

        if self.normalize:
            r_sample[0:3] = r_sample[0:3] * self.r_std
            r_sample[3:6] = r_sample[3:6] * self.r_std

        return r_sample.detach().cpu().numpy()
