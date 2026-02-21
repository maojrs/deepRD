import argparse
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from contextlib import nullcontext
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import deepRD.tools.analysisTools as analysisTools
import deepRD.tools.trajectoryTools as trajectoryTools

# Condition token meanings:
# - pi: velocity
# - pim: velocity from two steps before (t-2)
# - dqi: relative position between the two particles
# - absdqi: relative distance between the two particles
# - ri: auxiliary variable from previous step (t-1)
# - rim: auxiliary variable from two steps before (t-2)
CONDITION_TO_MODEL_CFG = {
    "pidqiri": {"c_dim": 15, "num_blocks": 2},
    "piabsdqiri": {"c_dim": 13, "num_blocks": 2},
    "piabsdqiririm": {"c_dim": 19, "num_blocks": 2},
    "pipimabsdqiririm": {"c_dim": 25, "num_blocks": 4},
    "piabsdqiririmrimm": {"c_dim": 25, "num_blocks": 2},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train AdaLN diffusion dimer model.")
    parser.add_argument("--data-root", type=str, default=os.environ.get("DATA", ""))
    parser.add_argument("--boxsize", type=int, default=5)
    parser.add_argument("--num-trajs", type=int, default=2500)
    parser.add_argument("--condition", type=str, default="piabsdqiririmrimm", choices=list(CONDITION_TO_MODEL_CFG))
    parser.add_argument("--normal-file", type=str, default="normal_file.npz")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--diffusion-steps", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pretrained-path", type=str, default="AdaLN_multihead_normal_10.pt")
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--standardize", action="store_true", default=True)
    parser.add_argument("--no-standardize", dest="standardize", action="store_false")
    parser.add_argument("--loss-weighting", type=str, default="minsnr", choices=["minsnr", "none"])
    parser.add_argument("--snr-gamma", type=float, default=5.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.save_path is None:
        args.save_path = f"AdaLN_{args.condition}_dimer_T{args.diffusion_steps}"
    return args


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_data_root(data_root: str) -> str:
    if not data_root:
        raise ValueError(
            "Missing data root. Please set --data-root or export DATA before running."
        )
    return data_root.rstrip("/") + "/"


def relative_position(pos1, pos2, boundary_type="periodic", boxsize=5):
    if not isinstance(boxsize, (list, tuple, np.ndarray)):
        boxsize = [boxsize] * len(pos1)
    p1periodic = 1.0 * pos1
    if boundary_type == "periodic" and boxsize is not None:
        for i in range(3):
            if (pos2[i] - pos1[i]) > 0.5 * boxsize[i]:
                p1periodic[i] += boxsize[i]
            if (pos2[i] - pos1[i]) < -0.5 * boxsize[i]:
                p1periodic[i] -= boxsize[i]
    return pos2 - p1periodic


class DimerDataset(Dataset):
    def __init__(self, particle1, particle2, condition, standardize=True, normal_file_path="normal_file.npz"):
        self.condition = condition
        particle1_data = particle1.copy()
        particle2_data = particle2.copy()
        assert particle1_data.shape == particle2_data.shape

        normal_file = None
        if standardize:
            if not os.path.exists(normal_file_path):
                raise FileNotFoundError(
                    f"Normalization file not found: {normal_file_path}. "
                    "Please provide --normal-file or disable standardization."
                )
            normal_file = np.load(normal_file_path)

        if condition == "pidqiri":
            relative_pos = self.trajs_relative_position(particle1_data[:, :-1, 1:4], particle2_data[:, :-1, 1:4])
            if standardize:
                particle1_data[:, :-1, 4:7] /= normal_file["v_std"]
                particle2_data[:, :-1, 4:7] /= normal_file["v_std"]
                particle1_data[:, :-1, 8:] /= normal_file["ri_std"]
                particle2_data[:, :-1, 8:] /= normal_file["ri_std"]
                relative_pos = (relative_pos - normal_file["rel_pos_mean"]) / normal_file["rel_pos_std"]

            velocity = np.concatenate([particle1_data[:, :-1, 4:7], particle2_data[:, :-1, 4:7]], axis=2)
            ri = np.concatenate([particle1_data[:, :-1, 8:], particle2_data[:, :-1, 8:]], axis=2)
            rip1 = np.concatenate([particle1_data[:, 1:, 8:], particle2_data[:, 1:, 8:]], axis=2)
            X = np.concatenate([rip1, velocity, relative_pos, ri], axis=2)
            Y = rip1

        elif condition == "piabsdqiri":
            relative_pos = self.trajs_relative_position(particle1_data[:, :-1, 1:4], particle2_data[:, :-1, 1:4])
            relative_dis = np.linalg.norm(relative_pos, axis=2)
            if standardize:
                particle1_data[:, :-1, 4:7] /= normal_file["v_std"]
                particle2_data[:, :-1, 4:7] /= normal_file["v_std"]
                particle1_data[:, :-1, 8:] /= normal_file["ri_std"]
                particle2_data[:, :-1, 8:] /= normal_file["ri_std"]
                relative_dis = np.expand_dims(
                    (relative_dis - normal_file["rel_dis_mean"]) / normal_file["rel_dis_std"], axis=2
                )
            else:
                relative_dis = np.expand_dims(relative_dis, axis=2)

            velocity = np.concatenate([particle1_data[:, :-1, 4:7], particle2_data[:, :-1, 4:7]], axis=2)
            ri = np.concatenate([particle1_data[:, :-1, 8:], particle2_data[:, :-1, 8:]], axis=2)
            rip1 = np.concatenate([particle1_data[:, 1:, 8:], particle2_data[:, 1:, 8:]], axis=2)
            X = np.concatenate([rip1, velocity, relative_dis, ri], axis=2)
            Y = rip1

        elif condition == "piabsdqiririm":
            relative_pos = self.trajs_relative_position(particle1_data[:, :-1, 1:4], particle2_data[:, :-1, 1:4])
            relative_dis = np.linalg.norm(relative_pos, axis=2)[:, 1:]
            if standardize:
                particle1_data[:, :-1, 4:7] /= normal_file["v_std"]
                particle2_data[:, :-1, 4:7] /= normal_file["v_std"]
                particle1_data[:, :-1, 8:] /= normal_file["ri_std"]
                particle2_data[:, :-1, 8:] /= normal_file["ri_std"]
                relative_dis = np.expand_dims(
                    (relative_dis - normal_file["rel_dis_mean"]) / normal_file["rel_dis_std"], axis=2
                )
            else:
                relative_dis = np.expand_dims(relative_dis, axis=2)

            velocity = np.concatenate([particle1_data[:, 1:-1, 4:7], particle2_data[:, 1:-1, 4:7]], axis=2)
            ri = np.concatenate([particle1_data[:, 1:-1, 8:], particle2_data[:, 1:-1, 8:]], axis=2)
            rim = np.concatenate([particle1_data[:, :-2, 8:], particle2_data[:, :-2, 8:]], axis=2)
            rip1 = np.concatenate([particle1_data[:, 2:, 8:], particle2_data[:, 2:, 8:]], axis=2)
            X = np.concatenate([rip1, velocity, relative_dis, ri, rim], axis=2)
            Y = rip1

        elif condition == "pipimabsdqiririm":
            relative_pos = self.trajs_relative_position(particle1_data[:, :-1, 1:4], particle2_data[:, :-1, 1:4])
            relative_dis = np.linalg.norm(relative_pos, axis=2)[:, 1:]
            if standardize:
                particle1_data[:, :-1, 4:7] /= normal_file["v_std"]
                particle2_data[:, :-1, 4:7] /= normal_file["v_std"]
                particle1_data[:, :-1, 8:] /= normal_file["ri_std"]
                particle2_data[:, :-1, 8:] /= normal_file["ri_std"]
                relative_dis = np.expand_dims(
                    (relative_dis - normal_file["rel_dis_mean"]) / normal_file["rel_dis_std"], axis=2
                )
            else:
                relative_dis = np.expand_dims(relative_dis, axis=2)

            velocity = np.concatenate([particle1_data[:, 1:-1, 4:7], particle2_data[:, 1:-1, 4:7]], axis=2)
            velocity_m1 = np.concatenate([particle1_data[:, :-2, 4:7], particle2_data[:, :-2, 4:7]], axis=2)
            ri = np.concatenate([particle1_data[:, 1:-1, 8:], particle2_data[:, 1:-1, 8:]], axis=2)
            rim = np.concatenate([particle1_data[:, :-2, 8:], particle2_data[:, :-2, 8:]], axis=2)
            rip1 = np.concatenate([particle1_data[:, 2:, 8:], particle2_data[:, 2:, 8:]], axis=2)
            X = np.concatenate([rip1, velocity, velocity_m1, relative_dis, ri, rim], axis=2)
            Y = rip1

        elif condition == "piabsdqiririmrimm":
            relative_pos = self.trajs_relative_position(particle1_data[:, :-1, 1:4], particle2_data[:, :-1, 1:4])
            relative_dis = np.linalg.norm(relative_pos, axis=2)[:, 2:]
            if standardize:
                particle1_data[:, :-1, 4:7] /= normal_file["v_std"]
                particle2_data[:, :-1, 4:7] /= normal_file["v_std"]
                particle1_data[:, :-1, 8:] /= normal_file["ri_std"]
                particle2_data[:, :-1, 8:] /= normal_file["ri_std"]
                relative_dis = np.expand_dims(
                    (relative_dis - normal_file["rel_dis_mean"]) / normal_file["rel_dis_std"], axis=2
                )
            else:
                relative_dis = np.expand_dims(relative_dis, axis=2)

            velocity = np.concatenate([particle1_data[:, 2:-1, 4:7], particle2_data[:, 2:-1, 4:7]], axis=2)
            ri = np.concatenate([particle1_data[:, 2:-1, 8:], particle2_data[:, 2:-1, 8:]], axis=2)
            rim = np.concatenate([particle1_data[:, 1:-2, 8:], particle2_data[:, 1:-2, 8:]], axis=2)
            rimm = np.concatenate([particle1_data[:, :-3, 8:], particle2_data[:, :-3, 8:]], axis=2)
            rip1 = np.concatenate([particle1_data[:, 3:, 8:], particle2_data[:, 3:, 8:]], axis=2)
            X = np.concatenate([rip1, velocity, relative_dis, ri, rim, rimm], axis=2)
            Y = rip1
        else:
            raise ValueError(f"Unknown condition: {condition}")

        self.X = torch.tensor(X, dtype=torch.float32).reshape(-1, X.shape[2])
        self.Y = torch.tensor(Y, dtype=torch.float32).reshape(-1, Y.shape[2])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def trajs_relative_position(self, data_x1, data_x2):
        relative_pos = []
        for i in range(len(data_x1)):
            for j in range(len(data_x1[i])):
                relative_pos.append(relative_position(data_x1[i][j], data_x2[i][j]))
        return np.array(relative_pos).reshape(data_x1.shape)


def cosine_beta_schedule(T, s=0.008):
    steps = np.arange(T + 1, dtype=np.float64)
    alphas_cumprod = np.cos(((steps / T) + s) / (1 + s) * np.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, 1e-5, 0.999)
    return torch.tensor(betas, dtype=torch.float32)


def prepare_diffusion_schedules(T, device, schedule_type="cosine"):
    if schedule_type == "cosine":
        betas = cosine_beta_schedule(T).to(device)
    elif schedule_type == "linear":
        betas = torch.linspace(1e-4, 0.02, T).to(device)
    else:
        raise ValueError(f"Unknown schedule_type: {schedule_type}")

    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
    }


def forward_process(r_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    sqrt_alpha = sqrt_alphas_cumprod[t].unsqueeze(-1)
    sqrt_one_minus = sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
    noise = torch.randn_like(r_0)
    r_t = sqrt_alpha * r_0 + sqrt_one_minus * noise
    return r_t, noise


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor, T: int = 100):
        if t.dim() == 1:
            t = t[:, None]
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        args = t * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class AdaLNBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU())

    def forward(self, h, gamma, beta):
        h_norm = self.norm(h)
        h_mod = h_norm * (1 + gamma) + beta
        out = self.ff(h_mod)
        return h + out


class Encoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class New_DenoiseNet(nn.Module):
    def __init__(
        self,
        x_dim: int = 6,
        c_dim: int = 15,
        hidden_dim: int = 256,
        time_dim: int = 64,
        num_blocks: int = 2,
        diffusion_steps: int = 40,
    ):
        super().__init__()
        self.diffusion_steps = diffusion_steps
        self.enc_x = Encoder(x_dim, hidden_dim)
        self.enc_c = Encoder(c_dim, hidden_dim)

        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        self.context_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.blocks = nn.ModuleList([AdaLNBlock(hidden_dim) for _ in range(num_blocks)])
        self.to_gamma = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_blocks)])
        self.to_beta = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_blocks)])
        self.out_r1 = self._make_out_head(hidden_dim)
        self.out_r2 = self._make_out_head(hidden_dim)
        for lin in list(self.to_gamma) + list(self.to_beta):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def _make_out_head(self, hidden_dim):
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 3),
        )

    def forward(self, x_input, t):
        x_t = x_input[:, 0:6]
        conds = x_input[:, 6:]
        hx = self.enc_x(x_t)
        hc = self.enc_c(conds)
        t_feat = self.time_mlp(self.time_emb(t, T=self.diffusion_steps))
        combined_ctx = self.context_mlp(torch.cat([t_feat, hc], dim=-1))

        h = hx
        for blk, to_g, to_b in zip(self.blocks, self.to_gamma, self.to_beta):
            gamma = to_g(combined_ctx)
            beta = to_b(combined_ctx)
            h = blk(h, gamma, beta)

        eps_r1 = self.out_r1(h)
        eps_r2 = self.out_r2(h)
        return torch.cat([eps_r1, eps_r2], dim=-1)


def train_multihead(
    model,
    dataloader,
    schedule,
    device,
    pretrained_path="checkpoint.pt",
    save_path="checkpoint",
    epochs=20,
    T=40,
    lr=1e-5,
    use_amp=True,
    grad_clip=1.0,
    loss_weighting="minsnr",
    snr_gamma=5.0,
    resume=False,
    save_every=5,
    condition="",
    standardize=True,
):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    start_epoch = 0

    if resume and os.path.exists(pretrained_path):
        print(f"Resuming training from checkpoint: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}.")
    else:
        print("Training from scratch.")

    alphas_cumprod = schedule["alphas_cumprod"].to(device)
    sqrt_alphas_cumprod = schedule["sqrt_alphas_cumprod"].to(device)
    sqrt_one_minus_alphas_cumprod = schedule["sqrt_one_minus_alphas_cumprod"].to(device)

    is_cuda = isinstance(device, torch.device) and device.type == "cuda"
    scaler = GradScaler(enabled=(use_amp and is_cuda))
    autocast_ctx = autocast if (use_amp and is_cuda) else nullcontext

    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"[Train Epoch {epoch + 1}/{start_epoch + epochs}]")

        for x_input, r_target in pbar:
            x_input = x_input.to(device)
            r_target = r_target.to(device)
            B = x_input.size(0)
            t = torch.randint(0, T, (B,), device=device)

            r_t, noise = forward_process(r_target, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
            x_in = x_input.clone()
            x_in[:, 0:6] = r_t

            with autocast_ctx():
                eps_pred = model(x_in, t)
                alpha_bar_t = alphas_cumprod[t].unsqueeze(-1)
                if loss_weighting == "minsnr":
                    snr = alpha_bar_t / (1.0 - alpha_bar_t + 1e-8)
                    w = torch.minimum(snr, torch.full_like(snr, snr_gamma)) / (snr + 1e-8)
                else:
                    w = 1.0
                loss = (w * (eps_pred - noise) ** 2).mean()

            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": float(loss.detach())})

        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch + 1}] Avg loss: {avg_loss:.6f}")

        if (epoch + 1) % save_every == 0 or (epoch + 1) == (start_epoch + epochs):
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": float(total_loss),
                    "avg_loss": float(avg_loss),
                    "config": {
                        "condition": condition,
                        "lr": lr,
                        "standardize": standardize,
                        "num_blocks": len(model.blocks),
                        "diffusion_T": T,
                    },
                },
                f"{save_path}_epoch_{epoch + 1}.pt",
            )


def load_data(data_root: str, bsize: int, num_trajs: int):
    parent_directory = f"{data_root}stochasticClosure/dimer/boxsize{bsize}/benchmark/"
    fnamebase = parent_directory + "simMoriZwanzig_"
    parameter_dictionary = analysisTools.readParameters(parent_directory + "parameters")
    nfiles = parameter_dictionary["numFiles"]
    boxsize = parameter_dictionary["boxsize"]

    if bsize != boxsize:
        print(f"Warning: requested boxsize={bsize} but dataset boxsize={boxsize}")

    trajs = []
    max_files = min(num_trajs, nfiles)
    print(f"Loading {max_files} trajectories from {parent_directory} ...")
    for i in range(max_files):
        traj = trajectoryTools.loadTrajectory(fnamebase, i)
        trajs.append(traj)
        sys.stdout.write(f"File {i + 1} of {max_files} done.\r")
    print("\nAll data loaded.")

    trajs_arr = np.array(trajs)
    particle1 = trajs_arr[:, 0::2, :]
    particle2 = trajs_arr[:, 1::2, :]
    print(f"Trajectory array shape: {trajs_arr.shape}")
    return particle1, particle2


def build_model(condition: str, diffusion_steps: int):
    if condition not in CONDITION_TO_MODEL_CFG:
        raise ValueError(f"Unknown condition: {condition}")
    cfg = CONDITION_TO_MODEL_CFG[condition]
    return New_DenoiseNet(
        x_dim=6,
        c_dim=cfg["c_dim"],
        hidden_dim=256,
        time_dim=64,
        num_blocks=cfg["num_blocks"],
        diffusion_steps=diffusion_steps,
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    data_root = resolve_data_root(args.data_root)

    particle1, particle2 = load_data(data_root, args.boxsize, args.num_trajs)
    dataset = DimerDataset(
        particle1=particle1,
        particle2=particle2,
        condition=args.condition,
        standardize=args.standardize,
        normal_file_path=args.normal_file,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = build_model(args.condition, args.diffusion_steps)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    schedule = prepare_diffusion_schedules(args.diffusion_steps, device)
    train_multihead(
        model=model,
        dataloader=dataloader,
        schedule=schedule,
        device=device,
        pretrained_path=args.pretrained_path,
        save_path=args.save_path,
        epochs=args.epochs,
        T=args.diffusion_steps,
        lr=args.lr,
        use_amp=True,
        grad_clip=args.grad_clip,
        loss_weighting=args.loss_weighting,
        snr_gamma=args.snr_gamma,
        resume=args.resume,
        save_every=args.save_every,
        condition=args.condition,
        standardize=args.standardize,
    )


if __name__ == "__main__":
    main()
