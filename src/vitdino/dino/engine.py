
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Tuple
import math, time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from ..models.vit import vit_base
from .loss import DINOLoss, DINOLossConfig

@dataclass
class TrainConfig:
    image_size: int = 224
    patch_size: int = 12
    out_dim: int = 65536
    epochs: int = 1
    warmup_epochs: int = 0
    base_lr: float = 5e-4
    weight_decay: float = 0.04
    ema_momentum: float = 0.996
    ema_final_momentum: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def cosine_schedule(base: float, final: float, steps: int, step: int) -> float:
    if steps <= 1: return final
    return final + 0.5 * (base - final) * (1 + math.cos(math.pi * step / (steps - 1)))

class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int = 768, out_dim: int = 65536, hidden_dim: int = 2048, bn: bool = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim) if bn else nn.Identity(),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim) if bn else nn.Identity(),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim, bias=True),
        )
    def forward(self, x): return self.net(x)

class DINOPair(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.student_backbone = vit_base(patch_size=cfg.patch_size, num_classes=None)
        self.teacher_backbone = vit_base(patch_size=cfg.patch_size, num_classes=None)
        self.student_head = ProjectionHead(out_dim=cfg.out_dim)
        self.teacher_head = ProjectionHead(out_dim=cfg.out_dim)
        # start teacher equal to student
        self.teacher_backbone.load_state_dict(self.student_backbone.state_dict())
        self.teacher_head.load_state_dict(self.student_head.state_dict())
        for p in self.teacher_backbone.parameters(): p.requires_grad = False
        for p in self.teacher_head.parameters(): p.requires_grad = False

    def forward_student(self, x):  # x: (B, C, H, W)
        return self.student_head(self.student_backbone.forward_features(x))
    def forward_teacher(self, x):
        with torch.no_grad():
            return self.teacher_head(self.teacher_backbone.forward_features(x))

def train_dino(dataloader: Iterable, cfg: TrainConfig):
    device = cfg.device
    pair = DINOPair(cfg).to(device)
    criterion = DINOLoss(DINOLossConfig(out_dim=cfg.out_dim)).to(device)

    params = list(pair.student_backbone.parameters()) + list(pair.student_head.parameters())
    opt = AdamW(params, lr=cfg.base_lr, weight_decay=cfg.weight_decay)
    lr_sched = CosineAnnealingLR(opt, T_max=max(1, cfg.epochs*len(dataloader)))

    global_step = 0
    for epoch in range(cfg.epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        for batch in pbar:
            # batch: (global_views, local_views) tuples of tensors
            # we expect batch["global"] list of 2 tensors, batch["local"] list of N tensors
            g1, g2, locals_ = batch["global"][0].to(device), batch["global"][1].to(device), [t.to(device) for t in batch["local"]]
            # teacher on globals
            with torch.no_grad():
                t_out = torch.cat([pair.forward_teacher(g1), pair.forward_teacher(g2)], dim=0)  # (2B, D)
            # student on all crops
            s_ins = [g1, g2] + locals_
            s_out = torch.cat([pair.forward_student(v) for v in s_ins], dim=0)  # ((2+L)B, D)

            loss = criterion(s_out, t_out)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            lr_sched.step()

            # EMA update for teacher
            with torch.no_grad():
                m = cosine_schedule(cfg.ema_momentum, cfg.ema_final_momentum, cfg.epochs*len(dataloader), global_step)
                for ps, pt in zip(pair.student_backbone.parameters(), pair.teacher_backbone.parameters()):
                    pt.data.mul_(m).add_(ps.data, alpha=1-m)
                for ps, pt in zip(pair.student_head.parameters(), pair.teacher_head.parameters()):
                    pt.data.mul_(m).add_(ps.data, alpha=1-m)

            pbar.set_postfix({"loss": f"{loss.item():.3f}", "lr": f"{lr_sched.get_last_lr()[0]:.2e}"})
            global_step += 1

    return pair
