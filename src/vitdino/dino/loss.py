
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class DINOLossConfig:
    out_dim: int = 65536
    teacher_temp: float = 0.04
    student_temp: float = 0.1
    center_momentum: float = 0.9

class DINOLoss(nn.Module):
    """
    Cross-entropy between softmax outputs of the teacher and student networks
    across multiple views. Implements teacher centering.
    """
    def __init__(self, cfg: DINOLossConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("center", torch.zeros(1, cfg.out_dim))

    def forward(self, student_out: torch.Tensor, teacher_out: torch.Tensor) -> torch.Tensor:
        """
        student_out: (B * Ncrops, D)
        teacher_out: (B * Nglobal, D)  # only global crops pass through teacher
        Returns: scalar loss
        """
        # temperature scaling
        student_logits = student_out / self.cfg.student_temp
        # teacher sharpen + centering
        teacher_logits = (teacher_out - self.center) / self.cfg.teacher_temp
        t_probs = F.softmax(teacher_logits, dim=-1).detach()
        s_log_probs = F.log_softmax(student_logits, dim=-1)

        # repeat teacher targets to match number of student crops per image
        # assume Ncrops = Nglobal + Nlocal and each image contributes Ncrops student outputs
        B_t = teacher_out.shape[0]  # equals B * Nglobal
        repeat_factor = student_out.shape[0] // B_t
        t_probs = t_probs.repeat_interleave(repeat_factor, dim=0)

        loss = -(t_probs * s_log_probs).sum(dim=-1).mean()
        # update center (EMA of teacher logits pre-softmax)
        batch_center = teacher_out.mean(dim=0, keepdim=True)
        self.center = self.center * self.cfg.center_momentum + batch_center * (1 - self.cfg.center_momentum)
        return loss
