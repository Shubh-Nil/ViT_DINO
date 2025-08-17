
from __future__ import annotations
import argparse, os, yaml, torch
from vitdino.dino.engine import TrainConfig, train_dino
from vitdino.data.multicrop import build_loader

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))
    out_dir = cfg.get("output_dir", "outputs/dino-vitb12")
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)

    tcfg = TrainConfig(
        image_size=cfg.get("image_size", 224),
        patch_size=cfg.get("patch_size", 12),
        out_dim=cfg.get("out_dim", 65536),
        epochs=cfg.get("epochs", 1),
        base_lr=cfg.get("base_lr", 5e-4),
        weight_decay=cfg.get("weight_decay", 0.04),
        ema_momentum=cfg.get("ema_momentum", 0.996),
        ema_final_momentum=cfg.get("ema_final_momentum", 1.0),
    )
    loader = build_loader(cfg["data_root"], batch_size=cfg.get("batch_size", 4), num_workers=cfg.get("num_workers", 2), image_size=tcfg.image_size)
    pair = train_dino(loader, tcfg)

    torch.save(pair.student_backbone.state_dict(), os.path.join(out_dir, "checkpoints", "student_backbone.pt"))
    torch.save(pair.student_head.state_dict(), os.path.join(out_dir, "checkpoints", "student_head.pt"))
    torch.save(pair.teacher_backbone.state_dict(), os.path.join(out_dir, "checkpoints", "teacher_backbone.pt"))
    torch.save(pair.teacher_head.state_dict(), os.path.join(out_dir, "checkpoints", "teacher_head.pt"))
    torch.save({"cfg": tcfg.__dict__}, os.path.join(out_dir, "checkpoints", "meta.pt"))
    print("Saved checkpoints to", os.path.join(out_dir, "checkpoints"))

if __name__ == "__main__":
    main()
