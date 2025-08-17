# Built ViT + DINO from Scratch

**Season:** Spring 2025

**Summary:** Implemented a Vision Transformer (**ViT-Base**, 12 encoder blocks, ~86M params) entirely from scratch in PyTorch — patch embedding, multi‑head self‑attention, MLP, class token, and learned positional embeddings. Integrated **DINO** self‑distillation with a **student–teacher** framework (EMA teacher, centering, temperature scheduling) to learn strong visual features **without manual labels**.

> ⚠️ This repository is for research/education. It is not optimized for speed or production training.

---

## Architecture

```
Images ─► Multi‑crop Augs (2 global + 6 local)
          ├──► Student ViT (trainable)
          └──► Teacher ViT (EMA, no grad)
             ▲
             └──── EMA weights from student
Loss: Cross‑entropy between teacher (sharpened, centered) and student distributions for each view.
```

- **ViT‑B/***p* (default **/12**): 12 layers, 768 dim, 12 heads, MLP ratio 4.0. Patch size is configurable; with /12 the parameter count is very close to the classic ~86M.
- **DINO**: teacher outputs use low temperature + running **center**; student uses higher temperature; teacher updated via **EMA** schedule.

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) (Optional) Prepare an unlabeled ImageFolder (any class names are ignored):
#    data/imagenet-mini/ ── classA/*.jpg, classB/*.jpg, ...
#    Or point to any folder of images with --data-root.

# 2) Sanity check (no dataset, uses random tensors)
python examples/run_sanity.py

# 3) Train DINO (toy settings; edit YAML for real runs)
python training/train_dino.py --config training/configs/dino_vitb12.yaml

# 4) Extract features & fit a linear probe (requires a labeled subset)
python examples/extract_features.py --data-root <labeled_folder> --ckpt outputs/dino-vitb12/checkpoints/student_last.pt
python examples/linear_probe.py --data-root <labeled_folder> --feats outputs/features.npy --labels outputs/labels.npy
```

---

## Repo Layout

```
ViT-DINO-from-scratch/
├─ src/vitdino/
│  ├─ models/vit.py        # PatchEmbed, MSA, MLP, Blocks, ViT
│  ├─ dino/loss.py         # DINO loss (centering + temp)
│  ├─ dino/engine.py       # Student/Teacher loop, EMA, schedules
│  └─ data/multicrop.py    # Multi-crop augmentations
├─ training/
│  ├─ train_dino.py        # CLI entrypoint
│  └─ configs/dino_vitb12.yaml
├─ examples/
│  ├─ run_sanity.py        # Forward pass + parameter count
│  ├─ extract_features.py  # Feature dump for linear probe
│  └─ linear_probe.py      # Logistic regression probe
├─ tests/
│  ├─ test_param_count.py  # ~86M params check (±1%)
│  └─ test_forward.py      # Shapes & loss smoke tests
├─ requirements.txt
├─ pyproject.toml
├─ LICENSE
└─ README.md
```

---

## Notes

- The implementation avoids external ViT/DINO libraries to keep the "from scratch" spirit.
- For real training, increase batch size / crops and train longer (see YAML).
- Patch size can be set to 12 (default) to match the project statement.
