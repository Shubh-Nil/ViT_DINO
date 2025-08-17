
import argparse, os, numpy as np, torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from vitdino.models.vit import vit_base

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True)  # Labeled ImageFolder
    ap.add_argument("--ckpt", type=str, default=None)        # student_backbone.pt
    ap.add_argument("--patch-size", type=int, default=12)
    ap.add_argument("--out-dir", type=str, default="outputs")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = vit_base(patch_size=args.patch_size, num_classes=None).to(device)
    if args.ckpt and os.path.isfile(args.ckpt):
        model.load_state_dict(torch.load(args.ckpt, map_location=device))

    t = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                            transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))])
    ds = datasets.ImageFolder(args.data_root, transform=t)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)

    all_feats, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            f = model.forward_features(x).cpu().numpy()
            all_feats.append(f); all_labels.append(y.numpy())
    feats = np.concatenate(all_feats, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    np.save(os.path.join(args.out_dir, "features.npy"), feats)
    np.save(os.path.join(args.out_dir, "labels.npy"), labels)
    print("Saved features to", args.out_dir)

if __name__ == "__main__":
    main()
