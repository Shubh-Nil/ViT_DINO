
import torch
from vitdino.models.vit import vit_base

m = vit_base(patch_size=12, num_classes=None)
x = torch.randn(2,3,224,224)
with torch.no_grad():
    feats = m.forward_features(x)
print("CLS features:", feats.shape)
print("Param count:", sum(p.numel() for p in m.parameters())/1e6, "M")
