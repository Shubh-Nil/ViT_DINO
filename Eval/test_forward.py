
import torch
from vitdino.models.vit import vit_base
from vitdino.dino.loss import DINOLoss, DINOLossConfig

def test_forward_and_loss():
    m = vit_base(patch_size=12, num_classes=None)
    B=2
    xg1 = torch.randn(B,3,224,224)
    xg2 = torch.randn(B,3,224,224)
    xl = torch.randn(B,3,112,112)
    with torch.no_grad():
        t_out = torch.cat([m.forward_features(xg1), m.forward_features(xg2)], dim=0)
    s_out = torch.cat([m.forward_features(xg1), m.forward_features(xg2), m.forward_features(torch.nn.functional.interpolate(xl, size=224))], dim=0)
    # small projection heads not used here; just check loss API with matching dims
    # map to logits dimension
    proj_dim = 65536
    t_out = torch.randn_like(t_out[:, :1]).repeat(1, proj_dim)  # fake logits
    s_out = torch.randn(s_out.shape[0], proj_dim)
    loss = DINOLoss(DINOLossConfig(out_dim=proj_dim))(s_out, t_out)
    assert loss.item() == loss.item()  # not NaN
