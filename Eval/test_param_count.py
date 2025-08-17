
from vitdino.models.vit import vit_base

def test_param_count_close():
    m = vit_base(patch_size=12, num_classes=None)
    n = sum(p.numel() for p in m.parameters())
    # ViT-Base ~86M; allow ±1.5M due to patch embedding size change
    assert abs(n - 86_000_000) < 1_500_000
