import torch

from model import MAFNet


def test_mafnet_forward_backward_cpu():
    torch.manual_seed(0)
    model = MAFNet(n_class=2, input_h=64, input_w=64, pretrained=False)
    model.eval()
    rgb = torch.randn(1, 3, 64, 64)
    disparity = torch.randn(1, 3, 64, 64)
    out = model(rgb, disparity)
    assert out.shape == (1, 2, 64, 64)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert any(p.grad is not None for p in model.parameters())
