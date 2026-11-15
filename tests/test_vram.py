import pytest
import torch

from relayvsr_studio.engine.gpu import held_bytes, release_vram, restore_vram

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def test_release_and_restore_keep_values_and_aliasing():
    layer = torch.nn.Linear(64, 64).cuda().half()
    base = torch.arange(1 << 20, dtype=torch.float32, device="cuda")
    view = base[1000:2000].view(10, 100)  # shares storage with ``base``
    orphan = torch.ones(1 << 20, device="cuda")[::2]  # its base has no Python reference
    expected = (layer.weight.float().sum().item(), view.sum().item(), orphan.sum().item())
    before = torch.cuda.memory_allocated()

    held = release_vram()
    assert held_bytes(held) >= (1 << 20) * 8
    assert torch.cuda.memory_allocated() <= before - (1 << 20) * 8  # this test's tensors left the GPU
    assert not base.is_cuda and not view.is_cuda and not layer.weight.is_cuda and not orphan.is_cuda
    assert isinstance(layer.weight, torch.nn.Parameter)

    restore_vram(held)
    assert held == [] and base.is_cuda and view.is_cuda and layer.weight.is_cuda and orphan.is_cuda
    assert (layer.weight.float().sum().item(), view.sum().item(), orphan.sum().item()) == expected
    base[1000] = -5.0  # still one storage after the round trip
    assert view[0, 0].item() == -5.0
    assert layer(torch.ones(1, 64, device="cuda", dtype=torch.half)).shape == (1, 64)
