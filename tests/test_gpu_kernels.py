"""GPU tests: SDPA fallbacks must match the Triton kernels; FP8 round trip. Skipped without CUDA."""

import pytest
import torch

pytestmark = pytest.mark.gpu

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")


def _triton_ok():
    from relayvsr_studio.engine import backends

    return torch.cuda.is_available() and backends.triton_available()


@cuda
def test_fp8_linear_roundtrip():
    from relayvsr_studio.engine.quant import quantize_linear

    torch.manual_seed(0)
    linear = torch.nn.Linear(256, 128).to("cuda", torch.bfloat16)
    q = quantize_linear(linear, device=torch.device("cuda"))
    x = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
    ref = linear(x).float()
    out = q(x).float()
    rel = (out - ref).norm() / ref.norm()
    assert rel < 0.05, rel


@cuda
@pytest.mark.skipif(not _triton_ok(), reason="Triton required")
def test_flash_local_attention_sdpa_matches_triton():
    from relayvsr_studio.engine import backends
    from relayvsr_studio.engine.models.flash._local_attention_triton import triton_local_attention

    torch.manual_seed(0)
    h, w = 40, 56
    windows = ((48, 48), (48, 48), (24, 24), (24, 24))
    q = torch.randn(1, 8, h * w, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 2, len(windows) * h * w, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 2, len(windows) * h * w, 64, device="cuda", dtype=torch.bfloat16)
    ref = triton_local_attention(q, k, v, h, w, windows).float()
    out = backends.sdpa_local_attention(q, k, v, h, w, windows, shifted=False).float()
    assert torch.allclose(out, ref, atol=2e-2, rtol=2e-2), (out - ref).abs().max()


@cuda
@pytest.mark.skipif(not _triton_ok(), reason="Triton required")
def test_wan_spatial_attention_sdpa_matches_triton():
    from relayvsr_studio.engine.models.wan.spatial_attention import SpatialAttentionSpec, spatial_attention
    from relayvsr_studio.engine.models.wan.transformer import _sdpa_spatial_attention

    torch.manual_seed(0)
    h, w, frames = 30, 50, 3
    spec = SpatialAttentionSpec(1, h, w, 22, 40, 4, key_frames=frames, query_frame_offset=frames - 1)
    q = torch.randn(1, h * w, 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, frames * h * w, 4, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, frames * h * w, 4, 128, device="cuda", dtype=torch.bfloat16)
    ref = spatial_attention(q, k, v, spec).float()
    out = _sdpa_spatial_attention(q, k, v, spec).float()
    assert torch.allclose(out, ref, atol=2e-2, rtol=2e-2), (out - ref).abs().max()
