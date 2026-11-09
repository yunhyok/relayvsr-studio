"""Inference engine: RelayVSR (Wan keyframe model + Flash decoder) adapted for Windows.

The model code under ``models/`` is derived from the official RelayVSR inference
release (Apache-2.0, see ``third_party/LICENSE.RelayVSR``). This package adds:

* Triton-or-SDPA attention backends so the engine runs without Triton,
* FP8 weight-only storage of the Wan transformer (halves VRAM on RTX 40/50),
* CPU block offload for the Wan transformer,
* spatial tiling of the keyframe model for high resolutions on small GPUs,
* native 2x (``RelayVSR_X2``) and 4x checkpoints,
* a streaming pipeline with progress reporting, cancellation, audio muxing.
"""
