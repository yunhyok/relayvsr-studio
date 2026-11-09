# Third-party notices

RelayVSR Studio is licensed under the Apache License 2.0 (see `LICENSE`). It builds on
the following projects.

## RelayVSR (inference code)

`relayvsr_studio/engine/models/` is derived from the official RelayVSR inference release
by Xijun Wang, Xin Li, Zirui Lang, Suhang Yao, Haoran Li and Zhibo Chen (USTC),
<https://github.com/kopperx/RelayVSR>, Apache License 2.0 (`third_party/LICENSE.RelayVSR`).
The Wan transformer implementation retains its upstream copyright notice
(Copyright 2025 The Wan Team and The HuggingFace Team).

Modifications in this repository: optional Triton import with an SDPA fallback, native 2x
geometry, FP8 weight-only storage hooks, CPU block offload hooks, spatial tiling,
residual detail control, and packaging changes.

Paper: *RelayVSR: Large-Small Model Collaboration for Efficient Real-World Video
Super-Resolution*, arXiv:2609.37850.

## LightX2V

The tiled SDPA local-attention formulation in `relayvsr_studio/engine/backends.py` follows
the RelayVSR SDPA backend of LightX2V (ModelTC), <https://github.com/ModelTC/LightX2V>,
Apache License 2.0.

## Model weights (downloaded at run time, not redistributed here)

* `kopper/RelayVSR` (native 4x), published by the RelayVSR authors on Hugging Face.
* `lightx2v/RelayVSR_X2` (native 2x), published by the LightX2V team on Hugging Face.

Check the respective model cards for their license terms before commercial use.

## Runtime dependencies (installed into the managed environment at first run)

PyTorch (BSD-3), Triton / triton-windows (MIT), Diffusers and Accelerate (Apache-2.0),
PyAV (BSD-3; its wheels bundle FFmpeg and codecs under LGPL/GPL terms), NumPy (BSD-3),
Pillow (MIT-CMU), safetensors (Apache-2.0), einops (MIT), huggingface_hub (Apache-2.0),
nvidia-ml-py (BSD), psutil (BSD-3), uv (MIT/Apache-2.0).

The GUI uses PySide6 (LGPL-3.0; dynamically linked, unmodified) and is packaged with
PyInstaller (GPL with bootloader exception) and Inno Setup.
