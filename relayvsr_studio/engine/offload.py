"""CPU block offload for the Wan transformer.

The 30 Wan blocks are kept in (pinned) host memory and streamed through two GPU slots:
while block *i* computes, block *i+1* is copied on a side stream. PCIe 4.0 x16 moves
the whole 10 GB (BF16) or 5 GB (FP8) model in well under a second, and the keyframe
model only runs every N frames, so offload costs little throughput while freeing
almost all weight memory. This makes 8-12 GB GPUs usable.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import torch
from torch import nn
from torch.func import functional_call

log = logging.getLogger(__name__)


class BlockOffloader:
    def __init__(self, blocks: Sequence[nn.Module], device: torch.device, *, slots: int = 2, pin_memory: bool = True):
        if slots < 2:
            raise ValueError("Block offload needs at least two GPU slots.")
        self.blocks = list(blocks)
        self.device = torch.device(device)
        self.slots = slots
        self.copy_stream = torch.cuda.Stream(device=self.device)
        self.cpu_states: list[dict[str, torch.Tensor]] = []
        pinned = pin_memory
        for block in self.blocks:
            state: dict[str, torch.Tensor] = {}
            for name, tensor in list(block.named_parameters()) + list(block.named_buffers()):
                host = tensor.detach().to("cpu")
                if pinned:
                    try:
                        host = host.pin_memory()
                    except RuntimeError as error:
                        log.warning("Pinned memory unavailable (%s); offload will be slower.", error)
                        pinned = False
                state[name] = host
                tensor.data = host  # release the device copy
            self.cpu_states.append(state)
        template = self.cpu_states[0]
        for state in self.cpu_states[1:]:
            if {k: (v.shape, v.dtype) for k, v in state.items()} != {k: (v.shape, v.dtype) for k, v in template.items()}:
                raise ValueError("Offloaded blocks must share one parameter layout.")
        self.slot_states = [
            {name: torch.empty_like(tensor, device=self.device) for name, tensor in template.items()} for _ in range(slots)
        ]
        self.slot_ready: list[torch.cuda.Event | None] = [None] * slots
        self.slot_free: list[torch.cuda.Event | None] = [None] * slots
        self.loaded = [-1] * slots
        self.bytes = sum(t.numel() * t.element_size() for state in self.cpu_states for t in state.values())
        log.info(
            "Block offload: %d blocks (%.2f GB) in %s host memory, %d GPU slots (%.2f GB).",
            len(self.blocks),
            self.bytes / 1e9,
            "pinned" if pinned else "pageable",
            slots,
            slots * sum(t.numel() * t.element_size() for t in template.values()) / 1e9,
        )

    def prefetch(self, index: int) -> None:
        slot = index % self.slots
        if self.loaded[slot] == index:
            return
        with torch.cuda.stream(self.copy_stream):
            if self.slot_free[slot] is not None:
                self.copy_stream.wait_event(self.slot_free[slot])
            for name, source in self.cpu_states[index].items():
                self.slot_states[slot][name].copy_(source, non_blocking=True)
            event = torch.cuda.Event()
            event.record(self.copy_stream)
        self.slot_ready[slot] = event
        self.loaded[slot] = index

    def run(self, index: int, block: nn.Module, *args, **kwargs):
        slot = index % self.slots
        self.prefetch(index)
        stream = torch.cuda.current_stream(self.device)
        stream.wait_event(self.slot_ready[slot])
        nxt = index + 1 if index + 1 < len(self.blocks) else 0
        self.prefetch(nxt)
        output = functional_call(block, self.slot_states[slot], args, kwargs)
        event = torch.cuda.Event()
        event.record(stream)
        self.slot_free[slot] = event
        return output


__all__ = ["BlockOffloader"]
