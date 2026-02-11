"""Safety-Lens core: model-agnostic hook management and persona vector extraction."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Optional
from pathlib import Path


class LensHooks:
    """
    Context manager for temporary PyTorch forward hooks on transformer layers.

    Captures the hidden-state output of a specified layer during a forward pass.
    Hooks are always cleaned up on exit, even if an exception occurs.

    Usage:
        with LensHooks(model, layer_idx=12) as lens:
            model(**inputs)
            hidden = lens.activations["last"]  # [batch, seq_len, dim]
    """

    def __init__(self, model: nn.Module, layer_idx: int):
        self.model = model
        self.layer_idx = layer_idx
        self.activations: dict[str, torch.Tensor] = {}
        self._hooks: list = []
        self._target_module = self._resolve_layer(model, layer_idx)

    @staticmethod
    def _resolve_layer(model: nn.Module, idx: int) -> nn.Module:
        """Resolve a transformer block by index across HF architectures."""
        # LLaMA, Mistral, Qwen, Phi-3, Gemma
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers[idx]
        # GPT-2, GPT-J, GPT-Neo
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer.h[idx]
        # OPT
        if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
            return model.model.decoder.layers[idx]
        # MPT
        if hasattr(model, "transformer") and hasattr(model.transformer, "blocks"):
            return model.transformer.blocks[idx]

        raise ValueError(
            f"Could not resolve transformer layers for {type(model).__name__}. "
            "Supported architectures: LLaMA, Mistral, Qwen, GPT-2, GPT-J, OPT, MPT."
        )

    def _hook_fn(self, module: nn.Module, input: tuple, output) -> None:
        """Forward hook that captures hidden states."""
        hidden = output[0] if isinstance(output, tuple) else output
        self.activations["last"] = hidden.detach()

    def __enter__(self) -> LensHooks:
        hook = self._target_module.register_forward_hook(self._hook_fn)
        self._hooks.append(hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self.activations.clear()
