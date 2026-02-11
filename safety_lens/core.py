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


class SafetyLens:
    """
    The MRI Machine. Extracts persona vectors and scans model internals.

    Usage:
        lens = SafetyLens(model, tokenizer)
        vec = lens.extract_persona_vector(pos_texts, neg_texts, layer_idx=15)
        score = lens.scan(input_ids, vec, layer_idx=15)
    """

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    def _collect_last_token_states(
        self, texts: list[str], layer_idx: int
    ) -> torch.Tensor:
        """Run texts through the model and collect last-token hidden states."""
        states = []
        for text in texts:
            with LensHooks(self.model, layer_idx) as lens:
                inputs = self.tokenizer(
                    text, return_tensors="pt", padding=False, truncation=True
                ).to(self.device)
                with torch.no_grad():
                    self.model(**inputs)
                last_state = lens.activations["last"][0, -1, :]
                states.append(last_state)
        return torch.stack(states)

    def extract_persona_vector(
        self,
        pos_texts: list[str],
        neg_texts: list[str],
        layer_idx: int,
    ) -> torch.Tensor:
        """
        PV-EAT: Extract a persona vector via difference-in-means.

        The resulting vector points from the negative centroid toward the
        positive centroid in activation space, then is L2-normalized.
        """
        pos_mean = self._collect_last_token_states(pos_texts, layer_idx).mean(dim=0)
        neg_mean = self._collect_last_token_states(neg_texts, layer_idx).mean(dim=0)
        direction = pos_mean - neg_mean
        return direction / torch.norm(direction)

    def scan(
        self, input_ids: torch.Tensor, vector: torch.Tensor, layer_idx: int
    ) -> float:
        """
        Compute alignment between a forward pass and a persona vector.

        Returns the dot-product projection of the last token's hidden state
        onto the persona vector.
        """
        with LensHooks(self.model, layer_idx) as lens:
            with torch.no_grad():
                self.model(input_ids.to(self.device))
            state = lens.activations["last"][0, -1, :]
            return torch.dot(state, vector.to(state.device)).item()

    def scan_all_layers(
        self,
        input_ids: torch.Tensor,
        vectors: dict[int, torch.Tensor],
    ) -> dict[int, float]:
        """Scan multiple layers, returning layer_idx -> alignment score."""
        scores = {}
        for layer_idx, vector in vectors.items():
            scores[layer_idx] = self.scan(input_ids, vector, layer_idx)
        return scores

    @staticmethod
    def save_vector(vector: torch.Tensor, path: str | Path) -> None:
        """Save a persona vector to disk."""
        torch.save(vector, path)

    @staticmethod
    def load_vector(path: str | Path) -> torch.Tensor:
        """Load a persona vector from disk."""
        return torch.load(path, weights_only=True)

    def quick_scan(
        self,
        text: str,
        layer_idx: int,
        persona_names: Optional[list[str]] = None,
    ) -> dict[str, float]:
        """
        One-liner scan using pre-built stimulus sets.

        Args:
            text: The prompt to scan.
            layer_idx: Layer to inspect.
            persona_names: Which personas to scan for. Defaults to all available.

        Returns:
            Mapping of persona name -> alignment score.
        """
        from safety_lens.vectors import STIMULUS_SETS

        if persona_names is None:
            persona_names = list(STIMULUS_SETS.keys())

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        results = {}
        for name in persona_names:
            stim = STIMULUS_SETS[name]
            vec = self.extract_persona_vector(stim["pos"], stim["neg"], layer_idx)
            results[name] = self.scan(inputs.input_ids, vec, layer_idx)
        return results
