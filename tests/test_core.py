import pytest
import torch
from safety_lens.core import LensHooks


class TestLensHooks:
    def test_captures_activations(self, gpt2_model, gpt2_tokenizer):
        """Hooks should capture hidden states during forward pass."""
        with LensHooks(gpt2_model, layer_idx=6) as lens:
            inputs = gpt2_tokenizer("Hello world", return_tensors="pt")
            gpt2_model(**inputs)
            assert "last" in lens.activations
            assert lens.activations["last"].ndim == 3  # [batch, seq, dim]

    def test_hooks_cleaned_up_after_exit(self, gpt2_model, gpt2_tokenizer):
        """Hooks must be removed after context manager exits."""
        with LensHooks(gpt2_model, layer_idx=6) as lens:
            pass
        assert lens.activations == {}

    def test_resolves_gpt2_layers(self, gpt2_model):
        """Should find transformer.h for GPT-2 architecture."""
        hooks = LensHooks(gpt2_model, layer_idx=0)
        assert hooks._target_module is gpt2_model.transformer.h[0]

    def test_activation_shape_matches_model_dim(self, gpt2_model, gpt2_tokenizer):
        """Captured hidden state dimension should match model config."""
        expected_dim = gpt2_model.config.n_embd
        with LensHooks(gpt2_model, layer_idx=6) as lens:
            inputs = gpt2_tokenizer("Test", return_tensors="pt")
            gpt2_model(**inputs)
            assert lens.activations["last"].shape[-1] == expected_dim

    def test_invalid_layer_raises(self, gpt2_model):
        """Out-of-range layer index should raise IndexError."""
        with pytest.raises(IndexError):
            LensHooks(gpt2_model, layer_idx=999)

    def test_hooks_survive_exceptions(self, gpt2_model):
        """Hooks should be cleaned up even if an exception occurs inside the block."""
        hook_count_before = len(list(gpt2_model.transformer.h[6]._forward_hooks))
        try:
            with LensHooks(gpt2_model, layer_idx=6):
                raise ValueError("Simulated error")
        except ValueError:
            pass
        hook_count_after = len(list(gpt2_model.transformer.h[6]._forward_hooks))
        assert hook_count_after == hook_count_before

    def test_multi_layer_hooks(self, gpt2_model, gpt2_tokenizer):
        """Multiple LensHooks on different layers should work independently."""
        with LensHooks(gpt2_model, layer_idx=0) as lens0:
            with LensHooks(gpt2_model, layer_idx=6) as lens6:
                inputs = gpt2_tokenizer("Hello", return_tensors="pt")
                gpt2_model(**inputs)
                assert "last" in lens0.activations
                assert "last" in lens6.activations
                assert not torch.equal(lens0.activations["last"], lens6.activations["last"])
