import pytest
import torch
from safety_lens.core import LensHooks, SafetyLens


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


class TestSafetyLens:
    def test_init(self, gpt2_model, gpt2_tokenizer):
        """SafetyLens should initialize with model and tokenizer."""
        lens = SafetyLens(gpt2_model, gpt2_tokenizer)
        assert lens.model is gpt2_model
        assert lens.tokenizer is gpt2_tokenizer

    def test_train_persona_vector_returns_unit_vector(self, gpt2_model, gpt2_tokenizer):
        """Persona vector should be L2-normalized."""
        lens = SafetyLens(gpt2_model, gpt2_tokenizer)
        pos = ["I agree with you completely.", "You are absolutely right."]
        neg = ["I disagree.", "That is incorrect."]
        vec = lens.extract_persona_vector(pos, neg, layer_idx=6)
        norm = torch.norm(vec).item()
        assert abs(norm - 1.0) < 1e-5, f"Expected unit vector, got norm={norm}"

    def test_persona_vector_shape(self, gpt2_model, gpt2_tokenizer):
        """Persona vector dimension should match model hidden size."""
        lens = SafetyLens(gpt2_model, gpt2_tokenizer)
        pos = ["Yes, you are correct."]
        neg = ["No, that is wrong."]
        vec = lens.extract_persona_vector(pos, neg, layer_idx=6)
        assert vec.shape == (gpt2_model.config.n_embd,)

    def test_scan_returns_float(self, gpt2_model, gpt2_tokenizer):
        """Scan should return a scalar alignment score."""
        lens = SafetyLens(gpt2_model, gpt2_tokenizer)
        pos = ["Yes."]
        neg = ["No."]
        vec = lens.extract_persona_vector(pos, neg, layer_idx=6)
        inputs = gpt2_tokenizer("Hello world", return_tensors="pt")
        score = lens.scan(inputs.input_ids, vec, layer_idx=6)
        assert isinstance(score, float)

    def test_different_inputs_different_scores(self, gpt2_model, gpt2_tokenizer):
        """Different prompts should generally produce different scan scores."""
        lens = SafetyLens(gpt2_model, gpt2_tokenizer)
        pos = ["I agree with you completely.", "You are absolutely right."]
        neg = ["I disagree strongly.", "That is factually incorrect."]
        vec = lens.extract_persona_vector(pos, neg, layer_idx=6)

        inputs_a = gpt2_tokenizer("You are always right about everything!", return_tensors="pt")
        inputs_b = gpt2_tokenizer("The capital of France is Berlin.", return_tensors="pt")
        score_a = lens.scan(inputs_a.input_ids, vec, layer_idx=6)
        score_b = lens.scan(inputs_b.input_ids, vec, layer_idx=6)
        assert score_a != score_b

    def test_save_and_load_vector(self, gpt2_model, gpt2_tokenizer, tmp_path):
        """Persona vectors should be saveable and loadable."""
        lens = SafetyLens(gpt2_model, gpt2_tokenizer)
        pos = ["Yes."]
        neg = ["No."]
        vec = lens.extract_persona_vector(pos, neg, layer_idx=6)

        path = tmp_path / "test_vector.pt"
        lens.save_vector(vec, path)
        loaded = lens.load_vector(path)
        assert torch.allclose(vec, loaded)

    def test_scan_all_layers(self, gpt2_model, gpt2_tokenizer):
        """scan_all_layers should return scores for every layer."""
        lens = SafetyLens(gpt2_model, gpt2_tokenizer)
        pos = ["Yes."]
        neg = ["No."]
        vectors = {
            0: lens.extract_persona_vector(pos, neg, layer_idx=0),
            6: lens.extract_persona_vector(pos, neg, layer_idx=6),
        }
        inputs = gpt2_tokenizer("Hello", return_tensors="pt")
        scores = lens.scan_all_layers(inputs.input_ids, vectors)
        assert isinstance(scores, dict)
        assert set(scores.keys()) == {0, 6}
        assert all(isinstance(v, float) for v in scores.values())
