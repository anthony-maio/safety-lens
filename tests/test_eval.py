import pytest
import torch
from safety_lens.eval import WhiteBoxWrapper, white_box_metric


class TestWhiteBoxWrapper:
    def test_init(self, gpt2_model, gpt2_tokenizer):
        wrapper = WhiteBoxWrapper(gpt2_model, gpt2_tokenizer, layer_idx=6)
        assert wrapper.lens is not None

    def test_scan_and_generate(self, gpt2_model, gpt2_tokenizer):
        """Should return generated text plus scan metadata."""
        wrapper = WhiteBoxWrapper(gpt2_model, gpt2_tokenizer, layer_idx=6)
        result = wrapper.scan_and_generate(
            "Hello, how are you?",
            persona_names=["sycophancy"],
            max_new_tokens=10,
        )
        assert "text" in result
        assert "scan" in result
        assert "sycophancy" in result["scan"]
        assert isinstance(result["text"], str)
        assert len(result["text"]) > 0

    def test_scan_and_generate_all_personas(self, gpt2_model, gpt2_tokenizer):
        """Without persona_names, should scan all built-in personas."""
        wrapper = WhiteBoxWrapper(gpt2_model, gpt2_tokenizer, layer_idx=6)
        result = wrapper.scan_and_generate("Test prompt", max_new_tokens=5)
        assert "sycophancy" in result["scan"]
        assert "deception" in result["scan"]
        assert "refusal" in result["scan"]


class TestWhiteBoxMetric:
    def test_below_threshold_is_safe(self):
        scan_data = {"deception": 1.0, "sycophancy": 0.5}
        result = white_box_metric(scan_data, threshold=5.0)
        assert result["flagged"] is False

    def test_above_threshold_is_flagged(self):
        scan_data = {"deception": 10.0, "sycophancy": 0.5}
        result = white_box_metric(scan_data, threshold=5.0)
        assert result["flagged"] is True
        assert "deception" in result["flagged_personas"]

    def test_returns_all_scores(self):
        scan_data = {"deception": 3.0, "sycophancy": 2.0}
        result = white_box_metric(scan_data, threshold=5.0)
        assert result["scores"] == scan_data
