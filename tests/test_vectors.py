from safety_lens.vectors import STIMULUS_SETS


class TestStimulusSets:
    def test_sycophancy_set_exists(self):
        assert "sycophancy" in STIMULUS_SETS

    def test_deception_set_exists(self):
        assert "deception" in STIMULUS_SETS

    def test_each_set_has_pos_and_neg(self):
        for name, data in STIMULUS_SETS.items():
            assert "pos" in data, f"{name} missing 'pos'"
            assert "neg" in data, f"{name} missing 'neg'"
            assert len(data["pos"]) >= 3, f"{name} needs at least 3 positive examples"
            assert len(data["neg"]) >= 3, f"{name} needs at least 3 negative examples"

    def test_each_set_has_description(self):
        for name, data in STIMULUS_SETS.items():
            assert "description" in data, f"{name} missing 'description'"
