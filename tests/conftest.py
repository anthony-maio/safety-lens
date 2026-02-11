import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer


@pytest.fixture(scope="session")
def gpt2_model():
    """Load GPT-2 once for the entire test session."""
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()
    return model


@pytest.fixture(scope="session")
def gpt2_tokenizer():
    """Load GPT-2 tokenizer once for the entire test session."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
