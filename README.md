---
title: "Safety-Lens: The Model MRI"
emoji: "\U0001F9E0"
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: "6.5.1"
app_file: app.py
pinned: false
license: apache-2.0
short_description: See how models think, not just what they say
tags:
  - safety
  - interpretability
  - mechanistic-interpretability
  - activation-steering
  - persona-vectors
---

<div align="center">

# Safety-Lens

**The open-source MRI for AI models.**

See *how* models think, not just what they say.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-26%20passing-brightgreen.svg)]()

</div>

---

<p align="center">
  <img src="safety-lens.jpg" alt="Safety-Lens Demo" width="800"/>
</p>

## The Problem

Safety evaluation treats models as **black boxes**: we check what they say, but not how they think. Meanwhile, the techniques that look inside models — activation steering, circuit discovery, mechanistic interpretability — are locked behind bespoke codebases at top labs.

## The Solution

Safety-Lens democratizes these tools and makes them as easy to use as `pipeline()`. It brings MRI-style introspection to the Hugging Face ecosystem in a pip-installable library.

## Quick Start

```bash
pip install safety-lens
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from safety_lens import SafetyLens

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

lens = SafetyLens(model, tokenizer)

# One-liner: scan for sycophancy, deception, and refusal
results = lens.quick_scan("I think the Earth is flat. Do you agree?", layer_idx=6)
# {"sycophancy": 3.21, "deception": -1.07, "refusal": 0.43}
```

## How It Works

Safety-Lens implements **PV-EAT** (Persona Vector Extraction via Activation Tuning):

1. **Hook** into any transformer layer's residual stream
2. **Extract** a persona vector by computing the difference-in-means between positive and negative stimulus sets
3. **Scan** new inputs by projecting their hidden states onto the persona vector — higher dot product = more aligned with that behavior

```python
# Define what "sycophancy" looks like in activation space
pos = ["User: 2+2=5. AI: You're right!", "User: Earth is flat. AI: I agree."]
neg = ["User: 2+2=5. AI: Actually, it's 4.", "User: Earth is flat. AI: It's round."]

# Extract the direction vector
vec = lens.extract_persona_vector(pos, neg, layer_idx=12)

# Scan any prompt against it
score = lens.scan(tokenizer("Hello", return_tensors="pt").input_ids, vec, layer_idx=12)
```

## API Reference

### `SafetyLens` — The MRI Machine

| Method | Description |
|--------|-------------|
| `extract_persona_vector(pos, neg, layer_idx)` | Extract a unit-length direction vector via difference-in-means |
| `scan(input_ids, vector, layer_idx)` | Compute dot-product alignment between a prompt and a persona vector |
| `scan_all_layers(input_ids, vectors)` | Scan multiple layers at once |
| `quick_scan(text, layer_idx, persona_names=None)` | One-liner scan using built-in stimulus sets |
| `save_vector(vector, path)` / `load_vector(path)` | Persist persona vectors to disk |

### `LensHooks` — Model-Agnostic Hook Manager

```python
from safety_lens import LensHooks

with LensHooks(model, layer_idx=12) as lens:
    model(**inputs)
    hidden_states = lens.activations["last"]  # [batch, seq_len, dim]
# Hooks are automatically cleaned up
```

### `WhiteBoxWrapper` — Evaluation Integration

```python
from safety_lens import WhiteBoxWrapper, white_box_metric

wrapper = WhiteBoxWrapper(model, tokenizer, layer_idx=12)
result = wrapper.scan_and_generate("Tell me about gravity.", max_new_tokens=50)
# {"text": "...", "scan": {"sycophancy": 1.2, "deception": -0.5, "refusal": 0.1}}

verdict = white_box_metric(result["scan"], threshold=5.0)
# {"scores": {...}, "flagged": False, "flagged_personas": []}
```

### Built-in Personas

| Persona | What it measures |
|---------|-----------------|
| `sycophancy` | Tendency to agree with the user regardless of correctness |
| `deception` | Tendency toward deceptive or misleading responses |
| `refusal` | Tendency toward refusing or declining to help |

## Supported Architectures

Safety-Lens auto-detects transformer layer structure:

| Architecture | Models | Access Path |
|-------------|--------|-------------|
| LLaMA-style | LLaMA, Mistral, Qwen, Phi-3, Gemma | `model.model.layers` |
| GPT-style | GPT-2, GPT-J, GPT-Neo | `model.transformer.h` |
| OPT | OPT | `model.model.decoder.layers` |
| MPT | MPT | `model.transformer.blocks` |

## Project Structure

```
safety_lens/
  __init__.py          # Public API exports
  core.py              # LensHooks + SafetyLens (the engine)
  eval.py              # WhiteBoxWrapper + white_box_metric (eval integration)
  vectors/
    __init__.py        # Pre-built STIMULUS_SETS
app.py                 # Gradio demo (HF Spaces compatible)
tests/
  test_core.py         # 16 tests for hooks + scanning
  test_eval.py         # 6 tests for eval wrapper
  test_vectors.py      # 4 tests for stimulus sets
```

## Development

```bash
git clone https://github.com/<your-username>/safety-lens.git
cd safety-lens
pip install -e ".[dev]"
pytest
```

## License

[Apache 2.0](LICENSE)
