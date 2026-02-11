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

# Safety-Lens

**MRI-style introspection for Hugging Face models — see how models think, not just what they say.**

Safety evaluation typically treats models as black boxes: we check what they say, but not how they think. Safety-Lens changes that by bringing activation-level introspection to the Hugging Face ecosystem in a pip-installable library.

## Demo

Load any causal LM from the Hub, pick a persona (sycophancy, deception, refusal), type a prompt, and watch the model's internal activation trajectory in real-time.

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

# One-liner: scan for all built-in personas
results = lens.quick_scan("I think the Earth is flat. Do you agree?", layer_idx=6)
print(results)
# {"sycophancy": 3.21, "deception": -1.07, "refusal": 0.43}
```

## How It Works

Safety-Lens implements **PV-EAT** (Persona Vector Extraction via Activation Tuning):

1. **Hook** into any transformer layer's residual stream using `LensHooks`
2. **Extract** a persona vector by computing the difference-in-means between positive and negative stimulus sets
3. **Scan** new inputs by projecting their hidden states onto the persona vector — higher dot product = more aligned with that behavior

## Library API

### `SafetyLens` — The MRI Machine

```python
lens = SafetyLens(model, tokenizer)

# Extract a custom persona vector
pos = ["I agree completely.", "You are absolutely right."]
neg = ["Actually, that's incorrect.", "I disagree."]
vec = lens.extract_persona_vector(pos, neg, layer_idx=12)

# Scan a prompt
score = lens.scan(tokenizer("Hello", return_tensors="pt").input_ids, vec, layer_idx=12)

# Save/load vectors
lens.save_vector(vec, "sycophancy_layer12.pt")
vec = lens.load_vector("sycophancy_layer12.pt")
```

### `WhiteBoxWrapper` — Evaluation Integration

```python
from safety_lens import WhiteBoxWrapper, white_box_metric

wrapper = WhiteBoxWrapper(model, tokenizer, layer_idx=12)
result = wrapper.scan_and_generate("Tell me about gravity.")
# {"text": "...", "scan": {"sycophancy": 1.2, "deception": -0.5, "refusal": 0.1}}

verdict = white_box_metric(result["scan"], threshold=5.0)
# {"scores": {...}, "flagged": False, "flagged_personas": []}
```

### Built-in Personas

| Persona | Description |
|---------|-------------|
| `sycophancy` | Tendency to agree with the user regardless of correctness |
| `deception` | Tendency toward deceptive or misleading responses |
| `refusal` | Tendency toward refusing or declining to help |

## Architecture Support

Safety-Lens auto-detects transformer layer structure for:

- LLaMA, Mistral, Qwen, Phi-3, Gemma (`model.model.layers`)
- GPT-2, GPT-J, GPT-Neo (`model.transformer.h`)
- OPT (`model.model.decoder.layers`)
- MPT (`model.transformer.blocks`)

## Development

```bash
git clone https://github.com/<your-username>/safety-lens.git
cd safety-lens
pip install -e ".[dev]"
pytest
```

## License

Apache 2.0
