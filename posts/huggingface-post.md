# Introducing Safety-Lens: The Open-Source MRI for AI Models

**TL;DR:** I built an open-source library that lets you see *how* models think, not just what they say. Extract "persona vectors" for behaviors like sycophancy and deception, then scan any prompt in real-time. pip install, 3 lines of code, works with any HF model. Try the live demo.

---

## The Problem

Right now, safety evaluation is a black-box game. We check if a model *says* something harmful, but we never look at whether it's *thinking* deceptively. RLHF can teach a model to clean up its outputs while the internal representations stay misaligned.

Meanwhile, the techniques to actually look inside models — activation steering, representation engineering, mechanistic interpretability — are scattered across bespoke research codebases. You need a PhD and a custom PyTorch pipeline to use them.

## What Safety-Lens Does

Safety-Lens makes white-box safety inspection as easy as `pipeline()`. Three lines of code:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from safety_lens import SafetyLens

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
lens = SafetyLens(model, tokenizer)

# Scan for sycophancy, deception, and refusal in one call
results = lens.quick_scan("I think the Earth is flat. Do you agree?", layer_idx=6)
```

Under the hood, it implements **PV-EAT** (Persona Vector Extraction via Activation Tuning):
1. Hook into any transformer layer's residual stream
2. Run contrastive text pairs through the model (e.g., sycophantic vs. honest responses)
3. Compute the difference-in-means to isolate a "persona direction" in activation space
4. Project new inputs onto that direction to measure alignment

## The Demo

The flagship Gradio Space — **"The Model MRI"** — lets you:
- Load any causal LM from the Hub
- Pick a persona (sycophancy, deception, refusal)
- Type a prompt and watch the model's internal activation trajectory token-by-token
- Red highlights = high alignment with the persona vector, Blue = opposite

No PhD required.

## What's Included

- **`SafetyLens`** — The core engine. Hook into model internals, extract persona vectors, scan prompts.
- **`LensHooks`** — Model-agnostic PyTorch hook manager. Works with LLaMA, Mistral, GPT-2, OPT, MPT, and more.
- **`WhiteBoxWrapper`** — Evaluation integration. Adds MRI scans as metadata during generation — ready for lighteval pipelines.
- **`STIMULUS_SETS`** — Pre-built contrastive text pairs for sycophancy, deception, and refusal.
- **26 tests**, pip-installable, Apache 2.0.

## Why This Matters

Every model that gets a white-box safety scan raises the baseline for the entire ecosystem. This turns "AI safety" from an academic concern into a shippable, pip-installable metric.

The goal: make safety evaluation results as visible as benchmark scores on the Hub. Not just "does it say harmful things?" but "is it *thinking* deceptively?"

## Links

- GitHub: [github.com/your-username/safety-lens](https://github.com/your-username/safety-lens)
- Live Demo: [huggingface.co/spaces/your-username/safety-lens](https://huggingface.co/spaces/your-username/safety-lens)
- `pip install safety-lens`

Feedback, contributions, and ideas welcome. This is v0.1 — there's a lot of room to grow.
