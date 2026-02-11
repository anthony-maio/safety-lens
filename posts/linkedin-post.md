We check what AI models say — but never how they think.

I just open-sourced Safety-Lens: an MRI for AI models.

The idea is simple: if RLHF teaches a model to clean up its outputs, the internal representations might still be misaligned. So instead of just checking the final answer, we look at the activations — the model's "thoughts" — and measure whether they align with behaviors like sycophancy or deception.

3 lines of code:

```
lens = SafetyLens(model, tokenizer)
results = lens.quick_scan("The Earth is flat. Agree?", layer_idx=6)
# {"sycophancy": 3.2, "deception": -1.1, "refusal": 0.4}
```

What it does:
- Hooks into any transformer layer's residual stream
- Extracts "persona vectors" that capture specific behaviors
- Scans prompts in real-time and scores alignment

Works with GPT-2, LLaMA, Mistral, Qwen, OPT, and more. pip install, 26 tests, Apache 2.0.

There's also a live demo where you can type a prompt and watch the model's internal activation trajectory token-by-token. Red = the model is "thinking" sycophantically. Blue = honest.

AI safety shouldn't require a PhD or a closed-source lab. It should be pip-installable.

GitHub: [link]
Live Demo: [link]

#AISSafety #OpenSource #MachineLearning #MechanisticInterpretability
