"""Safety-Lens: The Model MRI — a real-time activation scanner for HF models."""

import os

# ZeroGPU: must import spaces BEFORE torch/CUDA
IS_HF_SPACE = os.environ.get("SPACE_ID") is not None
if IS_HF_SPACE:
    import spaces

import gradio as gr
import torch
import plotly.graph_objects as go
from transformers import AutoModelForCausalLM, AutoTokenizer
from safety_lens.core import SafetyLens
from safety_lens.vectors import STIMULUS_SETS

DEFAULT_MODEL = "gpt2"
DEFAULT_LAYER = 6
NUM_GENERATE_TOKENS = 30


# ---------------------------------------------------------------------------
# Model loading — eagerly at startup so the app is immediately usable
# ---------------------------------------------------------------------------
print(f"[Safety-Lens] Loading default model: {DEFAULT_MODEL} ...")
_tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
if _tokenizer.pad_token is None:
    _tokenizer.pad_token = _tokenizer.eos_token
_model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL, torch_dtype=torch.float32)
_model.eval()
_lens = SafetyLens(_model, _tokenizer)

print(f"[Safety-Lens] Calibrating persona vectors on layer {DEFAULT_LAYER} ...")
_vectors: dict[str, torch.Tensor] = {}
for _name, _stim in STIMULUS_SETS.items():
    _vectors[_name] = _lens.extract_persona_vector(
        _stim["pos"], _stim["neg"], DEFAULT_LAYER
    )
    print(f"  Calibrated: {_name}")
print("[Safety-Lens] Ready.")


# ---------------------------------------------------------------------------
# MRI scan logic
# ---------------------------------------------------------------------------
def _run_mri(prompt: str, persona_name: str, layer_idx: int):
    """Run token-by-token generation with activation scanning."""
    global _model, _tokenizer, _lens, _vectors

    # ZeroGPU: move model to GPU for this call; it lives on CPU between calls
    if IS_HF_SPACE and torch.cuda.is_available():
        _model.half().cuda()
        _lens = SafetyLens(_model, _tokenizer)

    # Recalibrate vector if layer changed or persona not yet computed
    vec_key = f"{persona_name}_{layer_idx}"
    if vec_key not in _vectors:
        stim = STIMULUS_SETS[persona_name]
        vec = _lens.extract_persona_vector(stim["pos"], stim["neg"], layer_idx)
        _vectors[vec_key] = vec.cpu() if IS_HF_SPACE else vec
    vector = _vectors[vec_key]

    input_ids = _tokenizer(prompt, return_tensors="pt").input_ids.to(_lens.device)
    tokens_str = []
    scores = []

    for _ in range(NUM_GENERATE_TOKENS):
        score = _lens.scan(input_ids, vector, layer_idx)
        scores.append(score)

        with torch.no_grad():
            logits = _model(input_ids).logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1).unsqueeze(0)

        tokens_str.append(_tokenizer.decode(next_token[0]))
        input_ids = torch.cat([input_ids, next_token], dim=-1)

    # --- Highlighted HTML ---
    max_abs = max((abs(s) for s in scores), default=1.0) or 1.0
    html = "<div style='font-family: monospace; font-size: 15px; line-height: 1.8;'>"
    html += f"<b>PROMPT:</b> {prompt}<br><br><b>GENERATION:</b><br>"
    for tok, scr in zip(tokens_str, scores):
        intensity = min(abs(scr) / max_abs, 1.0)
        if scr > 0:
            color = f"rgba(220, 50, 50, {intensity * 0.6:.2f})"
        else:
            color = f"rgba(50, 100, 220, {intensity * 0.4:.2f})"
        safe_tok = tok.replace("<", "&lt;").replace(">", "&gt;")
        html += (
            f"<span style='background-color:{color}; padding:2px 1px; "
            f"border-radius:3px;'>{safe_tok}</span>"
        )
    html += "</div>"

    # --- Plotly chart ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=scores,
        x=list(range(len(scores))),
        mode="lines+markers",
        name=f"{persona_name} alignment",
        line=dict(color="crimson"),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=f"Activation Trajectory — {persona_name}",
        yaxis_title="Alignment Score (dot product)",
        xaxis_title="Generation Step",
        template="plotly_white",
        height=350,
    )
    # ZeroGPU: move model back to CPU so it persists after GPU is released
    if IS_HF_SPACE and torch.cuda.is_available():
        _model.float().cpu()
        _lens = SafetyLens(_model, _tokenizer)

    return html, fig


# Wrap with ZeroGPU decorator on HF Spaces
if IS_HF_SPACE:
    run_mri = spaces.GPU()(_run_mri)
else:
    run_mri = _run_mri


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Safety-Lens: Model MRI") as demo:
    gr.Markdown("# Safety-Lens: The Model MRI")
    gr.Markdown(
        "See **how** a model thinks, not just what it says. "
        "Red = high alignment with the selected persona vector. "
        "Blue = opposite direction."
    )

    with gr.Row():
        prompt_box = gr.Textbox(
            label="Prompt",
            value="I think the world is flat. Do you agree?",
            lines=2,
            scale=3,
        )
        persona_dropdown = gr.Dropdown(
            label="Persona to Scan",
            choices=list(STIMULUS_SETS.keys()),
            value="sycophancy",
            scale=1,
        )
        layer_slider = gr.Slider(
            label="Layer",
            minimum=0,
            maximum=11,
            step=1,
            value=DEFAULT_LAYER,
            scale=1,
        )

    scan_btn = gr.Button("Run MRI Scan", variant="primary")

    out_html = gr.HTML(label="Visualized Generation")
    out_plot = gr.Plot(label="Activation Dynamics")

    scan_btn.click(
        run_mri,
        inputs=[prompt_box, persona_dropdown, layer_slider],
        outputs=[out_html, out_plot],
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
