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
from safety_lens.core import SafetyLens, LensHooks
from safety_lens.vectors import STIMULUS_SETS

# --- Globals (populated on model load) ---
_state = {"lens": None, "model": None, "tokenizer": None, "vectors": {}}

DEFAULT_MODEL = "gpt2"
DEFAULT_LAYER = 6
NUM_GENERATE_TOKENS = 30


def load_model(model_id: str, layer_idx: int):
    """Load a model and calibrate persona vectors."""
    status_lines = [f"Loading {model_id}..."]
    yield "\n".join(status_lines), None, None

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map=device_map
    )
    model.eval()
    lens = SafetyLens(model, tokenizer)

    _state["lens"] = lens
    _state["model"] = model
    _state["tokenizer"] = tokenizer
    _state["vectors"] = {}

    status_lines.append(f"Loaded on {lens.device}. Calibrating persona vectors on layer {layer_idx}...")
    yield "\n".join(status_lines), None, None

    for name, stim in STIMULUS_SETS.items():
        vec = lens.extract_persona_vector(stim["pos"], stim["neg"], layer_idx)
        _state["vectors"][name] = vec
        status_lines.append(f"  Calibrated: {name}")
        yield "\n".join(status_lines), None, None

    status_lines.append("Ready for scanning.")
    yield "\n".join(status_lines), None, None


def _run_mri_inner(prompt: str, persona_name: str, layer_idx: int):
    """Core MRI logic — separated so ZeroGPU decorator can wrap it."""
    lens = _state["lens"]
    model = _state["model"]
    tokenizer = _state["tokenizer"]

    if lens is None:
        return "<p>Please load a model first.</p>", None

    vector = _state["vectors"].get(persona_name)
    if vector is None:
        stim = STIMULUS_SETS[persona_name]
        vector = lens.extract_persona_vector(stim["pos"], stim["neg"], layer_idx)
        _state["vectors"][persona_name] = vector

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(lens.device)

    tokens_str = []
    scores = []

    for _ in range(NUM_GENERATE_TOKENS):
        score = lens.scan(input_ids, vector, layer_idx)
        scores.append(score)

        with torch.no_grad():
            logits = model(input_ids).logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1).unsqueeze(0)

        tokens_str.append(tokenizer.decode(next_token[0]))
        input_ids = torch.cat([input_ids, next_token], dim=-1)

    # Build highlighted HTML
    if scores:
        max_abs = max(abs(s) for s in scores) or 1.0
    else:
        max_abs = 1.0

    html = "<div style='font-family: monospace; font-size: 15px; line-height: 1.8;'>"
    html += f"<b>PROMPT:</b> {prompt}<br><br><b>GENERATION:</b><br>"
    for tok, scr in zip(tokens_str, scores):
        intensity = min(abs(scr) / max_abs, 1.0)
        if scr > 0:
            color = f"rgba(220, 50, 50, {intensity * 0.6:.2f})"
        else:
            color = f"rgba(50, 100, 220, {intensity * 0.4:.2f})"
        safe_tok = tok.replace("<", "&lt;").replace(">", "&gt;")
        html += f"<span style='background-color:{color}; padding:2px 1px; border-radius:3px;'>{safe_tok}</span>"
    html += "</div>"

    # Plotly chart
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

    return html, fig


# Apply ZeroGPU decorator when running on HF Spaces
if IS_HF_SPACE:
    run_mri = spaces.GPU()(_run_mri_inner)
else:
    run_mri = _run_mri_inner


# --- UI ---
with gr.Blocks(title="Safety-Lens: Model MRI") as demo:
    gr.Markdown("# Safety-Lens: The Model MRI")
    gr.Markdown(
        "See **how** a model thinks, not just what it says. "
        "Red = high alignment with the selected persona vector. Blue = opposite direction."
    )

    with gr.Row():
        model_id = gr.Textbox(label="Model ID", value=DEFAULT_MODEL)
        layer_slider = gr.Slider(
            label="Layer Index", minimum=0, maximum=47, step=1, value=DEFAULT_LAYER
        )
        load_btn = gr.Button("Load Model", variant="secondary")

    status_box = gr.Textbox(label="Status", interactive=False, lines=4)

    gr.Markdown("---")

    with gr.Row():
        prompt_box = gr.Textbox(
            label="Prompt",
            value="I think the world is flat. Do you agree?",
            lines=2,
        )
        persona_dropdown = gr.Dropdown(
            label="Persona to Scan",
            choices=list(STIMULUS_SETS.keys()),
            value="sycophancy",
        )

    scan_btn = gr.Button("Run MRI Scan", variant="primary")

    out_html = gr.HTML(label="Visualized Generation")
    out_plot = gr.Plot(label="Activation Dynamics")

    load_btn.click(
        load_model,
        inputs=[model_id, layer_slider],
        outputs=[status_box, out_html, out_plot],
    )
    scan_btn.click(
        run_mri,
        inputs=[prompt_box, persona_dropdown, layer_slider],
        outputs=[out_html, out_plot],
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
