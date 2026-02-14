#!/usr/bin/env python3
"""
Test loading and inference for both Hugging Face PyTorch and ONNX models.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import os
from rich.console import Console

console = Console()


from huggingface_hub import HfApi
import getpass

# Automatically get username from Hugging Face API
def get_hf_username():
    try:
        api = HfApi()
        user = api.whoami()
        return user['name']
    except Exception:
        return getpass.getuser()  # fallback to local username

USERNAME = get_hf_username()
HF_REPO = f"{USERNAME}/gemma-3-1b-it-4bit-lora-dpo-aligned"
ONNX_REPO = f"{USERNAME}/gemma-3-1b-it-4bit-lora-dpo-aligned-onnx"
ONNX_MODEL_PATH = "model.onnx"  # Relative to ONNX repo root

PROMPT = "Explain quantum computing in simple terms."


def test_hf_model():
    console.print("[bold blue]Testing Hugging Face PyTorch model...[/bold blue]")
    tokenizer = AutoTokenizer.from_pretrained(HF_REPO)
    model = AutoModelForCausalLM.from_pretrained(HF_REPO)
    inputs = tokenizer(PROMPT, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    console.print("[green]PyTorch Model Output:[/green]")
    print(response)


def test_onnx_model():
    console.print("[bold blue]Testing ONNX model...[/bold blue]")
    # Try to use Optimum if available, else fallback to ONNXRuntime
    # Use robust ONNX loading and inference as in the notebook
    from pathlib import Path
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, GenerationConfig
    ONNX_REPO = "manu02/gemma-3-1b-it-4bit-lora-dpo-aligned-onnx"
    LOCAL_DIR = Path("./hf_onnx_model")
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=ONNX_REPO,
        local_dir=str(LOCAL_DIR),
        local_dir_use_symlinks=False,
    )
    # Prefer tokenizer in ONNX repo, fallback to base model
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(LOCAL_DIR), use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            "manu02/gemma-3-1b-it-4bit-lora-dpo-aligned",
            use_fast=True,
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        from optimum.onnxruntime import ORTModelForCausalLM
        import torch
        from transformers import GenerationConfig
        model = ORTModelForCausalLM.from_pretrained(
            str(LOCAL_DIR),
            file_name="model.onnx",
            provider="CPUExecutionProvider",
            use_cache=False,
        )
        model.config.use_cache = False
        gen_cfg = GenerationConfig.from_model_config(model.config)
        gen_cfg.use_cache = False
        inputs = tokenizer(PROMPT, return_tensors="pt")
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                generation_config=gen_cfg,
                max_new_tokens=120,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        console.print("[green]ONNX Model Output (Optimum):[/green]")
        print(response)
    except ImportError:
        console.print("[yellow]Optimum not installed or failed, falling back to ONNXRuntime API.[/yellow]")
        import numpy as np
        import onnxruntime
        session = onnxruntime.InferenceSession(str(LOCAL_DIR / "model.onnx"))
        inputs = tokenizer(PROMPT, return_tensors="np")
        ort_inputs = {k: v for k, v in inputs.items() if k in [i.name for i in session.get_inputs()]}
        ort_outputs = session.run([session.get_outputs()[0].name], ort_inputs)
        logits = ort_outputs[0]
        token_ids = np.argmax(logits, axis=-1)
        response = tokenizer.decode(token_ids[0], skip_special_tokens=True)
        console.print("[green]ONNX Model Output (onnxruntime):[/green]")
        print(response)


def main():
    test_hf_model()
    print("\n" + "="*60 + "\n")
    test_onnx_model()

if __name__ == "__main__":
    main()
