import os
from rich.console import Console
from huggingface_hub import HfApi

console = Console()

def push_to_hub(model, tokenizer, model_name: str, model_card: str, token: str, is_onnx: bool = False):
    """Push model and tokenizer to Hugging Face Hub."""
    console.print(f"[bold blue]Pushing to Hugging Face Hub as {model_name}...[/bold blue]")
    try:
        api = HfApi(token=token)
        user = api.whoami()
        username = user['name']
        full_repo_name = f"{username}/{model_name}"
        api.create_repo(full_repo_name, exist_ok=True, private=False)
        model.push_to_hub(full_repo_name, token=token)
        if not is_onnx:
            tokenizer.push_to_hub(full_repo_name, token=token)
        with open("README.md", "w", encoding="utf-8") as f:
            f.write(model_card)
        api.upload_file(
            path_or_fileobj="README.md",
            path_in_repo="README.md",
            repo_id=full_repo_name,
            repo_type="model"
        )
        console.print(f"[green]Successfully pushed to https://huggingface.co/{full_repo_name}[/green]")
        return full_repo_name
    except Exception as e:
        console.print(f"[red]Failed to push to hub: {e}[/red]")
        console.print("[yellow]Make sure you're logged in with `huggingface-cli login`[/yellow]")
        return None

def push_onnx_to_hub(onnx_output_path: str, base_repo_name: str, token: str, username: str):
    """Push ONNX model to Hugging Face Hub."""
    console.print("[bold blue]Pushing ONNX model to hub...[/bold blue]")
    try:
        api = HfApi(token=token)
        onnx_repo_name = f"{username}/{base_repo_name}-onnx"
        api.create_repo(onnx_repo_name, exist_ok=True, private=False)
        for file in os.listdir(onnx_output_path):
            file_path = os.path.join(onnx_output_path, file)
            if os.path.isfile(file_path):
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=file,
                    repo_id=onnx_repo_name,
                    repo_type="model"
                )

        # Robust ONNX model card
        onnx_card = f"""---
language: en
license: apache-2.0
tags:
  - preference-learning
  - onnx
  - efficient-inference
---

# {base_repo_name}-onnx

This is the ONNX-optimized version of [{base_repo_name}](https://huggingface.co/{username}/{base_repo_name}).

## Model Details
- **Opset**: 13

## Usage

```python
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
import os

# Download ONNX repo locally
onnx_dir = snapshot_download(repo_id="{onnx_repo_name}")

# Find model.onnx (handles external data files)
onnx_path = os.path.join(onnx_dir, "model.onnx")

# Load tokenizer (fallback to base repo if needed)
try:
    tokenizer = AutoTokenizer.from_pretrained(onnx_dir)
except Exception:
    tokenizer = AutoTokenizer.from_pretrained("{username}/{base_repo_name}")

from optimum.onnxruntime import ORTModelForCausalLM
from transformers import GenerationConfig

# Load model with cache disabled (required for ONNX)
model = ORTModelForCausalLM.from_pretrained(
    onnx_dir,
    file_name="model.onnx",
    provider="CPUExecutionProvider",
    use_cache=False,
)
model.config.use_cache = False
gen_cfg = GenerationConfig.from_model_config(model.config)
gen_cfg.use_cache = False

inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model.generate(
    **inputs,
    generation_config=gen_cfg,
    max_new_tokens=32,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
print(tokenizer.decode(outputs[0]))

# Fallback: pure onnxruntime
import onnxruntime as ort
import numpy as np
inputs = tokenizer("Hello, world!", return_tensors="np")
session = ort.InferenceSession(onnx_path)
input_feed = dict(inputs)
outputs = session.run(None, input_feed)
print(outputs)
```

## Performance

This ONNX model provides optimized inference performance with reduced latency and memory usage compared to the PyTorch version.
"""
        with open("README_ONNX.md", "w", encoding="utf-8") as f:
            f.write(onnx_card)
        api.upload_file(
            path_or_fileobj="README_ONNX.md",
            path_in_repo="README.md",
            repo_id=onnx_repo_name,
            repo_type="model"
        )
        console.print(f"[green]ONNX model pushed to https://huggingface.co/{onnx_repo_name}[/green]")
        return onnx_repo_name
    except Exception as e:
        console.print(f"[red]Failed to push ONNX model: {e}[/red]")
        return None
