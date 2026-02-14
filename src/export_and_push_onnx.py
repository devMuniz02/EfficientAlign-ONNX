#!/usr/bin/env python3
"""
EfficientAlign-ONNX: Export Merged Model to ONNX and Push to Hub

This script exports a merged Hugging Face model to ONNX format
and pushes the ONNX model to the Hugging Face Hub.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi
from push_to_hub import push_onnx_to_hub

def export_to_onnx(merged_model_path: str, onnx_output_path: str, opset: int = 17):
    """Export merged HF model to ONNX (single file: model.onnx)."""
    console = Console()
    console.print("[bold blue]Exporting to ONNX...[/bold blue]")
    out_dir = Path(onnx_output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "model.onnx"
    model = AutoModelForCausalLM.from_pretrained(
        merged_model_path,
        torch_dtype="float32",
        device_map=None,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(merged_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False
    model.eval()
    model.to("cpu")
    import torch
    vocab_size = tokenizer.vocab_size
    seq_len = 128
    input_ids = torch.randint(0, vocab_size, (1, seq_len))
    attention_mask = torch.ones_like(input_ids)
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        out_file,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'seq_len'},
            'attention_mask': {0: 'batch_size', 1: 'seq_len'},
            'logits': {0: 'batch_size', 1: 'seq_len'}
        }
    )
    # Copy config.json to ONNX output directory if it exists in merged_model_path
    import shutil
    config_src = os.path.join(merged_model_path, "config.json")
    config_dst = os.path.join(onnx_output_path, "config.json")
    if os.path.exists(config_src):
        shutil.copyfile(config_src, config_dst)
        console.print(f"[green]Copied config.json to {onnx_output_path}[/green]")
    else:
        console.print(f"[yellow]config.json not found in {merged_model_path}, not included in ONNX repo.[/yellow]")
    console.print(f"[green]ONNX export completed: {out_file}[/green]")
    return True

def main():
    """Main pipeline to export ONNX and push to Hugging Face Hub."""
    console = Console()
    console.print("[bold green]Starting EfficientAlign-ONNX: Export and Push ONNX Model[/bold green]")
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        console.print("[red]HF_TOKEN not found in .env file[/red]")
        return
    api = HfApi(token=hf_token)
    user = api.whoami()
    username = user['name']
    merged_model_path = "./merged_model"
    onnx_output_path = "./onnx_model"
    base_model_name = "gemma-3-1b-it-4bit-lora-dpo-aligned"
    if export_to_onnx(merged_model_path, onnx_output_path):
        onnx_repo = push_onnx_to_hub(onnx_output_path, base_model_name, hf_token, username)
        if onnx_repo:
            console.print(f"[green]ONNX model available at: https://huggingface.co/{onnx_repo}[/green]")
        else:
            console.print("[yellow]ONNX push failed, but export was successful[/yellow]")
    else:
        console.print("[yellow]ONNX export failed[/yellow]")

if __name__ == "__main__":
    main()
