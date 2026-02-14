#!/usr/bin/env python3
"""
EfficientAlign-ONNX: Push Merged Model to Hugging Face Hub

This script merges the trained LoRA weights into the base model
and pushes the merged model to the Hugging Face Hub.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import HfApi
from push_to_hub import push_to_hub

console = Console()

def merge_lora_weights(base_model_path: str, lora_adapter_path: str, output_path: str):
    """Merge LoRA weights into base model."""
    console.print("[bold blue]Merging LoRA weights into base model...[/bold blue]")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype="auto",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    console.print(f"[green]Merged model saved to {output_path}[/green]")
    return output_path, merged_model, tokenizer

def create_model_card(model_name: str, base_model: str, training_info: dict):
    """Create a comprehensive model card."""
    card_content = f"""---
language: en
license: apache-2.0
tags:
- dpo
- alignment
- gemma
- text-generation
- preference-learning
datasets:
- HuggingFaceH4/ultrafeedback_binarized
---

# {model_name}

This model is a fine-tuned version of [{base_model}](https://huggingface.co/{base_model}) using Direct Preference Optimization (DPO) on the ultrafeedback_binarized dataset.

## Training Details

- **Base Model**: {base_model}
- **Fine-tuning Method**: DPO (Direct Preference Optimization)
- **Dataset**: HuggingFaceH4/ultrafeedback_binarized
- **Training Samples**: {training_info.get('train_samples', 'N/A')}
- **Evaluation Samples**: {training_info.get('eval_samples', 'N/A')}
- **Epochs**: {training_info.get('epochs', 'N/A')}
- **Batch Size**: {training_info.get('batch_size', 'N/A')} (per device)
- **Gradient Accumulation**: {training_info.get('grad_accum', 'N/A')}
- **Learning Rate**: {training_info.get('lr', 'N/A')}
- **Beta (DPO)**: {training_info.get('beta', 'N/A')}
- **Max Length**: {training_info.get('max_length', 'N/A')}
- **Optimizer**: {training_info.get('optimizer', 'N/A')}
- **Precision**: bfloat16
- **Quantization**: 4-bit NF4 (during training)
- **LoRA Rank**: {training_info.get('lora_rank', 'N/A')}
- **LoRA Alpha**: {training_info.get('lora_alpha', 'N/A')}
- **Target Modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

## Memory Optimizations

- Gradient Checkpointing
- 8-bit AdamW Optimizer
- Pre-computed Reference Log Probabilities
- LoRA Parameter-Efficient Fine-tuning

## Intended Use

This model is intended for text generation tasks with improved alignment through DPO training. It maintains the capabilities of the base Gemma 3 model while being better aligned with human preferences.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "{model_name}"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Example inference
prompt = "Explain quantum computing in simple terms."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Limitations

- This model inherits limitations from the base Gemma 3 model
- DPO alignment may not cover all edge cases or preferences
- Performance may vary on different hardware configurations

## Citation

If you use this model, please cite the original Gemma model and the DPO paper:

```
@misc{{gemma3,
  title={{Gemma 3}},
  author={{Google DeepMind}},
  year={{2026}}
}}

@article{{rafailov2023direct,
  title={{Direct Preference Optimization: Your Language Model is Secretly a Reward Model}},
  author={{Rafailov, Rafael and Sharma, Archit and Mitchell, Eric and Manning, Christopher D and Finn, Chelsea and Ermon, Stefano}},
  journal={{arXiv preprint arXiv:2305.18290}},
  year={{2023}}
}}
```
"""
    return card_content

def main():
    """Main pipeline to push merged model to Hugging Face Hub."""
    console.print("[bold green]Starting EfficientAlign-ONNX: Push Merged Model to Hub[/bold green]")
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        console.print("[red]HF_TOKEN not found in .env file[/red]")
        return
    api = HfApi(token=hf_token)
    user = api.whoami()
    username = user['name']
    base_model_path = "google/gemma-3-1b-it"
    lora_adapter_path = "./final_model"
    merged_model_path = "./merged_model"
    base_model_name = "gemma-3-1b-it-4bit-lora-dpo-aligned"
    training_info = {
        "train_samples": 1000,
        "eval_samples": 100,
        "epochs": 1,
        "batch_size": 1,
        "grad_accum": 4,
        "lr": "5e-5",
        "beta": 0.1,
        "max_length": 1024,
        "optimizer": "adamw_8bit",
        "lora_rank": 16,
        "lora_alpha": 32,
    }
    merged_path, merged_model, tokenizer = merge_lora_weights(base_model_path, lora_adapter_path, merged_model_path)
    model_card = create_model_card(base_model_name, base_model_path, training_info)
    merged_repo = push_to_hub(merged_model, tokenizer, base_model_name, model_card, hf_token, is_onnx=False)
    if merged_repo:
        console.print(f"[green]Merged model available at: https://huggingface.co/{merged_repo}[/green]")
    else:
        console.print("[red]Failed to push merged model[/red]")

if __name__ == "__main__":
    main()
