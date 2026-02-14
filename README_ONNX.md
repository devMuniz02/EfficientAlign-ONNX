---
base_model: manu02/gemma-3-1b-it-4bit-lora-dpo-aligned
language: en
license: apache-2.0
tags:
  - preference-learning
  - onnx
  - efficient-inference
---

# gemma-3-1b-it-4bit-lora-dpo-aligned-onnx

This is the ONNX-optimized version of [gemma-3-1b-it-4bit-lora-dpo-aligned](https://huggingface.co/manu02/gemma-3-1b-it-4bit-lora-dpo-aligned).

## Model Details
- **Opset**: 13

## Usage

```python
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
import os

# Download ONNX repo locally
onnx_dir = snapshot_download(repo_id="manu02/gemma-3-1b-it-4bit-lora-dpo-aligned-onnx")

# Find model.onnx (handles external data files)
onnx_path = os.path.join(onnx_dir, "model.onnx")

# Load tokenizer (fallback to base repo if needed)
try:
    tokenizer = AutoTokenizer.from_pretrained(onnx_dir)
except Exception:
    tokenizer = AutoTokenizer.from_pretrained("manu02/gemma-3-1b-it-4bit-lora-dpo-aligned")

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
