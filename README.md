[![ArXiv](https://img.shields.io/badge/ArXiv-2512.16841-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2512.16841)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-devmuniz-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/devmuniz)
[![GitHub Profile](https://img.shields.io/badge/GitHub-devMuniz02-181717?logo=github&logoColor=white)](https://github.com/devMuniz02)
[![Portfolio](https://img.shields.io/badge/Portfolio-devmuniz02.github.io-0F172A?logo=googlechrome&logoColor=white)](https://devmuniz02.github.io/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-manu02-FFD21E?logoColor=black)](https://huggingface.co/manu02)

# gemma-3-1b-it-4bit-lora-dpo-aligned

--- base_model: google/gemma-3-1b-it language: en license: apache-2.0 tags: - dpo - alignment - gemma - text-generation - preference-learning datasets: - HuggingFaceH4/ultrafeedback_binarized ---

This model is a fine-tuned version of [google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it) using Direct Preference Optimization (DPO) on the ultrafeedback_binarized dataset.

## Overview

A complete end-to-end pipeline to Fine-tune (QLoRA), Align (DPO/RLHF), and Deploy (ONNX) Large Language Models on consumer hardware. Designed for low resource environments.

## Repository Structure

| Path | Description |
| --- | --- |
| `assets/` | Images, figures, or other supporting media used by the project. |
| `notebooks/` | Exploratory notebooks and experiment walkthroughs. |
| `src/` | Primary source code for the application or library. |
| `.gitignore` | Top-level file included in the repository. |
| `LICENSE` | Repository license information. |
| `README.md` | Primary project documentation. |
| `README_ONNX.md` | Top-level file included in the repository. |
| `requirements.txt` | Python dependency specification for local setup. |

## Getting Started

1. Clone the repository.

   ```bash
   git clone https://github.com/devMuniz02/EfficientAlign-ONNX.git
   cd EfficientAlign-ONNX
   ```

2. Prepare the local environment.

Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Run or inspect the project entry point.

Use the project-specific scripts or notebooks in the repository root to run the workflow.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gemma-3-1b-it-4bit-lora-dpo-aligned"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Example inference
prompt = "Explain quantum computing in simple terms."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Training Details

- **Base Model**: google/gemma-3-1b-it
- **Fine-tuning Method**: DPO (Direct Preference Optimization)
- **Dataset**: HuggingFaceH4/ultrafeedback_binarized
- **Training Samples**: 1000
- **Evaluation Samples**: 100
- **Epochs**: 1
- **Batch Size**: 1 (per device)
- **Gradient Accumulation**: 4
- **Learning Rate**: 5e-5
- **Beta (DPO)**: 0.1
- **Max Length**: 1024
- **Optimizer**: adamw_8bit
- **Precision**: bfloat16
- **Quantization**: 4-bit NF4 (during training)
- **LoRA Rank**: 16
- **LoRA Alpha**: 32
- **Target Modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

## Memory Optimizations

- Gradient Checkpointing
- 8-bit AdamW Optimizer
- Pre-computed Reference Log Probabilities
- LoRA Parameter-Efficient Fine-tuning

## Intended Use

This model is intended for text generation tasks with improved alignment through DPO training. It maintains the capabilities of the base Gemma 3 model while being better aligned with human preferences.

## Limitations

- This model inherits limitations from the base Gemma 3 model
- DPO alignment may not cover all edge cases or preferences
- Performance may vary on different hardware configurations

## Citation

If you use this model, please cite the original Gemma model and the DPO paper:

```
@misc{gemma3,
  title={Gemma 3},
  author={Google DeepMind},
  year={2026}
}

@article{rafailov2023direct,
  title={Direct Preference Optimization: Your Language Model is Secretly a Reward Model},
  author={Rafailov, Rafael and Sharma, Archit and Mitchell, Eric and Manning, Christopher D and Finn, Chelsea and Ermon, Stefano},
  journal={arXiv preprint arXiv:2305.18290},
  year={2023}
}
```
