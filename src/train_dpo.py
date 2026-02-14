#!/usr/bin/env python3
"""
EfficientAlign-ONNX: DPO Training Pipeline for Gemma 3 4B

QLoRA + DPO on ultrafeedback_binarized.
Optimized for ~12GB VRAM.

Key notes:
- Use TRL's DPOConfig (not transformers.TrainingArguments) for DPO-specific args.
- Disable use_cache when using gradient checkpointing.
- Use a single train/test split (don't split twice).
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset
from rich.console import Console

console = Console()


def setup_model_and_tokenizer():
    console.print("[bold blue]Setting up model and tokenizer...[/bold blue]")

    # 4-bit quant config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model_name = "google/gemma-3-1b-it"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,   # <- correct kwarg
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Required for gradient checkpointing + PEFT/QLoRA stability
    model.config.use_cache = False

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # LoRA setup
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"[green]Model ready. Trainable params: {trainable:,}[/green]")

    return model, tokenizer


def load_and_prepare_dataset(tokenizer):
    console.print("[bold blue]Loading dataset...[/bold blue]")

    # This split contains preference pairs (prompt/chosen/rejected)
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_sft")

    def format_dpo_sample(sample):
        user_message = sample["prompt"]
        chosen = sample["chosen"]
        rejected = sample["rejected"]

        # Gemma chat formatting (prompt only). Chosen/rejected are completions.
        prompt = (
            "<start_of_turn>user\n"
            f"{user_message}"
            "<end_of_turn>\n"
            "<start_of_turn>model\n"
        )

        # It's usually beneficial to terminate completions with end_of_turn
        # so the model learns to stop cleanly.
        chosen = f"{chosen}<end_of_turn>\n"
        rejected = f"{rejected}<end_of_turn>\n"

        return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

    ds = ds.map(
        format_dpo_sample,
        remove_columns=ds.column_names,
        num_proc=min(4, os.cpu_count() or 1),
        desc="Formatting DPO samples",
    )

    # Tokenize the dataset to avoid issues with trainer's tokenization
    def tokenize_function(examples):
        prompts = tokenizer(
            examples["prompt"], 
            max_length=512, 
            truncation=True, 
            padding=False
        )
        chosens = tokenizer(
            examples["chosen"], 
            max_length=512, 
            truncation=True, 
            padding=False, 
            add_special_tokens=False
        )
        rejecteds = tokenizer(
            examples["rejected"], 
            max_length=512, 
            truncation=True, 
            padding=False, 
            add_special_tokens=False
        )
        return {
            "prompt_input_ids": prompts["input_ids"],
            "prompt_attention_mask": prompts["attention_mask"],
            "chosen_input_ids": chosens["input_ids"],
            "chosen_attention_mask": chosens["attention_mask"],
            "rejected_input_ids": rejecteds["input_ids"],
            "rejected_attention_mask": rejecteds["attention_mask"],
        }

    ds = ds.map(
        tokenize_function,
        batched=True,
        num_proc=min(4, os.cpu_count() or 1),
        desc="Tokenizing dataset",
    )

    # Split ONCE (you were splitting twice)
    split = ds.train_test_split(test_size=0.05, seed=42)
    train_ds, eval_ds = split["train"], split["test"]

    # For testing: use smaller subsets to speed up
    train_ds = train_ds.select(range(min(1000, len(train_ds))))
    eval_ds = eval_ds.select(range(min(100, len(eval_ds))))

    console.print(f"[green]Dataset ready: {len(train_ds)} train / {len(eval_ds)} eval[/green]")
    return train_ds, eval_ds


def setup_dpo_config():
    # DPOConfig is TRL's config class (extends TrainingArguments with DPO fields).
    return DPOConfig(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,

        # Memory / speed
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_8bit",
        bf16=True,

        # Logging / saving
        logging_steps=10,
        save_steps=500,
        report_to="none",

        # DPO specifics
        beta=0.1,
        max_length=1024,
        max_prompt_length=512,
        truncation_mode="keep_end",

        # IMPORTANT: TRL DPO needs this off unless you use custom collators
        remove_unused_columns=False,
        label_names=[],

        # 12GB friendliness: precompute reference logprobs to reduce peak VRAM.
        # (Will take extra time up front.)
        precompute_ref_log_probs=True,

        dataloader_pin_memory=False,
    )


def main():
    console.print("[bold green]Starting DPO training pipeline[/bold green]")

    # Ensure CUDA is available
    if not torch.cuda.is_available():
        console.print("[red]CUDA not available! This script requires CUDA.[/red]")
        return
    console.print(f"[green]CUDA available: {torch.cuda.get_device_name(0)}[/green]")

    model, tokenizer = setup_model_and_tokenizer()
    # If tokenizer is a processor (for VLMs), extract the text tokenizer
    if hasattr(tokenizer, 'tokenizer'):
        tokenizer = tokenizer.tokenizer
    train_dataset, eval_dataset = load_and_prepare_dataset(tokenizer)
    dpo_args = setup_dpo_config()

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # TRL will create/use a reference model internally as needed
        args=dpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer, # must be PreTrainedTokenizerBase or ProcessorMixin
        peft_config=None,  # LoRA already applied
    )

    console.print("[bold blue]Training...[/bold blue]")
    trainer.train()

    trainer.save_model("./final_model")
    console.print("[green]Done! Saved to ./final_model[/green]")


if __name__ == "__main__":
    main()
