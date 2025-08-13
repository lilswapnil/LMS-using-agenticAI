import os, json
from dataclasses import dataclass
from typing import Dict, Any, List
import yaml
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from transformers import TrainingArguments
import torch

@dataclass
class Example:
    instruction: str
    tools: List[Dict[str, Any]]
    trajectory: List[Dict[str, Any]]
    final_answer: str

PROMPT = """
You are a helpful agent. You have access to tools and must produce a final answer.
Follow this format exactly:

Instruction: {instruction}
Tools: {tools}
Trajectory:
{trajectory}
Final Answer: {final_answer}
""".strip()

# format a single example into a training string

def format_example(ex: Dict[str, Any]) -> str:
    traj_lines = []
    for step in ex.get("trajectory", []):
        thought = step.get("thought", "")
        action = step.get("action", {})
        observation = step.get("observation", "")
        traj_lines.append(f"Thought: {thought}\nAction: {action}\nObservation: {observation}")
    traj = "\n".join(traj_lines)
    return PROMPT.format(
        instruction=ex.get("instruction", ""),
        tools=ex.get("tools", []),
        trajectory=traj,
        final_answer=ex.get("final_answer", ""),
    )

class JsonlDataset:
    def __init__(self, path: str):
        self.rows = [json.loads(l) for l in open(path, "r", encoding="utf-8").read().splitlines()]
    def to_list(self):
        return [{"text": format_example(r)} for r in self.rows]

if __name__ == "__main__":
    cfg = yaml.safe_load(open("train/config.yaml"))
    model_name = cfg["base_model"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # load in 4/8‑bit if available
    load_kwargs = {}
    if cfg.get("load_4bit", False) or cfg.get("load_8bit", False):
        load_kwargs.update({"device_map": "auto", "trust_remote_code": True})
        if cfg.get("load_4bit", False):
            load_kwargs.update({"load_in_4bit": True})
        elif cfg.get("load_8bit", False):
            load_kwargs.update({"load_in_8bit": True})

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    peft_cfg = LoraConfig(
        r=int(cfg["lora_r"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)

    ds = JsonlDataset(cfg["dataset_path"]).to_list()
    # Convert to HF Dataset so TRL can read columns
    ds = Dataset.from_list(ds)

    # Ensure numeric types (avoid '<=' between float and str' in AdamW)
    per_device_train_batch_size = int(cfg["per_device_train_batch_size"])
    gradient_accumulation_steps = int(cfg["gradient_accumulation_steps"])
    num_train_epochs = float(cfg["num_train_epochs"])
    learning_rate = float(cfg["learning_rate"])
    warmup_ratio = float(cfg.get("warmup_ratio", 0.0))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    logging_steps = int(cfg.get("logging_steps", 10))
    save_steps = int(cfg.get("save_steps", 250))
    max_seq_len = int(cfg.get("max_seq_len", 512))

    args = TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        logging_steps=logging_steps,
        save_steps=save_steps,
        fp16=False,   # disable mixed precision on CPU/MPS
        bf16=False,   # keep off on CPU/MPS
        report_to=["none"],
        max_steps=-1,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=ds,
        dataset_text_field="text",
        max_seq_length=max_seq_len,
        packing=cfg.get("packing", False),
    )

    trainer.train()
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print("✅ SFT complete →", cfg["output_dir"])
