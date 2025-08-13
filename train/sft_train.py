import os, json, yaml, torch
from dataclasses import dataclass
from typing import List, Dict, Any
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

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

def render_record(rec: Dict[str, Any]) -> str:
    instr = rec.get("instruction") or rec.get("query") or ""
    traj = rec.get("trajectory") or []
    ans = rec.get("final_answer") or rec.get("expected") or ""
    lines = [f"User: {instr}"]
    for step in traj:
        th = step.get("thought")
        obs = step.get("observation")
        if th: lines.append(f"Assistant (thought): {th}")
        if obs: lines.append(f"Tool observation: {obs}")
    lines.append(f"Assistant: {ans}")
    return "\n".join(lines).strip()

def load_text_dataset(path: str) -> Dataset:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t or t.startswith("//") or not t.startswith("{"):
                continue
            try:
                obj = json.loads(t)
                rows.append({"text": render_record(obj)})
            except Exception:
                continue
    if not rows:
        raise ValueError(f"No training rows found in {path}.")
    return Dataset.from_list(rows)

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

if __name__ == "__main__":
    cfg = yaml.safe_load(open("train/config.yaml", "r"))
    model_name = cfg["base_model"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model
    load_kwargs = {}
    if cfg.get("load_4bit") or cfg.get("load_8bit"):
        load_kwargs["device_map"] = "auto"
        if cfg.get("load_4bit"):
            load_kwargs["load_in_4bit"] = True
        if cfg.get("load_8bit"):
            load_kwargs["load_in_8bit"] = True
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    # LoRA
    peft_cfg = LoraConfig(
        r=int(cfg["lora_r"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]),
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)

    # Dataset
    ds = load_text_dataset(cfg["dataset_path"])

    # Cast numerics
    args = TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        num_train_epochs=float(cfg["num_train_epochs"]),
        learning_rate=float(cfg["learning_rate"]),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.0)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        logging_steps=int(cfg.get("logging_steps", 10)),
        save_steps=int(cfg.get("save_steps", 250)),
        fp16=False,
        bf16=False,
        report_to=["none"],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=ds,
        dataset_text_field="text",
        max_seq_length=int(cfg.get("max_seq_len", 512)),
        packing=bool(cfg.get("packing", False)),
    )

    device = get_device()
    model.to(device)
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print(f"Saved adapter to: {cfg['output_dir']}")
