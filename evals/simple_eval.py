import json, argparse, os
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from peft import PeftModel

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def build_prompt(tok, query: str, context: str | None = None) -> str:
    user = f"{('Context:\\n' + context + '\\n\\n') if context else ''}Question: {query}\nAnswer with the exact value only."
    apply_chat = getattr(tok, "apply_chat_template", None)
    if apply_chat and getattr(tok, "chat_template", None):
        return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
    return user

def extract_exact(answer: str) -> str:
    import re
    m = re.search(r"\b\d+(?:\.\d+)?\s*%", answer)
    if m: return m.group(0).strip()
    m = re.search(r"\b\d+(?:\.\d+)?\b", answer)
    return (m.group(0) if m else answer).strip()

def generate(model, tok, prompt, max_new_tokens=128):
    device = model.device
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tok.eos_token_id,
            pad_token_id=pad_id,
        )
    # Decode only newly generated tokens
    new_tokens = out[0][inputs["input_ids"].shape[-1]:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()

def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

class SimpleEmbedder:
    def __init__(self, model_name: str, device: str):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self.device = device

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        batch = self.tok(texts, padding=True, truncation=True, return_tensors="pt")
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            out = self.model(**batch)
            emb = mean_pool(out.last_hidden_state, batch["attention_mask"])
            emb = F.normalize(emb, p=2, dim=-1)
        return emb

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Base CausalLM (HF id or local path)")
    ap.add_argument("--adapter", default=None, help="Optional LoRA adapter dir (e.g., outputs/sft)")
    ap.add_argument("--eval_file", default="data/eval_samples.jsonl")
    ap.add_argument("--print-outputs", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--sim_threshold", type=float, default=0.6)
    ap.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    device = get_device()

    # Tokenizer from base model (adapter dirs typically lack tokenizer/config)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id

    # Load model + optional adapter
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    if args.adapter:
        cfg_p = os.path.join(args.adapter, "adapter_config.json")
        w_p = os.path.join(args.adapter, "adapter_model.safetensors")
        if os.path.exists(cfg_p) and os.path.exists(w_p):
            model = PeftModel.from_pretrained(model, args.adapter)
        else:
            print(f"Warning: adapter not found at {args.adapter}, skipping.", flush=True)
    model.to(device)
    model.eval()

    # Optional semantic metric
    embedder = SimpleEmbedder(args.embed_model, device)

    # Tolerant JSONL reader (ignores comments/blank lines)
    rows = []
    with open(args.eval_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            if not line.startswith("{"):
                continue
            rows.append(json.loads(line))

    total = exact = contains = semantic = 0
    for i, ex in enumerate(rows, 1):
        ctx = ex.get("context")
        prompt = build_prompt(tok, ex["query"], ctx)
        raw = generate(model, tok, prompt, max_new_tokens=args.max_new_tokens)
        ans = extract_exact(raw)
        gold = ex["expected"].strip()

        total += 1
        if ans == gold:
            exact += 1
        if gold.lower() in raw.lower():
            contains += 1

        pred_emb = embedder.encode(ans)
        gold_emb = embedder.encode(gold)
        sim = float(F.cosine_similarity(pred_emb, gold_emb).item())
        if sim >= args.sim_threshold:
            semantic += 1

        if args.print_outputs:
            print(f"\n=== {i} ===")
            print("Q:", ex["query"])
            if ctx: print("Context:", ctx)
            print("Pred:", raw, "=>", ans)
            print("Gold:", gold)
            print(f"sim={sim:.3f}")

    metrics = {
        "total": total,
        "exact": (exact / total) if total else 0.0,
        "contains": (contains / total) if total else 0.0,
        f"semantic@{args.sim_threshold:.2f}": (semantic / total) if total else 0.0,
    }
    print(metrics)
