import json, argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def build_prompt(tok, query: str) -> str:
    # Use chat template if available
    chat_tmpl = getattr(tok, "chat_template", None)
    apply_chat = getattr(tok, "apply_chat_template", None)
    if chat_tmpl and apply_chat:
        messages = [{"role": "user", "content": query}]
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"Question: {query}\nAnswer:"

def generate(model, tok, prompt, max_new_tokens=128):
    device = model.device
    ids = tok(prompt, return_tensors="pt")
    ids = {k: v.to(device) for k, v in ids.items()}
    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
        )
    # Decode only new tokens
    new_tokens = out[0][ids["input_ids"].shape[-1]:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval_file", default="data/eval_samples.jsonl")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    device = get_device()
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device)
    model.eval()

    total = 0
    exact = 0
    contains = 0
    for line in open(args.eval_file, "r", encoding="utf-8"):
        ex = json.loads(line)
        prompt = build_prompt(tok, ex["query"])
        ans = generate(model, tok, prompt, max_new_tokens=args.max_new_tokens)
        total += 1
        if ans.strip() == ex["expected"]:
            exact += 1
        if ex["expected"].lower() in ans.lower():
            contains += 1
    print({"total": total, "exact": exact/total, "contains": contains/total})
