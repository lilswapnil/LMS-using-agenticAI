#!/usr/bin/env python3
import json, argparse, ast, re
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch

# tiny toy corpus + retrieve tool
DOCS = [
    "Revenue grew 18% YoY to $1.2B.",
    "Operating margin improved to 14%.",
    "Risks include supply chain volatility and foreign exchange headwinds.",
    "The distributive property: a(b+c)=ab+ac.",
    "Quadratic formula: x = (-b ± √(b^2-4ac))/(2a).",
]
EMB = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
DOC_E = EMB.encode(DOCS, normalize_embeddings=True)

def tool_retrieve(args):
    q = args.get("query","")
    qe = EMB.encode([q], normalize_embeddings=True)[0]
    sims = DOC_E @ qe
    idx = sims.argsort()[-3:][::-1]
    return "\n".join(DOCS[i] for i in idx)

def tool_calc(args):
    expr = str(args.get("expr",""))
    # safe arithmetic subset
    if not re.fullmatch(r"[0-9\.\+\-\*\/\(\)\s]+", expr):
        return "error: unsafe expression"
    try:
        return str(eval(expr, {"__builtins__":None}, {}))
    except Exception as e:
        return f"error: {e}"

TOOLS = {
    "retrieve": {"fn": tool_retrieve, "desc": "semantic search over notes", "args":{"query":"str"}},
    "calc": {"fn": tool_calc, "desc": "safe arithmetic", "args":{"expr":"str"}},
}

SYS_PROMPT = """You are an agent. Think step-by-step and emit one JSON object ONLY with:
{"thought": "...", "action": {"tool": "<retrieve|calc|finish>", "args": {...}}}
- Use "retrieve" to look up facts with {"query": "..."}
- Use "calc" for arithmetic with {"expr": "..."}
- Use "finish" when you have the final answer with {"answer": "..."}
NO extra text. Only a single JSON object each turn.
"""

def parse_json(text):
    # extract first {...} block
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m: return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="outputs/sft")
    ap.add_argument("--out", default="data/rollouts.jsonl")
    ap.add_argument("--max-steps", type=int, default=6)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")

    print("Enter a problem (type 'exit' to quit):")
    while True:
        query = input("> ").strip()
        if not query or query.lower()=="exit":
            break

        history = [{"role":"system","content":SYS_PROMPT},
                   {"role":"user","content":f"Task: {query}"}]
        traj = []
        obs = ""

        for step in range(args.max_steps):
            prompt = "\n".join([
                history[0]["content"],
                f'User: {history[1]["content"]}',
                f'Observation: {obs}' if obs else "",
            ])

            ids = tok(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=192, do_sample=False, pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
            text = tok.decode(out[0], skip_special_tokens=True)
            js = parse_json(text)
            if not js:
                print("! couldn't parse model output, stopping.")
                break

            thought = js.get("thought","")
            action = js.get("action",{})
            tool = action.get("tool","")
            args_dict = action.get("args",{})

            if tool == "finish":
                final = args_dict.get("answer","")
                traj.append({"thought": thought, "action":{"tool":tool,"args":args_dict}, "observation":"<done>"} )
                print(f"Final Answer: {final}")
                # write JSONL
                row = {
                    "instruction": query,
                    "tools": [{"name":k,"desc":v["desc"],"args_schema":v["args"]} for k,v in TOOLS.items()],
                    "trajectory": traj,
                    "final_answer": final
                }
                with open(args.out,"a",encoding="utf-8") as f: f.write(json.dumps(row, ensure_ascii=False)+"\n")
                print(f"Saved to {args.out}")
                break
            else:
                if tool not in TOOLS:
                    obs = f"error: unknown tool '{tool}'"
                else:
                    obs = TOOLS[tool]["fn"](args_dict)
                traj.append({"thought": thought, "action":{"tool":tool,"args":args_dict}, "observation": obs})
                print(f"[{tool}] -> {obs}")

        else:
            print("Max steps reached without finish().")

if __name__ == "__main__":
    main()
