Agentic AI – Training‑First

Quick start:
1) python3 -m venv .venv && source .venv/bin/activate
2) pip install --upgrade pip && pip install -r requirements.txt
3) Edit train/config.yaml → set a loadable base_model (e.g., "HuggingFaceH4/zephyr-7b-beta" or a small instruct model).
4) Seed data lives in data/sft_samples.jsonl – add more trajectories.
5) Train:  python train/sft_train.py
6) Eval:   python evals/simple_eval.py --model outputs/sft

Notes:
- If you lack GPU, choose a tiny model (e.g., phi‑2 or a 1‑3B instruct) and set load_8bit=false, load_4bit=false.
- Next steps: collect more **trajectory** data (tool‑augmented CoT), then run **DPO** with TRL on chosen vs rejected trajectories.
- For rollouts, wrap your tools in agent/core.py and write real envs in envs/.
