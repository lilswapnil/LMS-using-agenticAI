# Agentic AI — Trainable LMS/ERP Assistant

A reproducible research scaffold for fine-tuning small open-source LLMs to act as tool-augmented assistants for **Learning Management Systems (LMS)** and **Enterprise Resource Planning (ERP)** tasks.  
Designed for lightweight experimentation on CPU and Apple MPS, using **LoRA adapters** for rapid iteration.

---

## 🔍 Research Purpose

The aim of this project is to explore **agentic AI** capabilities in academic and enterprise contexts by training a chat LLM to:
- **Reason step-by-step** using tool-augmented trajectories (thought → tool call → observation → final answer).
- **Integrate external tools** for policy lookup, grade extraction, attendance queries, and algebra tutoring.
- **Serve as a basis** for further RAG integration, enterprise connectors, and real-time decision support.

---

## ✨ Key Features
- **Tool-augmented SFT** — Model learns to interleave reasoning with tool usage.
- **Lightweight & Portable** — Runs on CPU/MPS without requiring high-end GPUs.
- **Structured JSONL Dataset** — Instruction, tools, trajectory, final answer.
- **Evaluation Suite** — Measures exact match, substring match, and semantic similarity.
- **Modular Tool Registry** — Easily register and test domain-specific tools.

---

## 📂 Repository Structure
```

train/           # Training scripts (LoRA SFT)
evals/           # Evaluation scripts & metrics
data/            # SFT and evaluation datasets
agent/           # Tool registry and tool execution logic
outputs/         # LoRA adapter weights after training

````

---

## 📦 Installation
```bash
# Create environment
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip setuptools wheel

# Install dependencies
pip install -r requirements.txt
````

> **Note:** On Apple Silicon, MPS acceleration will be used automatically if available.

---

## 📊 Dataset Format

Each training example is stored as a **single JSON object per line** in a `.jsonl` file:

```json
{
  "instruction": "Solve for x: 3x + 5 = 20",
  "tools": [
    {
      "name": "algebra",
      "desc": "symbolic steps",
      "args_schema": {}
    }
  ],
  "trajectory": [
    {
      "thought": "Subtract 5 from both sides.",
      "action": { "tool": "algebra", "args": {"operation": "subtract", "value": 5} },
      "observation": "3x = 15"
    },
    {
      "thought": "Divide by 3.",
      "action": { "tool": "algebra", "args": {"operation": "divide", "value": 3} },
      "observation": "x = 5"
    }
  ],
  "final_answer": "x = 5"
}
```

**Dataset hygiene:**
Remove comments/blank lines before training:

```bash
sed -e 's#//.*$##' -e '/^[[:space:]]*$/d' data/sft_samples.jsonl > data/sft_samples.clean.jsonl
```

---

## ⚙️ Configuration

Edit `train/config.yaml` to select:

* **Base model** (default: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`)
* **Dataset path**
* LoRA parameters (`lora_r`, `lora_alpha`, `lora_dropout`)
* Training hyperparameters (batch size, epochs, learning rate)

---

## 🚀 Training

```bash
python train/sft_train.py
```

This:

1. Loads the base model.
2. Attaches a LoRA adapter to attention and MLP layers.
3. Formats data into structured prompts.
4. Trains using **TRL’s SFTTrainer** on CPU or MPS.

---

## 📈 Evaluation

```bash
python evals/simple_eval.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --adapter outputs/sft \
  --eval_file data/eval_samples.jsonl \
  --print-outputs --max_new_tokens 32
```

Metrics:

* **Exact:** Output exactly matches expected answer.
* **Contains:** Expected string is present in output.
* **Semantic@τ:** Cosine similarity between answer and expected.

---

## 🛠 Tool Integration Examples

| Tool Name           | Purpose                                 |
| ------------------- | --------------------------------------- |
| `retrieve`          | Search policies, syllabi, announcements |
| `attendance_lookup` | Fetch attendance % from SIS             |
| `gradebook_query`   | Retrieve grades and weights from LMS    |
| `ticket_create`     | Open tickets in IT/HR systems           |

---

## 📅 Roadmap

* 🔗 LMS/ERP API connectors for real retrieval.
* 🛡️ PII redaction, RBAC/ABAC security, audit logging.
* 📊 Model serving with FastAPI and streaming output.
* 🧪 Continuous evaluation gates in CI/CD.
* 📈 Dashboarding for observability.

---

## ⚖️ License

This repository uses open models from Hugging Face for research purposes.
Ensure compliance with each model’s license before deployment.

---

## 📚 Citation

If you use this project in academic work:

```
@misc{agenticai2025,
  title  = {Agentic AI — Trainable LMS/ERP Assistant},
  author = {Bhalerao, Swapnil},
  year   = {2025},
  url    = {https://github.com/lilswapnil/LMS-using-agenticAI}
}
```

