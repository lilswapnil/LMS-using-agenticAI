# Agentic AI — Trainable LMS/ERP Assistant

A small, reproducible pipeline to fine-tune a chat LLM with tool-augmented trajectories (SFT) for Learning Management Systems (LMS) and Enterprise Resource Planning (ERP) tasks. It targets CPU/Mac MPS and small open models for quick iteration, using LoRA adapters.

Key features
- Tool-augmented SFT: imitate trajectories with thoughts, tool calls, and final answers.
- Lightweight: runs on CPU/MPS; defaults to TinyLlama-1.1B-Chat.
- Reproducible data format: JSONL with instruction/tools/trajectory/final_answer.
- Simple eval with context, exact matching, “contains,” and semantic similarity.

Example LMS/ERP use-cases
- Answer policy questions (grading, attendance, deadlines) with retrieval context.
- Extract structured values from text (e.g., “YoY growth,” “GPA,” “credit load”).
- Tutor-like algebra steps for pedagogy (trajectory shows rationale/actions).
- Ticket triage and routing (IT/HR) with a tool registry of back-end actions.

---

## 1. Repository structure

- train/
  - sft_train.py — LoRA SFT trainer (TRL SFTTrainer)
  - config.yaml — training hyperparameters and dataset path
- evals/
  - simple_eval.py — evaluation script with optional LoRA adapter
- data/
  - sft_samples.jsonl — small mixed examples (retrieval + algebra)
  - Algebra SFT Dataset.jsonl — algebra-focused trajectories
  - README.md — data schema
- agent/
  - core.py — minimal ToolRegistry (spec + callable tools)
- outputs/
  - sft/ — default training output (LoRA adapter)

---

## 2. Environment setup

Recommended: Python 3.12 on macOS. Python 3.13 is possible with tokenizers>=0.20.3 but may be less stable on some setups.

- Create venv and install dependencies:
  - python3 -m venv .venv && source .venv/bin/activate
  - pip install -U pip setuptools wheel
  - pip install -r requirements.txt

Note
- On Apple Silicon, the code auto-selects MPS if available. Otherwise, it falls back to CPU.
- Avoid mixing conda “base” with this venv.

---

## 3. Data format (SFT)

Each JSONL line is one example with the following fields:
- instruction: user goal/task (e.g., “Find ACME Corp’s YoY revenue growth…”)
- tools: list of available tools (name/desc/args_schema) visible to the agent
- trajectory: ordered steps with thought, action, observation
- final_answer: the final textual answer

Example (policy extraction with retrieval):
{"instruction":"Find ACME Corp's YoY revenue growth and answer with a number.","tools":[{"name":"retrieve","desc":"semantic search over notes","args_schema":{"query":"str"}}],"trajectory":[{"thought":"Search for YoY growth.","action":{"tool":"retrieve","args":{"query":"revenue grew YoY"}},"observation":"Revenue grew 18% YoY to $1.2B."}],"final_answer":"18%"}

Example (algebra pedagogy):
{"instruction":"Solve for x: 3x + 5 = 20","tools":[{"name":"algebra","desc":"symbolic steps","args_schema":{}}],"trajectory":[{"thought":"Subtract 5 from both sides.","action":{"tool":"algebra","args":{"operation":"subtract","value":5}},"observation":"3x = 15"},{"thought":"Divide by 3.","action":{"tool":"algebra","args":{"operation":"divide","value":3}},"observation":"x = 5.0"}],"final_answer":"x = 5"}

Data hygiene: JSONL must have one valid JSON object per line. Remove comments and stray text.

- Clean provided files (macOS):
  - sed -e 's#//.*$##' -e '/^[[:space:]]*$/d' data/sft_samples.jsonl > data/sft_samples.clean.jsonl
  - sed -e 's#//.*$##' -e '/^[[:space:]]*$/d' 'data/Algebra SFT Dataset.jsonl' | grep -E '^[[:space:]]*\{' > data/algebra.clean.jsonl

---

## 4. Configuration

Edit train/config.yaml to pick the base model and data. Defaults target small CPU/MPS-friendly runs.

Example:
base_model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
revision: null
load_4bit: false
load_8bit: false

bnb_quant_type: nf4
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05

max_seq_len: 512
per_device_train_batch_size: 1
per_device_eval_batch_size: 1
gradient_accumulation_steps: 16
num_train_epochs: 1
learning_rate: 2e-4
warmup_ratio: 0.03
weight_decay: 0.0
logging_steps: 10
save_steps: 250

dataset_path: data/sft_samples.clean.jsonl
output_dir: outputs/sft
packing: false

Notes
- On Mac, keep load_4bit/load_8bit false unless you know your quant stack.
- packing: false is safer for small datasets; if true, reduce max_seq_len (e.g., 256–512).

---

## 5. Training

Run:
- source .venv/bin/activate
- python train/sft_train.py

What happens
- Loads base_model and attaches a LoRA adapter (q/k/v/o/gate/up/down proj).
- Converts JSONL to single “text” strings with structured prompt format.
- Trains with TRL SFTTrainer using CPU or MPS automatically.
- Saves adapter weights and tokenizer to outputs/sft.

Tips
- If you hit accelerate API mismatches, ensure accelerate>=0.34.2 (as in requirements).
- For tiny datasets, increase gradient_accumulation_steps to simulate larger batches.

---

## 6. Evaluation

Add evaluation samples with optional context to reduce hallucinations:
- data/eval_samples.jsonl:
{"query":"What was ACME's YoY revenue growth?","context":"Revenue grew 18% YoY to $1.2B.","expected":"18%"}

Run base model:
- python evals/simple_eval.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --eval_file data/eval_samples.jsonl --print-outputs --max_new_tokens 32

Run with adapter:
- python evals/simple_eval.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --adapter outputs/sft --eval_file data/eval_samples.jsonl --print-outputs --max_new_tokens 32

Metrics
- exact: exact string match with expected
- contains: expected substring appears in model output
- semantic@τ: cosine similarity between answer and expected using an encoder (default sentence-transformers/all-MiniLM-L6-v2)

The evaluator supports chat templates automatically and decodes only new tokens.

---

## 7. Agent tools (for LMS/ERP)

See agent/core.py:
- Register tool(name, fn, desc)
- Call tools by name with args
- Expose tool specs to the model (tools list in SFT examples)

Examples for LMS/ERP:
- retrieve: semantic search over policies/syllabi/announcements
- attendance_lookup: read attendance % from SIS
- gradebook_query: fetch grades/weights from LMS
- ticket_create: open a ticket in IT/HR systems

Train the agent with these tool specs in the tools field, and show realistic trajectory steps with observations.

---

## 8. Reproducibility and experiment protocol

- Python and packages: requirements.txt pins a known-good stack for Transformers 4.46+ and tokenizers 0.20+.
- Hardware: CPU and MPS are supported via auto device selection.
- Data splits: keep evaluation JSONL separate (no leakage).
- Seeds: for strict reproducibility, add a fixed seed in sft_train.py (torch, numpy, random).
- Logging: use logging_steps in config; integrate wandb later if desired.

Suggested ablations
- With vs. without context in eval.
- Vary max_seq_len and packing.
- Compare pure instruction-answer vs. tool-augmented trajectories.
- Use a larger base model if you have a GPU (e.g., 7B).

---

## 9. Common issues

- Tokenizers build error on Python 3.13:
  - Use Python 3.12, or keep tokenizers>=0.20.3 and transformers>=4.46.
- Accelerate API mismatch:
  - Ensure accelerate>=0.34.2 (see requirements).
- JSONL parse errors:
  - Clean comments/blank lines as shown in Section 3.
- Adapter not found in eval:
  - Ensure outputs/sft has adapter_config.json and adapter_model.safetensors or omit --adapter.

---

## 10. License and attribution

- This repository uses open models from Hugging Face for research purposes.
- Ensure compliance with each model’s license.
- Cite this repo in academic work as:
  - “Agentic AI — Trainable LMS/ERP Assistant (2025). https://github.com/<your-org>/agentic-ai-trainable”

---

## 11. Roadmap to a deployment‑ready product

This section outlines the technical work to evolve this research pipeline into a secure, reliable LMS/ERP assistant for production.

1) Serving and APIs
- Model serving
  - Merge LoRA → standalone model for simpler inference, or load base+adapter at runtime.
  - Use an inference server (e.g., vLLM, TGI, OpenAI-compatible FastAPI) with streaming tokens.
- API contract
  - POST /v1/chat: messages[], optional tools[], context[], return tool_calls, citations.
  - POST /v1/embeddings: batch text → vectors (for retrieval).
  - POST /v1/eval: run offline eval suite on a tagged model build.
- Latency/throughput
  - Enable KV-cache; pick static max_new_tokens; prefer greedy/top‑p small values.
  - Quantization (int8/4bit) and tensor parallel where GPUs are available.

2) Retrieval and data connectors (LMS/ERP)
- Connectors
  - LMS: Canvas/Moodle/Blackboard APIs (courses, assignments, submissions, announcements).
  - ERP/HCM: Workday/Oracle/SAP for HR, finance, purchasing, tickets.
  - File stores: S3/GCS/Azure Blob, Box/Drive/SharePoint for policies and docs.
- Indexing
  - Chunking with metadata (course_id, term, doc_type); multilingual tokenization.
  - Periodic incremental sync; soft‑delete and versioning.
- RAG policy
  - Per‑request authorization on retrieved chunks; add citations and timestamps to answers.
  - Domain-specific reranking; query rewriting with guardrails to minimize leakage.

3) Security, privacy, and compliance
- Identity and access
  - SSO (SAML/OIDC) with institutional IdP; enforce RBAC/ABAC at tool and data layers.
  - Per‑tenant encryption keys; least‑privilege credentials for connectors.
- Data protection
  - PII/PHI redaction and masking; on‑prem or VPC isolation as required.
  - FERPA, GDPR, SOC2 controls; data retention policies and right‑to‑erasure flows.
- Auditability
  - Structured audit logs for prompts, retrieved sources, tool calls, and final outputs.
  - Signed logs and tamper detection; incident response runbooks.

4) Reliability and observability
- SLOs and autoscaling
  - Define latency/error SLOs; HPA on CPU/GPU; queue backpressure and circuit breakers.
- Telemetry
  - Tracing (OpenTelemetry), structured logs, metrics (tokens/s, latency p95/p99, OOMs).
  - Prompt/output sampling to a privacy‑safe data store for offline review.
- Health and resilience
  - Readiness/liveness probes; warm pools of models; graceful draining on deploy.

5) Model lifecycle and governance
- Registries and versioning
  - Track base model, adapter hash, merged build, dataset snapshot, and config.
  - Semantic versioning for model releases; provenance for experiments.
- CI/CD
  - Pre‑merge checks: unit tests, data validations, small‑batch dry‑runs, eval gates.
  - Canary deploys and automatic rollback on SLO regression.
- Human‑in‑the‑loop (HITL)
  - Feedback UI for staff to approve/edit answers; mine high‑value corrections for SFT/RLHF.

6) Evaluation, safety, and policy controls
- Eval suites
  - Domain evals (policies, schedules, grade rules), adversarial prompts, privacy leakage tests.
  - Data drift monitors for retrieval corpora and usage patterns.
- Safety layers
  - Prompt templates with system policies; allowlist tools; response classifiers (PII leaks, toxicity).
  - Refusal and escalation policies to humans for out‑of‑scope or risky requests.

7) Performance and cost optimization
- Inference
  - Quantize (AWQ/GPTQ/4‑bit), compile (torch.compile), and use flash attention where available.
  - Batch requests for embeddings; cache RAG results; reuse KV across turns.
- Training
  - Dataset curation and dedup; curriculum mixing; LoRA target module ablations.
  - Distill to smaller models for edge use (advisors’ laptops, kiosks).
- Storage
  - Cold/hot tiering for indices; content‑addressable storage for artifacts.

8) Multi‑tenant and enterprise integration
- Tenant isolation
  - Separate indices, encryption keys, and resource quotas per institution.
- Policy engines
  - Centralize authorization (OPA/Cedar) to evaluate fine‑grained policies at request time.
- LMS/ERP change management
  - Schema evolution, retry policies, and contract tests for upstream API changes.

9) Reference deployment (suggested)
- Infrastructure
  - Kubernetes with GPU/MPS nodes as available; Terraform for reproducible infra.
  - Secrets via external secret stores (e.g., AWS Secrets Manager).
- Services
  - api-gateway (authn/z, rate limits) → router → inference → retrieval → tools.
  - Background workers for indexing, evaluations, fine‑tuning jobs.
- Images and artifacts
  - Minimal, reproducible Dockerfiles; SBOMs; signed artifacts; vulnerability scans.

10) Developer ergonomics
- Local dev
  - Make targets for train/eval/format/test; VS Code tasks; pre‑commit hooks.
- Testing
  - Unit tests for tool calls and prompt renderers; golden tests for eval queries.
- Docs
  - Runbooks for oncall; data onboarding guides; policy authoring for institutions.

Action checklist
- Add a FastAPI inference service with streaming and OpenAI‑compatible endpoints.
- Implement a minimal RAG service with connectors and a per‑tenant index.
- Add CI eval gates (exact/contains/semantic) that must meet thresholds before release.
- Introduce model registry entries per training run; tag merged adapters.
- Add audit logging and PII redaction; wire to SIEM.
- Ship dashboards (latency, tokens/s, error rate) and alerts tied to SLOs.
- Harden prompts and add a safety classifier for outputs.
- Provide a deployment helm chart and Terraform module for one‑click installs.

---

## 11. Quick start (cheat sheet)

- Setup:
  - python3 -m venv .venv && source .venv/bin/activate
  - pip install -U pip && pip install -r requirements.txt
- Clean data (recommended):
  - sed -e 's#//.*$##' -e '/^[[:space:]]*$/d' data/sft_samples.jsonl > data/sft_samples.clean.jsonl
- Edit config:
  - train/config.yaml → dataset_path: data/sft_samples.clean.jsonl
- Train:
  - python train/sft_train.py
- Eval (base + context):
  - python evals/simple_eval.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --eval_file data/eval_samples.jsonl --print-outputs --max_new_tokens 32
- Eval (with adapter):
  - python evals/simple_eval.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --adapter outputs/sft --eval_file data/eval_samples.jsonl --print-outputs --max_new_tokens 32
