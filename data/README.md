Data format: JSONL, one example per line.
Fields:
- instruction: user goal / task
- tools: list of available tools (names & signatures)
- trajectory: list of steps [{"thought": str, "action": {"tool": str, "args": {}}, "observation": str}] 
- final_answer: the final response the agent should produce

We train via SFT to imitate high‑quality trajectories (tool‑augmented CoT). Later, do DPO using pairwise {chosen,rejected} trajectories.
