from typing import Dict, Any, List
from sentence_transformers import SentenceTransformer
import numpy as np

class ToyCorpus:
    def __init__(self, docs: List[str]):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.docs = docs
        self.embs = self.model.encode(docs, normalize_embeddings=True)
    def search(self, query: str, k: int = 3) -> str:
        q = self.model.encode([query], normalize_embeddings=True)[0]
        sims = (self.embs @ q)
        idx = np.argsort(-sims)[:k]
        return "\n".join(self.docs[i] for i in idx)

DEFAULT_DOCS = [
    "Revenue grew 18% YoY to $1.2B.",
    "Operating margin improved to 14%.",
    "Risks include supply chain volatility and foreign exchange headwinds.",
]

CORPUS = ToyCorpus(DEFAULT_DOCS)

# Tool function

def tool_retrieve(args: Dict[str, Any]) -> str:
    return CORPUS.search(args.get("query", ""), k=3)
