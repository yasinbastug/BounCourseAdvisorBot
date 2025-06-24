# rerank.py
"""
Takes a user query
→ top-K candidates from CourseRetriever (bi-encoder + FAISS)
→ Cross-Encoder scores & re-orders
→ returns the TOP_N best docs + scores
"""

import numpy as np
import torch
from sentence_transformers import CrossEncoder
from retriever import CourseRetriever         # <-- your wrapper

# ---------------------------------------------------------------------
# CONFIG
TOP_K      = 20      # pull this many from FAISS
TOP_RETAIN = 3       # keep this many after re-ranking
CE_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"   # or ./ce_tuned
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------------------------------------------------------

ce   = CrossEncoder(CE_MODEL, device=DEVICE)
retr = CourseRetriever()                      # uses CPU encoder by default


def search(query: str, k: int = TOP_RETAIN):
    """
    Returns two lists of length ≤ k
        best_docs  – Mongo documents
        ce_scores  – float relevance scores
    """
    # 1) bi-encoder + FAISS
    hits = retr.search(query, k=TOP_K)        # [{doc, score, _id}, …]

    # 2) Cross-Encoder re-ranking
    pairs   = [[query, h["doc"]["description"][:512]] for h in hits]
    scores  = ce.predict(pairs)
    order   = np.argsort(scores)[::-1][:k]    # highest first

    best_docs  = [hits[i]["doc"]   for i in order]
    best_score = [float(scores[i]) for i in order]
    return best_docs, best_score
