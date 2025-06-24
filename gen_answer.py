#!/usr/bin/env python
# gen_answer.py  –  LoRA-aware version
"""
Retrieve the best-matching course document, rerank it, and generate
a grounded answer (one paragraph max) using the LoRA-fine-tuned
DialoGPT model in ./gen_lora/
"""

import json, sys, faiss, torch
from pathlib import Path
from typing import List, Dict

import config  # your project-wide config file  ──────────────────────────────
# ---------------------------------------------------------------------------

MONGO_URI      = config.MONGO_URI
DB_NAME, COLL  = "course_advisor", "courses"

FAISS_INDEX    = config.FAISS_INDEX_FILE
FAISS_ID_MAP   = config.FAISS_IDMAP_FILE
KR_MODEL_PATH  = config.KR_MODEL_PATH
RERANKER_PATH  = config.RERANKER_NAME

BASE_MODEL     = "microsoft/DialoGPT-medium"   # same as during LoRA training
GEN_ADAPTERS   = "./gen_lora"                  # ← LoRA folder (adapter_config.json)

TOP_K_RETRIEVE = 8
TOP_K_RERANK   = 3
MAX_NEW_TOKENS = 80
# ---------------------------------------------------------------------------

# 1) Mongo ───────────────────────────────────────────────────────────────────
from pymongo import MongoClient
mongo     = MongoClient(MONGO_URI)
coursecol = mongo[DB_NAME][COLL]

# 2) Dense Retriever (KR + FAISS) ────────────────────────────────────────────
from sentence_transformers import SentenceTransformer
kr_encoder = SentenceTransformer(KR_MODEL_PATH, device="cpu")

faiss_index = faiss.read_index(FAISS_INDEX)
with open(FAISS_ID_MAP, encoding="utf-8") as f:
    id_map: Dict[int, str] = json.load(f)      # faiss-id ➜ Mongo _id (hex str)


def dense_retrieve(query: str, k: int = TOP_K_RETRIEVE) -> List[Dict]:
    vec = kr_encoder.encode([query],
                            normalize_embeddings=True,
                            convert_to_numpy=True).astype("float32")
    D, I = faiss_index.search(vec, k)
    hits = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        mongo_id = id_map[idx]
        doc      = coursecol.find_one({"_id": mongo_id})
        doc.pop("embedding", None)             # drop heavy binary field
        hits.append({"doc": doc, "score": float(score)})
    return hits


# 3) Optional Cross-Encoder reranker ─────────────────────────────────────────
if Path(RERANKER_PATH).exists():
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder(RERANKER_PATH, device="cpu")
else:
    reranker = None


def rerank(query: str, hits: List[Dict]) -> Dict:
    if not reranker:
        return hits[0]["doc"] if hits else None
    pairs  = [[query, json.dumps(h["doc"], ensure_ascii=False)] for h in hits]
    scores = reranker.predict(pairs, convert_to_tensor=False)
    best   = int(scores.argmax())
    return hits[best]["doc"]


# 4) Generator  (base + LoRA adapters)  ──────────────────────────────────────
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

tok = AutoTokenizer.from_pretrained(BASE_MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

_base = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
gen_model = PeftModel.from_pretrained(_base, GEN_ADAPTERS).merge_and_unload()
gen_model.eval()              # inference mode
del _base                     # free a bit of memory


def template_answer(c: Dict) -> str:
    return (f"{c['name']} ({c['code']}) is a {c['credits']}-credit course "
            f"worth {c['ects']} ECTS. {c['description']}")


def generate_answer(query: str, course: Dict) -> str:
    # If the user asks for a summary, use a special prompt
    if "summaris" in query.lower() or "summariz" in query.lower():
        prompt = (
            "You are a university-course assistant. "
            "Provide a **one-sentence summary** of the course, using only the "
            "description text.\n\n"
            f"Description:\n{course['description']}\n\n"
            f"Q: {query}\nA:"
        )
    else:
        prompt = (
            "You are a helpful university-course assistant. "
            "Answer **only** with facts copied verbatim from the JSON context "
            "below.\n\n"
            f"JSON:\n{json.dumps(course, ensure_ascii=False)}\n\n"
            f"Q: {query}\nA:"
        )

    inputs = tok(prompt, return_tensors="pt",
                 truncation=True, padding=True).to(gen_model.device)

    with torch.inference_mode():
        out_ids = gen_model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,           # deterministic
            temperature=0.7, top_p=0.9,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
        )[0]

    full = tok.decode(out_ids, skip_special_tokens=True)
    answer = full.split("A:", 1)[-1].strip().split("\n", 1)[0].strip()

    # fallback if generation failed
    if len(answer) < 5 or answer.lower().startswith(("sorry", "i don")):
        answer = template_answer(course)
    return answer.split("<eos>")[0].strip()


# 5) CLI ─────────────────────────────────────────────────────────────────────
def main():
    query = " ".join(sys.argv[1:]).strip() or input("Enter your question: ").strip()
    hits  = dense_retrieve(query)
    if not hits:
        print("No matching course found.")
        return
    course = rerank(query, hits)
    ans    = generate_answer(query, course)
    print(f"\nQ: {query}\nA: {ans}")


if __name__ == "__main__":
    main()
