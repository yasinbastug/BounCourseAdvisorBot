#!/usr/bin/env python
"""
Rebuild FAISS index from course embeddings stored in MongoDB.
  – uses sequential int64 IDs inside FAISS
  – saves a JSON list mapping position → Mongo _id (hex string)

Run:  python build_faiss.py
"""

import os, json, struct, sys, tqdm, numpy as np, faiss
from bson import ObjectId
from pymongo import MongoClient, errors
from sentence_transformers import SentenceTransformer   # not used here but handy
from config import MONGO_URI

# ---------- Mongo connection -------------------------------------------------
DB_NAME     = "course_advisor"
COLL_NAME   = "courses"

try:
    mc   = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10_000)
    mc.admin.command("ping")
except errors.ServerSelectionTimeoutError as e:
    sys.exit(f"[ERROR] Cannot reach MongoDB: {e}")

coll = mc[DB_NAME][COLL_NAME]

# ---------- Gather vectors ---------------------------------------------------
cursor = coll.find(
    {"embedding": {"$exists": True}},
    {"embedding": 1}        # project only what we need
)

vectors   = []
id_lookup = []              # idx -> real ObjectId (as str)

for i, doc in enumerate(tqdm.tqdm(cursor, desc="Loading embeddings")):
    vec = np.frombuffer(doc["embedding"], dtype=np.float32)
    vectors.append(vec)
    id_lookup.append(str(doc["_id"]))   # keep the 24-hex string

if not vectors:
    sys.exit("No embeddings found, aborting.")

vectors = np.vstack(vectors).astype("float32")
dim     = vectors.shape[1]

print(f"[OK] Loaded {len(vectors):,} vectors  (dim={dim})")

# ---------- Build / train / add ---------------------------------------------
index_flat = faiss.IndexFlatIP(dim)          # cosine if you normalised vectors
index      = faiss.IndexIDMap(index_flat)    # lets us pass our own IDs

faiss_ids  = np.arange(len(vectors), dtype='int64')  # 0,1,2,...
index.add_with_ids(vectors, faiss_ids)

print(f"[OK] Index size = {index.ntotal:,}")

# ---------- Persist ----------------------------------------------------------
os.makedirs("faiss_store", exist_ok=True)
faiss.write_index(index, "faiss_store/courses.index")

with open("faiss_store/courses_id_map.json", "w") as f:
    json.dump(id_lookup, f)

print("[DONE] Saved FAISS index and id-map ➜  faiss_store/")
