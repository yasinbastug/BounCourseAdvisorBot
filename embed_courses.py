"""
Run this script any time you add / edit courses or when you fine-tune KR.
"""
import pymongo, tqdm, numpy as np
from typing import Any
from config        import *
from models        import get_encoder, to_binary

client  = pymongo.MongoClient(MONGO_URI)
coll    = client[MONGO_DB][COURSE_COLL]
encoder = get_encoder()

def generate_doc_text(doc: dict[str,Any]) -> str:
    return f"{doc.get('code','')} {doc.get('name','')} {doc.get('description','')}"

def main():
    total = coll.count_documents({})
    for doc in tqdm.tqdm(coll.find({}, projection=["_id","code","name","description"]),
                         total=total, desc="Embedding"):
        vec = encoder.encode(generate_doc_text(doc), normalize_embeddings=True)
        coll.update_one({"_id": doc["_id"]},
                        {"$set": {"embedding": to_binary(vec)}})

    print("✅  embeddings updated for", total, "courses")

if __name__ == "__main__":
    main()
