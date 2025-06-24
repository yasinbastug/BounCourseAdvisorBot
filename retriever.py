import json, faiss, numpy as np
from sentence_transformers import SentenceTransformer
from bson import ObjectId
from pymongo import MongoClient
from config import MONGO_URI

class CourseRetriever:
    """
    Light wrapper around the FAISS index + Mongo lookup.
    """

    def __init__(
        self,
        index_path      = "faiss_store/courses.index",
        id_map_path     = "faiss_store/courses_id_map.json",
        kr_model_path   = "./kr_first_tuned",
        mongo_uri       = MONGO_URI,
        db_name         = "course_advisor",
        coll_name       = "courses"
    ):
        # vector part
        self.index   = faiss.read_index(index_path)
        with open(id_map_path) as f:
            self.id_map = json.load(f)
        # encoder
        self.kr      = SentenceTransformer(kr_model_path, device="cpu")
        # mongo
        self.coll    = MongoClient(mongo_uri)[db_name][coll_name]

    # ---------------------------------------------------------

    def search(self, text:str, k:int=5):
        """
        returns list[dict] of length ≤ k:
            { _id, score, doc }
        """
        # 1) encode -> L2-normalised vector
        vec = self.kr.encode([text],
                             normalize_embeddings=True,
                             convert_to_numpy=True).astype("float32")
        # 2) FAISS search
        D, I = self.index.search(vec, k)
        hits  = []
        for score, idx in zip(D[0], I[0]):
            # idx = -1 means "no more"
            if idx == -1:
                continue        
            doc     = self.coll.find_one({"_id": self.id_map[idx]})
            hits.append({"_id": self.id_map[idx], "score": float(score), "doc": doc})
        return hits
