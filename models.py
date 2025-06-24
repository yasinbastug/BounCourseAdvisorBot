from typing import Any
import numpy as np
from bson.binary import Binary
from sentence_transformers import SentenceTransformer
import torch
from config import EMBED_DIM, KR_MODEL_NAME

# ---------- Encoder ----------------------------------------------------------
_encoder: SentenceTransformer | None = None

def get_encoder() -> SentenceTransformer:
    global _encoder
    if _encoder is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _encoder = SentenceTransformer(KR_MODEL_NAME, device=device)
    return _encoder

# ---------- (de)serialisation helpers ---------------------------------------
def to_binary(vec: np.ndarray) -> Binary:
    """float32 ndarray ➜ Binary for Mongo"""
    return Binary(vec.astype("float32").tobytes())

def from_binary(b: Binary) -> np.ndarray:
    return np.frombuffer(b, dtype="float32", count=EMBED_DIM)
