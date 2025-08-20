# app/utils/embeddings.py
import pickle, numpy as np
from typing import Dict
from app.core.config import EMBEDDING_PKL_PATH

_cache = None


def load_embeddings() -> Dict[str, np.ndarray]:
    global _cache
    if _cache is None:
        with open(EMBEDDING_PKL_PATH, "rb") as f:
            data = pickle.load(f)  # {isbn13: list/np.array}
        _cache = {str(k): np.asarray(v, dtype="float32") for k, v in data.items()}
    return _cache
