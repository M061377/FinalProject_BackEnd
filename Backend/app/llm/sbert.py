# app/llm/sbert.py
import os
import threading
from typing import Iterable, Union
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from app.core.config import SBERT_DIR, DEVICE

# 전역 모델 객체와 락
_model_lock = threading.Lock()
_model: SentenceTransformer | None = None


def _assert_model_path():
    # SBERT 모델 경로가 올바른지 확인
    if not SBERT_DIR or not os.path.exists(SBERT_DIR):
        raise FileNotFoundError(f"SBERT_DIR 경로가 잘못되었습니다: {SBERT_DIR}")


def get_model() -> SentenceTransformer:
    # SBERT 모델을 한 번만 로드해서 재사용
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _assert_model_path()
                m = SentenceTransformer(str(SBERT_DIR))
                if DEVICE == "cuda" and torch.cuda.is_available():
                    m = m.to("cuda")
                else:
                    m = m.to("cpu")
                _model = m
    return _model


def embed(
    texts: Union[str, Iterable[str]],
    normalize: bool = True,
    batch_size: int = 32,
) -> np.ndarray:
    # 입력 문장을 SBERT 임베딩 벡터로 변환
    model = get_model()
    single = isinstance(texts, str)
    if single:
        texts = [texts]

    vecs = model.encode(
        list(texts),
        convert_to_numpy=True,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=False,
    )

    if normalize:
        vecs = _l2_normalize(vecs)

    return vecs[0] if single else vecs


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # 두 벡터 간 코사인 유사도 계산
    a_n = a / (np.linalg.norm(a) + 1e-12)
    b_n = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a_n, b_n))


def cosine_similarity_matrix(query: np.ndarray, docs: np.ndarray) -> np.ndarray:
    # 하나의 쿼리 벡터와 여러 문서 벡터들 간 코사인 유사도 계산
    q = query / (np.linalg.norm(query) + 1e-12)
    d = docs / (np.linalg.norm(docs, axis=1, keepdims=True) + 1e-12)
    return (d @ q).astype(np.float32)


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # L2 정규화 수행
    if x.ndim == 1:
        return x / (np.linalg.norm(x) + eps)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def encode(text: str) -> np.ndarray:
    # 기존 recommend 코드와의 호환용 래퍼
    return embed(text)  # embed는 문자열도 입력 가능하게 구현됨
