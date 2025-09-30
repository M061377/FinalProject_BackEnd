# app/services/recommend_service.py
from typing import Any, Dict, List, Tuple, Optional
import numpy as np

from app.llm.sbert import encode  # 쿼리 임베딩
from app.llm.kluebert import classify  # 감정 분류
from app.utils.embeddings import load_embeddings  # {isbn13: vector}
from app.utils.emotion_cache import get_isbns_for_emotion  # 감정별 ISBN 후보

# 🔧 유틸 모듈을 별칭으로 임포트해서 이름 충돌/섀도잉 방지
import app.utils.firebase_util as fb

MAX_TOTAL = 100  # 상위 100권 제한


async def get_recommendations_paginated(
    query: str, page: int = 1, page_size: int = 10
) -> Dict[str, Any]:
    """
    추천 목록 API 로직 (유사도 내림차순, 100권 제한, 페이지네이션).
    """
    # 1) 감정 분류 + 쿼리 임베딩
    query_emotion, _conf = classify(query)
    qvec = encode(query)  # sbert.embed의 래퍼, L2 normalize 적용됨

    # 2) 감정별 후보 ISBN 추출
    isbns: List[str] = get_isbns_for_emotion(query_emotion) or []
    if not isbns:
        return RecommendResponse(
            query=query,
            emotion=query_emotion,
            page=page,
            page_size=page_size,
            total_candidates=0,
            items=[],
        ).model_dump()

    # 3) 임베딩 로드
    emb_dict: Dict[str, np.ndarray] = load_embeddings()

    # 4) 코사인 유사도 계산
    scored: List[Tuple[str, float]] = []
    for isbn in isbns:
        vec = emb_dict.get(isbn)
        if vec is None:
            continue
        v = np.array(vec, dtype=float)
        denom = np.linalg.norm(qvec) * np.linalg.norm(v)
        sim = float(np.dot(qvec, v) / denom) if denom != 0 else 0.0
        scored.append((isbn, sim))

    # 5) 유사도 내림차순 정렬 → 상위 100권 제한
    scored.sort(key=lambda x: x[1], reverse=True)
    scored = scored[:MAX_TOTAL]
    total_candidates = len(scored)

    # 6) 페이지네이션
    start = (page - 1) * page_size
    end = start + page_size
    page_slice = scored[start:end]
    page_isbns = [i for i, _ in page_slice]

    # 7) Firestore에서 목록용 최소 필드 배치 조회
    mini = fb.get_books_min_fields(
        page_isbns
    )  # {isbn13: {title, author, publisher, cover}}

    # 8) 응답 아이템 구성
    items: List[RecommendItem] = []
    for isbn, sim in page_slice:
        info = mini.get(isbn, {})
        items.append(
            RecommendItem(
                isbn13=isbn,
                title=info.get("title", ""),
                author=info.get("author", ""),  # 문자열
                publisher=info.get("publisher", ""),
                cover=info.get("cover", ""),
                similarity=round(sim, 4),
            )
        )

    resp = RecommendResponse(
        query=query,
        emotion=query_emotion,
        page=page,
        page_size=page_size,
        total_candidates=total_candidates,
        items=items,
    )
    return resp.model_dump()
