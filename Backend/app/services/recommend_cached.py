# app/services/recommend_cached.py
import numpy as np
from app.llm.sbert import encode
from app.llm.kluebert import classify
from app.utils.embeddings import load_embeddings
from app.utils.emotion_cache import get_isbns_for_emotion
from app.utils.firebase_util import get_book_by_isbn


async def get_recommendations(query: str):
    # --- 0) 하이퍼파라미터
    TOPK = 10
    TH_STRONG = 0.6
    TH_WEAK = 0.4

    # 1) 쿼리 감정/임베딩
    query_emotion, confidence = classify(query)
    query_embedding = encode(query)

    # 2) 감정으로 ISBN 후보
    isbns = get_isbns_for_emotion(query_emotion)
    if not isbns:
        # 프론트가 리스트 중간 공지를 쓰도록 items에도 notice 형태를 넣어줌
        return {
            "query": query,
            "query_emotion": query_emotion,
            "items": [
                {
                    "kind": "notice",
                    "level": "info",
                    "text": "해당 감정에 맞는 책이 없습니다.",
                }
            ],
            "message": "해당 감정에 맞는 책이 없습니다.",  # (하위 호환)
        }

    # 3) 임베딩 로드
    embedding_dict = load_embeddings()

    # 4) 코사인 유사도
    sims = []
    for isbn in isbns:
        vec = embedding_dict.get(isbn)
        if vec is None:
            continue
        sim = float(
            np.dot(query_embedding, vec)
            / (np.linalg.norm(query_embedding) * np.linalg.norm(vec))
        )
        sims.append((isbn, sim))

    # 5) 정렬
    sims.sort(key=lambda x: x[1], reverse=True)

    # 6) 강/약 매칭 분리
    strong = [(i, s) for i, s in sims if s >= TH_STRONG]
    weak = [(i, s) for i, s in sims if TH_WEAK <= s < TH_STRONG]

    # 7) 책 정보 빌더
    def _to_book_item(isbn: str, sim: float):
        book = get_book_by_isbn(isbn)
        if not book:
            return None
        return {
            "kind": "book",
            "isbn13": book.get("isbn13", isbn),
            "title": book.get("title", ""),
            "categoryName": book.get("categoryName", ""),
            "description": book.get("description", ""),
            "similarity": round(sim, 4),
        }

    # 8) items 구성 (중간 공지 삽입)
    items = []

    # 8-1) 강한 매칭 먼저 채우기
    for isbn, sim in strong:
        if len(items) >= TOPK:
            break
        book_item = _to_book_item(isbn, sim)
        if book_item:
            items.append(book_item)

    # 8-2) 약한 매칭을 붙이기 시작하는 순간, 경고 notice를 "강→약" 경계에 삽입
    warning_inserted = False
    if len(items) < TOPK and weak:
        # 경고 notice (중간 삽입)
        items.append(
            {
                "kind": "notice",
                "level": "warning",
                "text": "결과의 정확도가 떨어질 수 있습니다.",
            }
        )
        warning_inserted = True

        # 이제 약한 매칭으로 채움
        for isbn, sim in weak:
            if len(items) >= TOPK:
                break
            book_item = _to_book_item(isbn, sim)
            if book_item:
                items.append(book_item)

    # 8-3) 최종 개수가 부족하면 리스트 "끝"에 부족 notice 삽입
    not_enough_inserted = False
    if len([x for x in items if x.get("kind") == "book"]) < TOPK:
        items.append(
            {"kind": "notice", "level": "info", "text": "유사한 책이 부족합니다."}
        )
        not_enough_inserted = True

    # 9) (하위 호환) 기존 warning/message 필드도 넣어주기
    result = {
        "query": query,
        "query_emotion": query_emotion,
        "items": items,
    }
    if warning_inserted:
        result["warning"] = "결과의 정확도가 떨어질 수 있습니다."
    if not_enough_inserted:
        result["message"] = "유사한 책이 부족합니다."

    return result
