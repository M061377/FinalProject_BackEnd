# app/services/recommend_cached.py
import numpy as np
from app.llm.sbert import encode
from app.llm.kluebert import classify
from app.utils.embeddings import load_embeddings
from app.utils.emotion_cache import get_isbns_for_emotion
from app.utils.firebase_util import get_book_by_isbn


async def get_recommendations(query: str):
    # 1. 쿼리 감정 분류
    query_emotion, confidence = classify(query)
    query_embedding = encode(query)

    # 2. emotion_isbn.json에서 ISBN 목록 가져오기
    isbns = get_isbns_for_emotion(query_emotion)
    if not isbns:
        return {
            "query": query,
            "query_emotion": query_emotion,
            "items": [],
            "message": "해당 감정에 맞는 책이 없습니다.",
        }

    # 3. 임베딩 로드
    embedding_dict = load_embeddings()

    # 4. 유사도 계산
    similarities = []
    for isbn in isbns:
        if isbn not in embedding_dict:
            continue
        book_vec = embedding_dict[isbn]
        sim = float(
            np.dot(query_embedding, book_vec)
            / (np.linalg.norm(query_embedding) * np.linalg.norm(book_vec))
        )
        similarities.append((isbn, sim))

    # 5. 유사도 정렬
    similarities.sort(key=lambda x: x[1], reverse=True)

    # 6. 필터링 로직
    strong_matches = [(i, s) for i, s in similarities if s >= 0.6]
    all_matches = strong_matches.copy()

    warning = None
    message = None

    if len(strong_matches) < 10:
        weak_matches = [(i, s) for i, s in similarities if 0.4 <= s < 0.6]
        if weak_matches:  # 유사도 0.6 미만 책을 포함하기 시작할 때만 warning
            warning = "결과의 정확도가 떨어질 수 있습니다."
        all_matches.extend(weak_matches)

    if len(all_matches) < 10:
        message = "유사한 책이 부족합니다."

    # 7. 최종 상위 10권 ISBN
    final = all_matches[:10]

    # 8. Firebase에서 책 정보 가져오기
    items = []
    for isbn, sim in final:
        book = get_book_by_isbn(isbn)
        if not book:
            continue
        items.append(
            {
                "isbn13": book.get("isbn13", isbn),
                "title": book.get("title", ""),
                "categoryName": book.get("categoryName", ""),
                "description": book.get("description", ""),
                "similarity": round(sim, 4),
            }
        )

    # 9. 최종 응답 dict를 순서대로 구성
    result = {"query": query, "query_emotion": query_emotion, "items": items}
    if warning:
        result["warning"] = warning
    if message:
        result["message"] = message  # 항상 마지막에 들어가도록 순서 제어

    return result
