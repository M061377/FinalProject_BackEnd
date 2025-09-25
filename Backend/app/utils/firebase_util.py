# app/utils/firebase_util.py
from typing import Optional, Dict, List
from app.core.config import FIREBASE_CRED_JSON
import firebase_admin
from firebase_admin import credentials, firestore

_db = None


def get_db():
    global _db
    if _db is not None:
        return _db
    try:
        app = firebase_admin.get_app()  # 이미 초기화된 앱 있으면 재사용
    except ValueError:
        cred = credentials.Certificate(str(FIREBASE_CRED_JSON))
        app = firebase_admin.initialize_app(cred)
    _db = firestore.client(app)
    return _db


def get_book_by_isbn(isbn13: str) -> Optional[Dict]:
    """
    (기존) 단일 도서의 기본 필드만 반환합니다.
    기존 로직을 유지하기 위해 이 함수는 그대로 둡니다.
    """
    db = get_db()
    doc = db.collection("books").document(isbn13).get()
    if doc.exists:
        d = doc.to_dict()
    else:
        q = list(db.collection("books").where("isbn13", "==", isbn13).limit(1).stream())
        if not q:
            return None
        d = q[0].to_dict()
    return {
        "isbn13": d.get("isbn13", isbn13),
        "title": d.get("title", ""),
        "description": d.get("description", ""),
        "emotion": d.get("emotion", ""),
    }


# =========================
# ✅ (추가) 추천 목록용: 최소 필드 배치 조회
# =========================
def get_books_min_fields(isbn_list: List[str]) -> Dict[str, Dict]:
    """
    추천 '목록' 화면에 필요한 최소 필드만 반환합니다.
    반환 형태: {isbn13: {title, author, publisher, cover}}
    """
    result: Dict[str, Dict] = {}
    if not isbn_list:
        return result

    db = get_db()

    # 1) 문서 ID 직접 조회 (10개씩)
    for i in range(0, len(isbn_list), 10):
        part = isbn_list[i : i + 10]
        docs = [db.collection("books").document(x).get() for x in part]
        for d in docs:
            if not d.exists:
                continue
            data = d.to_dict() or {}
            isbn = data.get("isbn13") or d.id
            result[isbn] = {
                "title": data.get("title", ""),
                "author": data.get("author", ""),  # 문자열
                "publisher": data.get("publisher", ""),
                "cover": data.get("cover", ""),  # Firestore에 저장된 cover
            }

    # 2) isbn13 필드 기반 조회 (where in, 10개 제한)
    for i in range(0, len(isbn_list), 10):
        part = isbn_list[i : i + 10]
        q = list(db.collection("books").where("isbn13", "in", part).stream())
        for d in q:
            data = d.to_dict() or {}
            isbn = data.get("isbn13")
            if not isbn:
                continue
            if isbn in result:
                continue
            result[isbn] = {
                "title": data.get("title", ""),
                "author": data.get("author", ""),
                "publisher": data.get("publisher", ""),
                "cover": data.get("cover", ""),
            }

    return result


# === REMOVABLE: BOOK DETAIL API (BEGIN) ===
# =========================
# ✅ (추가) 상세: 문서 전체 반환
# =========================
def get_book_detail_all(isbn13: str) -> Optional[Dict]:
    """
    상세 페이지용: Firestore에 저장된 모든 필드를 그대로 반환합니다.
    """
    db = get_db()

    # 1) 문서 ID가 isbn13인 경우
    doc = db.collection("books").document(isbn13).get()
    if doc.exists:
        return doc.to_dict()

    # 2) isbn13 필드로 저장된 경우
    q = list(db.collection("books").where("isbn13", "==", isbn13).limit(1).stream())
    if not q:
        return None
    return q[0].to_dict()


# === REMOVABLE: BOOK DETAIL API (END) ===
