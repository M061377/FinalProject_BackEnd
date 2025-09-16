# app/utils/firebase_client.py
from typing import Optional, Dict
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
