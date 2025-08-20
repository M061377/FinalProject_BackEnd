# app/utils/emotion_cache.py
import json
from typing import List, Dict
from app.core.config import EMOTION_ISBN_JSON

_cache = None


def load_emotion_isbns() -> Dict[str, List[str]]:
    global _cache
    if _cache is None:
        with open(EMOTION_ISBN_JSON, "r", encoding="utf-8") as f:
            _cache = json.load(f)  # {"감동":[isbn13,...], "불안":[...], ...}
    return _cache


def get_isbns_for_emotion(emotion: str) -> List[str]:
    data = load_emotion_isbns()
    return data.get(emotion, [])
