from pydantic import BaseModel, Field
from typing import List, Optional


class ErrorResponse(BaseModel):
    error: dict


class RecommendItem(BaseModel):
    isbn13: str
    title: str
    author: str
    publisher: str
    cover: str
    similarity: float


class RecommendResponse(BaseModel):
    query: str
    emotion: str
    page: int
    page_size: int
    total_candidates: int
    items: list[RecommendItem]
