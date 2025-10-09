from pydantic import BaseModel, Field
from typing import List, Optional


class AnalyzeIn(BaseModel):
    title: str = Field(..., description="책 제목")
    fullDescription: str = Field(..., description="책 설명 전문")


class AnalyzeOut(BaseModel):
    title: str
    summary: str
    emotion: str
    confidence: float
    model: str


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
