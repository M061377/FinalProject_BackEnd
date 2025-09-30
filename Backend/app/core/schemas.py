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


# 회원가입 요청 시 필요한 데이터
class UserSignup(BaseModel):
    userEmail: str
    userPW: str
    userNickname: str  # 닉네임은 Firestore에 추가로 저장합니다.


# 로그인 요청 시 필요한 데이터
class UserLogin(BaseModel):
    userEmail: str
    userPW: str


class FirebaseTokenResponse(BaseModel):
    """
    파이어베이스 토큰, UID, 그리고 사용자 이름을 포함한 응답 모델입니다.
    """

    firebase_token: str
    uid: str
    display_name: str


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
