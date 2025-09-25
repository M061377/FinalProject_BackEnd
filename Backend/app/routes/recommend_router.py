# app/routes/recommend_router.py
from fastapi import APIRouter, Query, HTTPException
from app.services import recommend_service
from app.core.schemas import (
    RecommendResponse,
    BookDetail,  # === REMOVABLE: BOOK DETAIL API
)  # ← 기존 schemas.py에 "추가"할 추천 스키마 사용

router = APIRouter()


@router.get("/recommend", response_model=RecommendResponse)
async def recommend_books(
    q: str = Query(..., description="사용자 입력 문장"),
    page: int = Query(1, ge=1, description="페이지 번호 (1부터 시작)"),
    page_size: int = Query(10, ge=1, le=10, description="페이지 크기 (최대 10)"),
):
    """
    사용자 입력 문장을 기반으로 책을 추천합니다.
    - 코사인 유사도 내림차순
    - 상위 100권 제한 후 페이지네이션
    """
    try:
        return await recommend_service.get_recommendations_paginated(
            query=q, page=page, page_size=page_size
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === REMOVABLE: BOOK DETAIL API (BEGIN) ===
@router.get("/books/{isbn13}", response_model=BookDetail)
async def get_book_detail(isbn13: str):
    """
    Firestore에 저장된 특정 ISBN13 책의 모든 정보를 반환합니다.
    """
    book = recommend_service.get_book_detail_all(isbn13)
    if not book:
        raise HTTPException(status_code=404, detail="해당 도서를 찾을 수 없습니다.")
    return book


# === REMOVABLE: BOOK DETAIL API (END) ===
