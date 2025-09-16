# app/routes/recommend_cached.py
from fastapi import APIRouter
from pydantic import BaseModel
from app.services import recommend_cached

router = APIRouter()


# ✅ 입력 스키마: query만 받음
class RecommendRequest(BaseModel):
    query: str


@router.post("/recommend_cached")  # ✅ 중복 제거
async def recommend_books(request: RecommendRequest):
    return await recommend_cached.get_recommendations(request.query)
