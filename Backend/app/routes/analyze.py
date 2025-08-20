# app/routes/analyze.py

from fastapi import APIRouter, HTTPException
from app.core.schemas import AnalyzeIn, AnalyzeOut
from app.services.pipeline import analyze_pipeline # pipeline.py import

router = APIRouter(tags=["analyze"])

@router.post("/analyze", response_model=AnalyzeOut)
async def analyze(payload: AnalyzeIn):
    try:
        # 클라이언트로부터 받은 payload를 pipeline 함수에 전달
        result = analyze_pipeline(payload.fullDescription)
        
        # pipeline에서 반환된 딕셔너리를 AnalyzeOut 스키마에 맞게 변환
        return AnalyzeOut(
            title=payload.title,
            summary=result["summary"],
            emotion=result["emotion"],
            confidence=result["confidence"],
            model=result["model"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))