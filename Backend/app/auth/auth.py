# app/auth/auth.py

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from app.auth.auth_service import verify_kakao_token_and_get_firebase_custom_token
from app.core.schemas import FirebaseTokenResponse

router = APIRouter()

class KakaoToken(BaseModel):
    access_token: str

@router.post(
    "/kakao_login",
    response_model=FirebaseTokenResponse,
    summary="카카오 Access Token을 이용한 로그인 또는 회원가입",
    description="유효한 카카오 Access Token을 전달하면 Firebase Custom Token을 반환합니다. 신규 사용자일 경우 Firebase 사용자 계정을 생성합니다."
)
async def kakao_login(token: KakaoToken):
    """
    카카오 Access Token으로 Firebase Custom Token을 받아옵니다.
    """
    firebase_token_data = await verify_kakao_token_and_get_firebase_custom_token(
        kakao_access_token=token.access_token
    )
    # 서비스 함수에서 반환된 딕셔너리를 직접 반환합니다.
    return firebase_token_data
