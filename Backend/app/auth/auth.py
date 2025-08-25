# app/auth/auth.py

from fastapi import APIRouter, HTTPException, Depends, Security
from pydantic import BaseModel
from app.auth.auth_service import verify_kakao_token_and_get_firebase_custom_token
from app.auth.auth_service import kakao_unlink_by_admin_key, hard_delete_user_everywhere
from app.utils.auth import verify_id_token
from app.core.schemas import FirebaseTokenResponse

router = APIRouter()


class KakaoToken(BaseModel):
    access_token: str


@router.post(
    "/kakao_login",
    response_model=FirebaseTokenResponse,
    summary="카카오 Access Token을 이용한 로그인 또는 회원가입",
    description="유효한 카카오 Access Token을 전달하면 Firebase Custom Token을 반환합니다. 신규 사용자일 경우 Firebase 사용자 계정을 생성합니다.",
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


@router.delete(
    "/me",
    summary="카카오 회원탈퇴 (unlink + Firebase/Firestore 삭제)",
    description="ID Token 인증 후 카카오 unlink → Firestore 사용자 문서 삭제 → Firebase Auth 사용자 삭제를 수행합니다.",
)
async def delete_me(
    current_uid: str = Security(verify_id_token),  # 👈 Depends 대신 Security로
):
    unlink_result = await kakao_unlink_by_admin_key(current_uid)
    await hard_delete_user_everywhere(current_uid)
    return {"ok": True, "unlinked": unlink_result}
