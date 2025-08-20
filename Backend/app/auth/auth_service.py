# app/auth/auth_service.py (전체 내용)

import firebase_admin
from firebase_admin import auth, exceptions, firestore
from firebase_admin.auth import UserNotFoundError
from fastapi import HTTPException
import httpx

# 카카오 REST API 키는 config 모듈에서 가져옵니다.
from app.core.config import KAKAO_REST_API_KEY
from app.core.schemas import FirebaseTokenResponse

# Firestore 클라이언트 초기화
db = firestore.client()

async def get_kakao_user_info(access_token: str):
    """
    카카오 Access Token으로 사용자 정보를 가져옵니다.
    """
    url = "https://kapi.kakao.com/v2/user/me"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-type": "application/x-www-form-urlencoded;charset=utf-8",
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=400,
                detail=f"카카오 토큰 검증 실패: {e.response.text}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"카카오 사용자 정보 조회 중 오류 발생: {e}"
            )


async def verify_kakao_token_and_get_firebase_custom_token(kakao_access_token: str):
    """
    카카오 Access Token을 검증하고, Firebase Custom Token을 생성합니다.
    """
    try:
        user_info = await get_kakao_user_info(access_token=kakao_access_token)
        
        print("카카오 사용자 정보:", user_info)
        
        kakao_uid = str(user_info.get("id"))
        
        profile = user_info.get("kakao_account", {}).get("profile", {})
        display_name = profile.get("nickname", "카카오 사용자")
        photo_url = profile.get("profile_image_url")
        
        print(f"카카오 UID: {kakao_uid}")

        is_new_user = False
        try:
            user_record = auth.get_user(kakao_uid)
            print(f"이미 존재하는 사용자입니다. UID: {user_record.uid}")
        except UserNotFoundError:
            print(f"새로운 사용자입니다. UID: {kakao_uid}")
            is_new_user = True
            user_record = auth.create_user(
                uid=kakao_uid,
                display_name=display_name,
                photo_url=photo_url,
            )

        # Firebase Authentication에는 사용자가 있지만 Firestore에는 문서가 없을 수 있으므로,
        # Firestore 문서가 존재하는지 먼저 확인합니다.
        user_ref = db.collection('users').document(kakao_uid)
        user_doc = user_ref.get()

        user_data = {
            'userID': kakao_uid,
            'userNickname': display_name,
            'photo_url': photo_url,
            'provider': 'kakao',
            'updated_at': firestore.SERVER_TIMESTAMP
        }

        if not user_doc.exists:
            # Firestore 문서가 없으면 새로 생성합니다.
            user_data['created_at'] = firestore.SERVER_TIMESTAMP
            user_ref.set(user_data)
            print("새로운 사용자 문서가 Firestore에 성공적으로 생성되었습니다.")
        else:
            # Firestore 문서가 있으면 업데이트합니다.
            user_ref.update(user_data)
            print("기존 사용자 문서가 Firestore에 성공적으로 업데이트되었습니다.")

        firebase_custom_token = auth.create_custom_token(kakao_uid)
        
        decoded_token = firebase_custom_token.decode('utf-8')

        return {
            "firebase_token": decoded_token,
            "uid": kakao_uid,
            "display_name": display_name
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"파이어베이스 토큰 생성 중 오류 발생: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"파이어베이스 토큰 생성 중 오류 발생: {e}"
        )