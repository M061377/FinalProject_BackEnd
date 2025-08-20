# app/main.py

import firebase_admin
from firebase_admin import credentials
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# config.py 모듈에서 설정 변수들을 임포트합니다.
from app.core.config import FIREBASE_CRED_JSON

# Firebase 앱 초기화
try:
    # config.py에 정의된 변수를 사용하여 서비스 계정 파일 경로를 가져옵니다.
    print(f"Firebase 서비스 계정 파일 경로: {FIREBASE_CRED_JSON}") # 경로를 출력하여 확인
    cred = credentials.Certificate(FIREBASE_CRED_JSON)
    firebase_admin.initialize_app(cred)
    print("Firebase Admin SDK가 성공적으로 초기화되었습니다.")
except FileNotFoundError:
    print(f"오류: Firebase 서비스 계정 파일이 '{FIREBASE_CRED_JSON}'에 없습니다.")
    exit(1)
except Exception as e:
    print(f"Firebase 초기화 중 오류 발생: {e}")
    exit(1)

# FastAPI 앱 생성
app = FastAPI()

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 임포트 및 등록
from app.auth.auth import router as auth_router
from app.routes.analyze import router as analyze_router
from app.routes.recommend_cached import router as recommend_cached_router

app.include_router(auth_router, prefix="/api/auth", tags=["auth"])
app.include_router(analyze_router, prefix="/v1", tags=["analyze"])
app.include_router(recommend_cached_router, prefix="/v1", tags=["recommend_cached"])
