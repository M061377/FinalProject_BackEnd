# app/core/config.py

import os
from pathlib import Path
from dotenv import load_dotenv

# .env 파일을 로드합니다.
load_dotenv()

# 프로젝트 루트 디렉토리 설정
# __file__의 부모 디렉토리('core')의 부모 디렉토리('app')의 부모 디렉토리('Backend')
ROOT = Path(__file__).resolve().parents[2]

# 모델 파일 경로 설정
MODELS_DIR = ROOT / "models"
KLUEBERT_DIR = os.getenv("KLUEBERT_DIR", str(MODELS_DIR / "kluebert-finetuned"))
SBERT_DIR = os.getenv("SBERT_DIR", str(MODELS_DIR / "sbert"))

# Firebase 서비스 계정 파일 경로
FIREBASE_CRED_JSON = os.getenv("FIREBASE_CRED_JSON", str(ROOT.parent / "firebase_key.json"))

# 데이터 파일 경로 설정
EMBEDDING_PKL_PATH = os.getenv(
    "EMBEDDING_PKL_PATH", str(ROOT / "data" / "embedding_summary.pkl")
)
EMOTION_ISBN_JSON = os.getenv(
    "EMOTION_ISBN_JSON", str(ROOT / "data" / "emotion_isbn.json")
)

# 카카오 REST API 키
KAKAO_REST_API_KEY = os.getenv("KAKAO_REST_API_KEY")
if not KAKAO_REST_API_KEY:
    raise ValueError("KAKAO_REST_API_KEY 환경 변수가 설정되지 않았습니다.")

# 텐서플로우/토치 등에서 사용할 디바이스 설정
DEVICE = "cuda" if os.getenv("USE_CUDA", "0") == "1" else "cpu"

