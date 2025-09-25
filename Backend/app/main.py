# app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

# FastAPI 앱 생성
app = FastAPI(title="Backend API", version="1.0.0")


# 🔑 Swagger에 JWT 보안 스키마 추가
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        routes=app.routes,
    )
    openapi_schema["components"]["securitySchemes"] = {
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        }
    }
    for path in openapi_schema["paths"].values():
        for method in path.values():
            method.setdefault("security", [{"bearerAuth": []}])
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

# CORS 미들웨어
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
from app.auth.auth import router as auth_router
from app.routes.analyze import router as analyze_router
from app.routes.recommend_router import router as recommend_cached_router

app.include_router(auth_router, prefix="/api/auth", tags=["auth"])
app.include_router(analyze_router, prefix="/v1", tags=["analyze"])
app.include_router(recommend_cached_router, prefix="/v1", tags=["recommend_router"])
