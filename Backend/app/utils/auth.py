from fastapi import Header, HTTPException
from firebase_admin import auth as fb_auth


async def verify_id_token(
    authorization: str | None = Header(None, alias="Authorization")
):
    """
    Authorization: Bearer <ID_TOKEN>
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401, detail="Invalid auth header format. Use 'Bearer <token>'"
        )

    token = authorization[len("Bearer ") :].strip()
    try:
        decoded = fb_auth.verify_id_token(token)
        return decoded["uid"]
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid ID token: {e}")
