import firebase_admin
from firebase_admin import credentials, firestore

# 서비스 계정 키 경로
cred = credentials.Certificate(r"C:\Users\KHJ\C135124\GP\DB\firebase_key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# 사용자 정보
user_data = {
    "userPW": "test_password",
    "userEmail": "test@email.com",
    "userNickname": "test_nickname",
}

# userID를 문서 ID로 사용
user_id = "test_id"

# user 컬렉션에 userID 문서 생성
db.collection("users").document(user_id).set(user_data)

print("사용자 등록 완료")
