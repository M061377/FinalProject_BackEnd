import json
import firebase_admin
from firebase_admin import credentials, firestore

# Firebase 초기화
cred = credentials.Certificate(r"C:\Users\KHJ\C135124\GP\DB\firebase_key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# JSON 파일 불러오기
with open("final_aladin.json", "r", encoding="utf-8") as f:
    books = json.load(f)

# Firestore에 한 권씩 저장
for book in books:
    if "isbn13" in book:
        doc_id = book["isbn13"]
        db.collection("books").document(doc_id).set(book)
        print(f"{doc_id} 저장 완료")
    else:
        print("isbn13 누락된 데이터 있음")

print("✅ 전체 저장 완료")
