# scripts/fetch_bestsellers.py
import requests
import time
import json
import os

TTBKEY = "ttbgjj05181954001"

# 저장 경로 설정
SAVE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "bestseller_books.json"))

# 1️⃣ 베스트셀러 100권 받아오기
itemlist_url = "https://www.aladin.co.kr/ttb/api/ItemList.aspx"
params = {
    "ttbkey": TTBKEY,
    "QueryType": "Bestseller",
    "MaxResults": 100,
    "start": 1,
    "SearchTarget": "Book",
    "output": "js",
    "Version": "20131101"
}

response = requests.get(itemlist_url, params=params)
books = response.json()["item"]

# 2️⃣ 각 책의 ISBN13으로 상세정보 조회
lookup_url = "https://www.aladin.co.kr/ttb/api/ItemLookUp.aspx"
results = []

for book in books:
    isbn13 = book["isbn13"]
    lookup_params = {
        "ttbkey": TTBKEY,
        "itemIdType": "ISBN13",
        "ItemId": isbn13,
        "output": "js",
        "Version": "20131101",
        "Cover": "Mid",
        "OptResult": "fullDescription"
    }

    try:
        res = requests.get(lookup_url, params=lookup_params)
        item = res.json()["item"][0]  # 정상 응답 시

        results.append({
    "title": item.get("title"),
    "isbn13": item.get("isbn13"),
    "author": item.get("author"),
    "description": item.get("description"),
    "image": item.get("cover"),
    "link": item.get("link"),
    "fullDescription": item.get("fullDescription")  # ✅ 필드명 대소문자 정확히!
})


        print(f"✅ {item['title']}")

    except Exception as e:
        print(f"❌ ISBN {isbn13} 에러: {e}")
    
    time.sleep(0.3)  # 너무 빠르게 요청하지 않도록 sleep

# 3️⃣ 결과 저장
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

with open(SAVE_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n🎉 총 {len(results)}권 저장 완료 → {SAVE_PATH}")
