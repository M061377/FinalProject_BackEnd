import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 1. 모델 로딩
model = SentenceTransformer("jhgan/ko-sroberta-multitask")

# 2. JSON 파일 로딩
with open("for_embedding_test.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 3. 제목과 설명 분리
titles = [book["title"] for book in data["books"]]
descriptions = [book["description"] for book in data["books"]]

# 4. 고정된 입력 문장
query = "과거를 배경으로 한 약간 무서운 내용의 한국 추리 소설 추천해줘"

# 5. 임베딩
desc_embeddings = model.encode(descriptions)
query_embedding = model.encode([query])

# 6. 유사도 계산
cos_sim = cosine_similarity(query_embedding, desc_embeddings)[0]

# 7. 상위 3개 인덱스 추출
top3_idx = np.argsort(cos_sim)[::-1][:3]

# 8. 결과 출력
print("\n입력 문장:", query)
print("\n📚 유사한 책 Top 3:")
for rank, idx in enumerate(top3_idx, 1):
    print(f"{rank}. {titles[idx]} (유사도: {cos_sim[idx]:.4f})")
