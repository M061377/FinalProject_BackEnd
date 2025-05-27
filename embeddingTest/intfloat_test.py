from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import json
import numpy as np

# 모델 로딩
tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base")
model = AutoModel.from_pretrained("intfloat/multilingual-e5-base")

# 책 데이터 로드
with open("for_embedding_test.json", "r", encoding="utf-8") as f:
    data = json.load(f)

titles = [book["title"] for book in data["books"]]
descriptions = [book["description"] for book in data["books"]]

# 쿼리 입력 (고정)
query = "과거를 배경으로 한 약간 무서운 내용의 한국 추리 소설 추천해줘"
query_input = "query: " + query
desc_inputs = ["passage: " + d for d in descriptions]

# 전체 임베딩 계산
inputs = tokenizer(
    [query_input] + desc_inputs, padding=True, truncation=True, return_tensors="pt"
)

with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)

query_emb = embeddings[0].unsqueeze(0)
desc_embs = embeddings[1:]

# 유사도 계산
scores = cosine_similarity(query_emb, desc_embs)[0]
top3_idx = np.argsort(scores)[::-1][:3]

# 결과 출력
print(f"\n입력 문장: {query}")
print("\n📚 유사한 책 Top 3:")
for i, idx in enumerate(top3_idx, 1):
    print(f"{i}. {titles[idx]} (유사도: {scores[idx]:.4f})")
