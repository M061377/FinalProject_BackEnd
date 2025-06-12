import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. 경로 설정
model_path = "./kobert-finetuned-emotion-v4"
data_path = "../data/bestseller_books.json"
output_path = "../data/result2.json"

# 2. 모델 및 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# 3. 감정 라벨 매핑
label_mapping = {
    0: 'Anger',
    1: 'Disgust',
    2: 'Fear',
    3: 'Joy',
    4: 'Neutral',
    5: 'Sadness',
    6: 'Surprise'
}

# 4. JSON 데이터 로드 및 감정 예측
results = []
with open(data_path, "r", encoding="utf-8") as f:
    books = json.load(f)

for book in books:
    title = book.get("title", "(제목 없음)")
    description = book.get("description", "")

    inputs = tokenizer(description, return_tensors="pt", padding=True, truncation=True, max_length=64)
    if 'token_type_ids' in inputs:
        inputs.pop('token_type_ids')

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1).squeeze()
        predicted_id = logits.argmax(dim=-1).item()

    result = {
        "title": title,
        "description": description,
        "predicted_emotion": label_mapping[predicted_id],
        "emotion_probs": {label_mapping[i]: round(float(prob), 4) for i, prob in enumerate(probs)}
    }
    results.append(result)

# 5. 결과 저장
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"✅ 감정 예측 결과를 {output_path}에 저장했습니다.")