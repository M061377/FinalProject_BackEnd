import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 모델과 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("./kobert-finetuned-emotion")
model = AutoModelForSequenceClassification.from_pretrained("./kobert-finetuned-emotion")
model.eval()

# 감정 라벨 매핑
label_mapping = {
    0: 'Anger',
    1: 'Disgust',
    2: 'Fear',
    3: 'Joy',
    4: 'Neutral',
    5: 'Sadness',
    6: 'Surprise'
}


# 문장 입력
sentence = "나는 어제 집에 가서 밥을 먹었다"
inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=64)

# ★ token_type_ids 제거 (중요)
if 'token_type_ids' in inputs:
    inputs.pop('token_type_ids')

# 모델에 입력
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax(dim=-1).item()

# 결과 출력
print(f"입력 문장: {sentence}")
print(f"예측 감정: {label_mapping[predicted_class_id]}")

# 1개 문장 예측 후
print("logits:", logits)
print("softmax probs:", torch.nn.functional.softmax(logits, dim=-1))
