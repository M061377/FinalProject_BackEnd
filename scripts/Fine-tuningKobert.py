import transformers
print("🤖 Transformers version:", transformers.__version__)

import os
import gc
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)
from sklearn.metrics import classification_report

# 1. 데이터 로딩 및 전처리
print("✅ 데이터 로딩 중...")
df = pd.read_csv('korean_emotion_dataset.csv')
label_encoder = LabelEncoder()
df['labels'] = label_encoder.fit_transform(df['label'])

train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    df['text'].tolist(), df['labels'].tolist(),
    test_size=0.3, random_state=42, stratify=df['labels']
)

val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels,
    test_size=0.5, random_state=42, stratify=temp_labels
)

train_dataset = Dataset.from_dict({'text': train_texts, 'labels': train_labels})
val_dataset = Dataset.from_dict({'text': val_texts, 'labels': val_labels})
test_dataset = Dataset.from_dict({'text': test_texts, 'labels': test_labels})

# 2. 토크나이저
print("✅ 토크나이징 중...")
tokenizer = AutoTokenizer.from_pretrained("skt/kobert-base-v1")

def tokenize_function(examples):
    tokenized = tokenizer(
        examples['text'],
        padding="max_length",  # ← longest는 실제 에러 유발 가능, max_length가 안전
        truncation=True,
        max_length=64
    )
    if 'token_type_ids' in tokenized:
        tokenized.pop('token_type_ids')  # KoBERT 전용 처리
    return tokenized

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# 3. 모델 초기화
model_path = "./kobert-finetuned-emotion-v4"
print("🧼 모델 초기화 및 fresh 시작")
model = AutoModelForSequenceClassification.from_pretrained("skt/kobert-base-v1", num_labels=7)

# 4. 학습 설정
training_args = TrainingArguments(
    output_dir=model_path,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,  # 필요 시 쪼개서 실행
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2
)

# 5. Trainer 구성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# 6. 학습
print("🚀 1회차 학습 시작")
trainer.train()

# 7. 모델 저장
gc.collect()
torch.cuda.empty_cache()
model.save_pretrained(model_path, safe_serialization=False)
tokenizer.save_pretrained(model_path)

# 8. 성능 평가
predictions = trainer.predict(val_dataset)
preds = predictions.predictions.argmax(-1)
labels = predictions.label_ids

print(classification_report(labels, preds, target_names=label_encoder.classes_))
print("✅ 1회차 학습 및 저장 완료!")
