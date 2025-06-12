from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast, Trainer, TrainingArguments
from datasets import Dataset
import json

print("📦 라이브러리 로딩 완료")

# Load dataset
def load_dataset(path):
    print(f"📂 데이터셋 로딩 중: {path}")
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    print(f"✅ 데이터셋 로딩 완료: {len(data)}개 샘플")
    return Dataset.from_list(data)

def preprocess_function_kobart(example):
    model_input = tokenizer(example["input"], padding="max_length", truncation=True, max_length=512)
    label = tokenizer(example["output"], padding="max_length", truncation=True, max_length=128)
    model_input["labels"] = label["input_ids"]
    return model_input

# ✅ 기존 학습된 모델에서 이어서 시작
model_name = "./kobart_finetuned"  # 기존 fine-tuned 모델 경로
print(f"🔍 기존 finetuned 모델 로딩 중: {model_name}")
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)
print("✅ 기존 모델 및 토크나이저 로딩 완료")

# 📂 데이터 로드 및 전처리
dataset = load_dataset("../data/dataset_kobart.json")
print("🔁 전처리 중...")
tokenized_dataset = dataset.map(preprocess_function_kobart, remove_columns=dataset.column_names)
print("✅ 전처리 완료")

# 🧾 학습 설정
output_dir = "./kobart_finetuned"  # 결과 저장 디렉터리
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    logging_steps=50,
    evaluation_strategy="no",
    report_to="none"
)

# 🚀 Trainer 초기화
print("🚀 Trainer 준비 중...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

# 🏋️ 학습 시작
print("🏋️ 학습 시작...")
trainer.train()

# 💾 결과 저장
print("💾 모델 저장 중...")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("🎉 ✅ KoBART 누적 학습 완료 및 저장됨")
