import json
from transformers import T5ForConditionalGeneration, AutoTokenizer

# 모델 로딩
model = T5ForConditionalGeneration.from_pretrained("./t5_finetuned")
tokenizer = AutoTokenizer.from_pretrained("./t5_finetuned")

# 입력 데이터 로드
with open("../data/bestseller_books.json", encoding="utf-8") as f:
    books = json.load(f)

# 요약 함수
def summarize_t5(text):
    input_text = "summarize: " + text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512)
    output = model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# 요약 실행
results = []
for book in books:
    summary = summarize_t5(book["description"])
    results.append({
        "title": book["title"],
        "description": book["description"],
        "description_t5": summary
    })

# 결과 저장
with open("../data/t5_result.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("✅ T5 요약 완료 및 저장됨")
