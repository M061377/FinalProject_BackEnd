import json
from transformers import (
    BartForConditionalGeneration, PreTrainedTokenizerFast
)

# 1. KoBART 모델 불러오기
kobart_model = BartForConditionalGeneration.from_pretrained("./kobart_finetuned")
kobart_tokenizer = PreTrainedTokenizerFast.from_pretrained("./kobart_finetuned")

# 2. 입력 데이터 로드
with open("../data/bestseller_books.json", encoding="utf-8") as f:
    books = json.load(f)

# 3. 요약 함수
def summarize_kobart(text):
    input_ids = kobart_tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=512)
    output = kobart_model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
    return kobart_tokenizer.decode(output[0], skip_special_tokens=True)

# 4. 요약 실행
results = []
total = len(books)

for i, book in enumerate(books):
    title = book.get("title", f"Book {i+1}")
    full_desc = book.get("fullDescription", "").strip()

    if not full_desc:
        print(f"⚠️ [{i+1}/{total}] '{title}' fullDescription 비어 있어서 건너뜀")
        continue

    print(f"🔄 요약 중... [{i+1}/{total}] '{title}'")
    kobart_sum = summarize_kobart(full_desc)

    results.append({
        "title": title,
        "fullDescription": full_desc,
        "description_kobart": kobart_sum
    })

# 5. 결과 저장
output_path = "../data/summary_result.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("🎉 KoBART 요약 전체 완료 및 저장됨 →", output_path)
