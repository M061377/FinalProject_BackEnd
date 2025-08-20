"""
# app/llm/kluebert.py

import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig # 🚨 AutoConfig import
from pathlib import Path

# 현재 파일(__file__)을 기준으로 모델 폴더 경로를 정확하게 지정합니다.
MODEL_DIR = Path(__file__).resolve().parents[3] / "models" / "kluebert-finetuned"

LABELS = ["감동", "공포", "분노", "불안", "쉬움", "슬픔", "중립", "흥미"]

# 🚨 변경된 부분: AutoConfig.from_pretrained를 사용하여 config를 로드합니다.

# 1. AutoConfig.from_pretrained를 사용하여 config.json을 로드합니다.
config = AutoConfig.from_pretrained(str(MODEL_DIR))

# 2. 모델 가중치 파일(pytorch_model.bin)을 직접 로드합니다.
model_path = MODEL_DIR / "pytorch_model.bin"
state_dict = torch.load(str(model_path), map_location=torch.device('cpu'))

# 3. 모델 객체를 생성하고 가중치를 로드합니다.
model = AutoModelForSequenceClassification.from_config(config)
model.load_state_dict(state_dict)

# 4. 토크나이저도 로컬에서 로드합니다.
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)

def classify(text: str):
    # 모델을 CPU로 옮깁니다. (GPU가 없는 환경을 위해)
    model.to(torch.device('cpu'))
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=-1).item()
    return LABELS[pred_idx], probs[0][pred_idx].item()

"""

# app/llm/kluebert.py

import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
from app.core.config import KLUEBERT_DIR

# ✅ 모델 폴더 경로를 정확하게 지정합니다.
MODEL_DIR = Path(__file__).resolve().parents[3] / "models" / "kluebert-finetuned"

# ✅ 감정 라벨 리스트 (이전 답변에서 수정했던 내용입니다)
LABELS = ["감동", "공포", "분노", "불안", "쉬움", "슬픔", "중립", "흥미"] 

def classify(text: str):
    # 🚨 수정된 부분: from_pretrained로 토크나이저와 모델을 한 번에 로드합니다.
    # 이 방식은 config, 모델 가중치를 모두 자동으로 처리합니다.
    tokenizer = AutoTokenizer.from_pretrained(str(KLUEBERT_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(KLUEBERT_DIR))
    model.eval()
    
    # 모델을 CPU로 옮깁니다. (GPU가 없는 환경을 위해)
    model.to(torch.device('cpu'))
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=-1).item()
    
    return LABELS[pred_idx], probs[0][pred_idx].item()