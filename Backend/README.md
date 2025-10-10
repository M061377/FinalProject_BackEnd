# Backend (FastAPI)

## Run
python -m venv .venv (처음 한번만)
.\.venv\Scripts\activate
pip install -r requirements.txt (처음 한번만)
cd Backend
uvicorn app.main:app --reload

http://127.0.0.1:8000/docs
