python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn api_backend:app --host 0.0.0.0 --port 8000