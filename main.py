# ============================================================
# Project 1a: Crypto Risk Engine — Series 1 Foundation Service
# main.py -- Thin entrypoint: launches single uvicorn/FastAPI service
#
# All startup logic (DB init, seeding, model loading, scheduler)
# lives in api_backend.py lifespan so Railway / any WSGI runner
# works without this file.
#
# Run locally:  python main.py
# Run on Railway / prod:  uvicorn api_backend:app --host 0.0.0.0 --port $PORT
# ============================================================

import uvicorn
from config import PORT

if __name__ == "__main__":
    uvicorn.run(
        "api_backend:app",
        host      = "0.0.0.0",
        port      = PORT,
        reload    = False,
        log_level = "info",
    )