# farmlingua_backend/app/main.py
import os
import sys
import logging
import uuid
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.tasks.rag_updater import schedule_updates
from app.utils import config
from app.agents.crew_pipeline import run_pipeline

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)

app = FastAPI(
    title="farmlingua AI Backend",
    description="Backend service for FARMLINGUA AI with RAG updates, multilingual support, and expert AI pipeline",
    version="1.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=getattr(config, "ALLOWED_ORIGINS", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    logging.info("Starting farmlingua AI backend...")
    schedule_updates()

@app.get("/")
def home():
    """Health check endpoint."""
    return {
        "status": "Farmlingua AI backend running",
        "version": "1.2.0",
        "vectorstore_path": config.VECTORSTORE_PATH
    }

@app.post("/ask")
def ask_farmbot(
    query: str = Body(..., embed=True),
    session_id: str = Body(None, embed=True)
):
    """
    Ask farmlingua AI a farming-related question.
    - Supports Hausa, Igbo, Yoruba, Swahili, Amharic, and English.
    - Automatically detects user language, translates if needed,
      and returns response in the same language.
    - Maintains separate conversation memory per session_id.
    """
    if not session_id:
        session_id = str(uuid.uuid4())  # assign new session if missing

    logging.info(f"Received query: {query} [session_id={session_id}]")
    answer_data = run_pipeline(query, session_id=session_id)

    detected_lang = answer_data.get("detected_language", "Unknown")
    logging.info(f"Detected language: {detected_lang}")

    return {
        "query": query,
        "answer": answer_data.get("answer"),
        "session_id": answer_data.get("session_id"),
        "detected_language": detected_lang
    }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=getattr(config, "PORT", 7860),
        reload=bool(getattr(config, "DEBUG", False))
    )
