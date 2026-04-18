from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from .rag import TutorEngine


BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
STORAGE_DIR = BASE_DIR / "storage"

load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)

app = FastAPI(title="RAG Tutor", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

engine = TutorEngine(base_dir=BASE_DIR)


@app.get("/")
async def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.api_route("/health", methods=["GET", "HEAD"])
async def health() -> dict:
    return {"status": "ok", "groqConfigured": bool(os.getenv("GROQ_API_KEY"))}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)) -> dict:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    topic = await engine.ingest_pdf(file)
    return topic


@app.post("/chat")
async def chat(payload: str = Form(...)) -> dict:
    data = json.loads(payload)
    topic_id = data.get("topicId")
    message = data.get("message", "").strip()
    history = data.get("history", [])
    if not topic_id or not message:
        raise HTTPException(status_code=400, detail="topicId and message are required.")
    return await engine.answer_question(topic_id=topic_id, message=message, history=history)


@app.get("/images/{topic_id}")
async def images(topic_id: str) -> dict:
    return engine.get_images(topic_id)


@app.get("/topics/{topic_id}")
async def topic(topic_id: str) -> dict:
    return engine.get_topic(topic_id)
