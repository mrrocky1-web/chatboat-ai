"""FastAPI service exposing the support chatbot.

Run:  uvicorn app:app --reload
"""
import logging
import os
import secrets
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "database"))
import database as db
from nlp_engine import ChatEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("chatbot.app")

BASE = Path(__file__).parent
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "change-me")
state = {}


@asynccontextmanager
async def lifespan(app):
    db.init_db()
    state["engine"] = ChatEngine(BASE / "faqs.json")
    if ADMIN_TOKEN == "change-me":
        log.warning("ADMIN_TOKEN is the default value - set it before deploying!")
    yield


app = FastAPI(title="Support Chatbot", version="1.0.0", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    session_id: str | None = Field(None, max_length=64)


class ChatResponse(BaseModel):
    session_id: str
    message_id: int
    reply: str
    intent: str
    confidence: float
    matched_question: str | None = None
    suggestions: list[str] = []
    fallback: bool


class FeedbackRequest(BaseModel):
    message_id: int
    helpful: bool
    comment: str | None = Field(None, max_length=500)


def require_admin(token):
    if not token or not secrets.compare_digest(token, ADMIN_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid or missing X-Admin-Token")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(BASE.parent / "frontend" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "backend": state["engine"].backend}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    text = req.message.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Message must not be blank")
    session_id = req.session_id or uuid.uuid4().hex
    result = state["engine"].respond(text, last_turn=db.get_last_turn(session_id))
    message_id = db.log_message(session_id, text, result)
    return ChatResponse(
        session_id=session_id,
        message_id=message_id,
        reply=result["reply"],
        intent=result["intent"],
        confidence=result["confidence"],
        matched_question=result["matched_question"],
        suggestions=result["suggestions"],
        fallback=result["fallback"],
    )


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    if not db.message_exists(req.message_id):
        raise HTTPException(status_code=404, detail="Unknown message_id")
    db.add_feedback(req.message_id, req.helpful, req.comment)
    return {"status": "recorded"}


@app.get("/history/{session_id}")
def history(session_id: str, limit: int = Query(50, ge=1, le=200)):
    return db.get_history(session_id, limit)


@app.get("/admin/logs")
def admin_logs(limit: int = Query(100, ge=1, le=1000), only_fallback: bool = False,
               x_admin_token: str | None = Header(None)):
    require_admin(x_admin_token)
    return db.recent_logs(limit, only_fallback)


@app.get("/admin/stats")
def admin_stats(x_admin_token: str | None = Header(None)):
    require_admin(x_admin_token)
    return db.stats()
