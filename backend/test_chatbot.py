import os
import tempfile

import pytest

_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ["CHATBOT_DB"] = _tmp.name
os.environ["CHATBOT_BACKEND"] = "tfidf"
os.environ["ADMIN_TOKEN"] = "test-token"

from fastapi.testclient import TestClient  # noqa: E402

import app as app_module  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app_module.app) as c:
        yield c


def chat(client, msg, sid=None):
    r = client.post("/chat", json={"message": msg, "session_id": sid})
    assert r.status_code == 200, r.text
    return r.json()


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


@pytest.mark.parametrize("msg,expected", [
    ("where is my package?", "orders-track"),
    ("I want to cancel my purchase", "orders-cancel"),
    ("forgot my password", "account-reset-password"),
    ("do you accept UPI payments", "payments-methods"),
    ("is delivery free?", "shipping-cost"),
    ("my item stopped working, any warranty?", "warranty"),
])
def test_faq_matching(client, msg, expected):
    out = chat(client, msg)
    assert out["intent"] == "faq"
    assert out["matched_question"] == app_module.state["engine"].by_id[expected]["question"]


def test_smalltalk(client):
    assert chat(client, "hello there")["intent"] == "greeting"
    assert chat(client, "thanks a lot")["intent"] == "thanks"
    assert chat(client, "let me talk to a human")["intent"] == "handoff"


def test_fallback_on_nonsense(client):
    out = chat(client, "what is the airspeed velocity of an unladen swallow")
    assert out["fallback"] is True


def test_context_followup(client):
    first = chat(client, "How do I start a return?")
    sid = first["session_id"]
    second = chat(client, "how long does it take to get the money?", sid)
    assert second["intent"] == "faq"
    assert second["matched_question"] == "How long does a refund take?"


def test_session_isolation(client):
    a = chat(client, "How do I start a return?")
    b = chat(client, "how long does it take?")  # new session, no context
    assert b["session_id"] != a["session_id"]


def test_feedback_and_history(client):
    out = chat(client, "track my order")
    sid = out["session_id"]
    assert client.post("/feedback", json={"message_id": out["message_id"], "helpful": True}).status_code == 200
    assert client.post("/feedback", json={"message_id": 999999, "helpful": True}).status_code == 404
    hist = client.get(f"/history/{sid}").json()
    assert len(hist) == 1 and hist[0]["user_message"] == "track my order"


def test_validation(client):
    assert client.post("/chat", json={"message": ""}).status_code == 422
    assert client.post("/chat", json={"message": "   "}).status_code == 422
    assert client.post("/chat", json={"message": "x" * 1001}).status_code == 422


def test_admin_auth_and_stats(client):
    assert client.get("/admin/stats").status_code == 401
    assert client.get("/admin/logs", headers={"X-Admin-Token": "nope"}).status_code == 401
    h = {"X-Admin-Token": "test-token"}
    stats = client.get("/admin/stats", headers=h).json()
    assert stats["total_messages"] > 0 and stats["feedback_total"] >= 1
    fb = client.get("/admin/logs?only_fallback=true", headers=h).json()
    assert fb and all(r["is_fallback"] == 1 for r in fb)
