# AI-Powered Support Chatbot

An FAQ / customer-support chatbot built with **Python, NLTK, scikit-learn, optional Hugging Face Transformers, FastAPI and SQLite**. It answers from a JSON knowledge base, remembers the topic within a session, and logs every exchange for analysis.

## How it works

1. **Preprocess** – lower-case, tokenise, drop stopwords, lemmatise (NLTK; falls back to a bundled stopword list and a simple stemmer when NLTK data can't be downloaded).
2. **Small-talk / hand-off** – greetings, thanks, goodbye and "talk to a human" are caught by regex intents.
3. **Retrieve** – each FAQ is indexed under its question plus paraphrases. Scoring backends (`CHATBOT_BACKEND`):
   - `tfidf` – TF-IDF (uni+bigrams) cosine similarity, no downloads needed
   - `transformer` – sentence embeddings (`all-MiniLM-L6-v2`, mean-pooled) via `transformers`
   - `hybrid` – 0.65 semantic + 0.35 lexical
   - `auto` (default) – hybrid if the model loads, otherwise TF-IDF
4. **Context** – for follow-ups ("how long does *it* take?") the previous answer's category gets a score boost, so the bot stays on topic. Context is stored per `session_id` in SQLite.
5. **Fallback** – below the confidence threshold the bot says it didn't understand, offers related questions, and logs the message as a fallback so you can find gaps in the FAQ.

## Project structure

```
chatbot-project/
├── frontend/   index.html            chat UI (HTML/CSS/JS)
├── backend/    app.py                FastAPI server + API routes
│               nlp_engine.py         NLTK preprocessing + TF-IDF/Transformer matching
│               faqs.json             FAQ knowledge base
│               test_chatbot.py       tests
├── database/   database.py           SQLite code (logs, sessions, feedback)
│               chatbot.db            SQLite database file
├── requirements.txt
└── README.md
```

## Run it

```bash
pip install -r requirements.txt
# optional semantic matching:
# pip install torch transformers

export ADMIN_TOKEN='a-long-random-string'   # protects /admin/*
cd backend
uvicorn app:app --reload
```

Open http://127.0.0.1:8000 for the chat UI, or http://127.0.0.1:8000/docs for the API.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/chat` | `{"message": "...", "session_id": "optional"}` → reply, confidence, suggestions, `message_id` |
| POST | `/feedback` | `{"message_id": 1, "helpful": true}` |
| GET | `/history/{session_id}` | conversation so far |
| GET | `/admin/logs?only_fallback=true` | interaction logs (header `X-Admin-Token`) |
| GET | `/admin/stats` | totals, fallback rate, top FAQs, feedback (header `X-Admin-Token`) |
| GET | `/health` | status and active backend |

## Configuration

| Env var | Default | Meaning |
|---|---|---|
| `CHATBOT_BACKEND` | `auto` | `tfidf`, `transformer`, `hybrid`, `auto` |
| `CHATBOT_THRESHOLD` | 0.30 (tfidf) / 0.45 (semantic) | minimum score to answer |
| `CHATBOT_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Hugging Face encoder |
| `CHATBOT_DB` | `database/chatbot.db` | SQLite path |
| `ADMIN_TOKEN` | `change-me` | admin endpoints token — **change before deploying** |

## Customise

Edit `backend/faqs.json` (`id`, `category`, `question`, `variations`, `answer`) and restart. Add more `variations` for questions users phrase in many ways; review `/admin/logs?only_fallback=true` regularly to find what's missing.

## Tests

```bash
cd backend
pytest -q
```

## Deploying

```bash
cd backend
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 2
```
Put it behind HTTPS (nginx/Caddy), set a strong `ADMIN_TOKEN`, and back up `database/chatbot.db`. A Dockerfile is straightforward: `python:3.12-slim`, `pip install -r requirements.txt`, run the command above.

