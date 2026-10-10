"""NLP engine: NLTK preprocessing + TF-IDF and/or Transformer sentence-embedding retrieval.

Backends (env CHATBOT_BACKEND):
  tfidf        - lexical retrieval only (fast, no model download)
  transformer  - semantic retrieval with a Hugging Face encoder
  hybrid       - weighted mix of both
  auto         - hybrid if the Transformer model loads, otherwise tfidf (default)
"""
import json
import logging
import os
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

log = logging.getLogger("chatbot.nlp")

MODEL_NAME = os.environ.get("CHATBOT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ---------------------------------------------------------------- preprocessing
# Bundled copy of NLTK's English stopword list, used when the corpus can't be downloaded
# (offline / locked-down networks) so behaviour stays identical either way.
_FALLBACK_STOPWORDS = set("""
i me my myself we our ours ourselves you your yours yourself yourselves he him his himself
she her hers herself it its itself they them their theirs themselves what which who whom
this that these those am is are was were be been being have has had having do does did doing
a an the and but if or because as until while of at by for with about against between into
through during before after above below to from up down in out on off over under again
further then once here there when where why how all any both each few more most other some
such no nor not only own same so than too very s t can will just don should now d ll m o re
ve y ain aren couldn didn doesn hadn hasn haven isn ma mightn mustn needn shan shouldn wasn
weren won wouldn please would could shall
""".split())
# Words that carry meaning in support questions even though NLTK lists them as stopwords.
_KEEP = {"not", "no", "nor", "after", "before", "free", "again"}


def _load_nltk():
    """Return (stopwords:set, lemmatizer|None). Falls back gracefully when offline."""
    stop, lemm = set(_FALLBACK_STOPWORDS), None
    try:
        import nltk
        from nltk.corpus import stopwords, wordnet
        from nltk.stem import WordNetLemmatizer

        for pkg, path in (("stopwords", "corpora/stopwords"), ("wordnet", "corpora/wordnet")):
            try:
                nltk.data.find(path)
            except LookupError:
                try:
                    nltk.download(pkg, quiet=True)
                except Exception:
                    pass
        try:
            stop = set(stopwords.words("english"))
        except LookupError:
            log.warning("NLTK stopwords unavailable; using built-in list")
        try:
            wordnet.ensure_loaded()
            lemm = WordNetLemmatizer()
        except LookupError:
            log.warning("NLTK wordnet unavailable; using suffix stemmer")
    except ImportError:
        log.warning("NLTK not installed; using built-in preprocessing")
    return stop - _KEEP, lemm


def _naive_stem(tok):
    for suf in ("ing", "ed", "es", "s"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            return tok[: -len(suf)]
    return tok


class Preprocessor:
    TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

    def __init__(self):
        self.stop, self.lemm = _load_nltk()

    def tokens(self, text):
        out = []
        for tok in self.TOKEN_RE.findall(text.lower()):
            tok = tok.split("'")[0] if "'" in tok else tok
            if tok in self.stop or len(tok) < 2 and not tok.isdigit():
                continue
            if self.lemm:
                tok = self.lemm.lemmatize(self.lemm.lemmatize(tok, "n"), "v")
            else:
                tok = _naive_stem(tok)
            out.append(tok)
        return out

    def __call__(self, text):
        return self.tokens(text)


# --------------------------------------------------------------- small-talk intents
SMALLTALK = [
    ("greeting", re.compile(r"^\s*(hi|hello|hey|hola|namaste|good\s+(morning|afternoon|evening))\b"),
     "Hello! I'm the support assistant. Ask me about orders, returns, payments, shipping or your account."),
    ("thanks", re.compile(r"\b(thanks|thank you|thx|appreciate)\b"),
     "You're welcome! Is there anything else I can help with?"),
    ("goodbye", re.compile(r"^\s*(bye|goodbye|see you|that'?s all|nothing else)\b"),
     "Goodbye! Have a great day."),
    ("handoff", re.compile(r"\b(human|agent|representative|real person|talk to someone|speak to someone)\b"),
     "I'll flag this for a human agent. You can also reach our team Mon-Sat, 9 AM-8 PM IST at "
     "1800-000-000 or support@example.com."),
]

FALLBACK_REPLY = (
    "I'm not sure I understood that. Could you rephrase, or pick one of the related topics below? "
    "If it's urgent, type 'agent' to reach a human."
)

CONTEXT_BOOST = 0.10  # score bonus for FAQs in the same category as the previous answer

ANAPHORA = {"it", "that", "this", "those", "them", "they", "same", "also", "else", "then"}


# ---------------------------------------------------------------------- engine
class ChatEngine:
    def __init__(self, faq_path="faqs.json", backend=None, threshold=None):
        with open(faq_path, encoding="utf-8") as fh:
            self.faqs = json.load(fh)
        self.by_id = {f["id"]: f for f in self.faqs}
        self.pre = Preprocessor()

        # One indexed row per phrasing (question + variations) -> parent FAQ
        self.docs, self.doc_faq = [], []
        for faq in self.faqs:
            for text in [faq["question"], *faq.get("variations", [])]:
                self.docs.append(text)
                self.doc_faq.append(faq["id"])

        self.tfidf = TfidfVectorizer(analyzer=self._analyze, sublinear_tf=True)
        self.doc_tfidf = self.tfidf.fit_transform(self.docs)

        self.encoder = None
        backend = (backend or os.environ.get("CHATBOT_BACKEND", "auto")).lower()
        if backend in ("transformer", "hybrid", "auto"):
            try:
                self._load_encoder()
            except Exception as exc:  # offline, missing torch, etc.
                if backend != "auto":
                    raise
                log.warning("Transformer unavailable (%s); falling back to TF-IDF", exc)
        self.backend = "tfidf" if self.encoder is None else ("transformer" if backend == "transformer" else "hybrid")
        default_thr = 0.30 if self.backend == "tfidf" else 0.45
        self.threshold = float(os.environ.get("CHATBOT_THRESHOLD", threshold if threshold is not None else default_thr))
        log.info("ChatEngine ready: backend=%s threshold=%.2f", self.backend, self.threshold)

    # TF-IDF analyzer: word unigrams + bigrams from the NLTK-processed tokens
    def _analyze(self, text):
        toks = self.pre.tokens(text)
        return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]

    def _load_encoder(self):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME).eval()
        self.encoder = True
        self.doc_emb = self._embed(self.docs)

    def _embed(self, texts):
        torch = self._torch
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            out = self.model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        emb = torch.nn.functional.normalize(emb, dim=1)
        return emb.numpy()

    # ---------------------------------------------------------------- retrieval
    def _scores(self, query):
        """Best similarity per FAQ id."""
        lex = (self.doc_tfidf @ self.tfidf.transform([query]).T).toarray().ravel()
        if self.encoder is None:
            doc_scores = lex
        else:
            sem = self.doc_emb @ self._embed([query])[0]
            doc_scores = sem if self.backend == "transformer" else 0.65 * sem + 0.35 * lex
        best = {}
        for score, fid in zip(doc_scores, self.doc_faq):
            if score > best.get(fid, -1):
                best[fid] = float(score)
        return sorted(best.items(), key=lambda kv: kv[1], reverse=True)

    def _is_followup(self, message):
        toks = set(re.findall(r"[a-z']+", message.lower()))
        return len(self.pre.tokens(message)) <= 3 or bool(toks & ANAPHORA)

    # ------------------------------------------------------------------ public
    def respond(self, message, last_turn=None):
        text = message.strip()

        for intent, pattern, reply in SMALLTALK:
            if pattern.search(text.lower()):
                return {"reply": reply, "intent": intent, "faq_id": None, "category": None,
                        "confidence": 1.0, "fallback": False, "used_context": False,
                        "matched_question": None, "suggestions": []}

        ranked = self._scores(text)
        used_context = False

        # Contextual follow-up ("how long does it take?", "and the refund?"): nudge FAQs from the
        # topic the user was just discussing, so ties/near-ties resolve toward the ongoing topic.
        if last_turn and last_turn.get("category") and self._is_followup(text):
            cat = last_turn["category"]
            boosted = sorted(
                ((fid, s + (CONTEXT_BOOST if self.by_id[fid]["category"] == cat else 0.0)) for fid, s in ranked),
                key=lambda kv: kv[1], reverse=True)
            changed = boosted[0][0] != ranked[0][0] or (
                ranked[0][1] < self.threshold <= boosted[0][1])
            ranked, used_context = boosted, changed

        top_id, top_score = ranked[0]
        faq = self.by_id[top_id]
        confident = top_score >= self.threshold
        related = [self.by_id[fid]["question"]
                   for fid, s in ranked[(1 if confident else 0):4] if s >= self.threshold * 0.5][:3]

        if confident:
            return {"reply": faq["answer"], "intent": "faq", "faq_id": top_id,
                    "category": faq["category"], "confidence": round(top_score, 3),
                    "fallback": False, "used_context": used_context,
                    "matched_question": faq["question"], "suggestions": related}
        return {"reply": FALLBACK_REPLY, "intent": "fallback", "faq_id": None, "category": None,
                "confidence": round(top_score, 3), "fallback": True, "used_context": False,
                "matched_question": None, "suggestions": related}
