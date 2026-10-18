"""Answer questions about this project in plain English.

Retrieval-augmented generation, with one addition that matters: a question is
first routed to the source that can actually answer it.

  "why was weather dropped?"            -> retrieval over the fact base
  "what is Midtown's median fare?"      -> the zone's own statistics
  "Midtown to JFK on Friday at 5pm?"    -> the fare model itself

Retrieving a number that should have been predicted is the usual way a project
like this ends up confidently wrong, so a question that names two zones is
priced rather than looked up, and the real prediction is handed to the language
model as a fact it must use.

Retrieval is TF-IDF rather than embeddings. The fact base is small and the
vocabulary is narrow, TF-IDF needs no second API call on the way in, and
scikit-learn is already a dependency - an embedding model would have added
hundreds of megabytes to a deployment that has a size limit.
"""
import json
import os
import re
import time
import urllib.error
import urllib.request

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

API_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
TIMEOUT_S = 45

# Tried in order. The newest model returns 503 "experiencing high demand" often
# enough on the free tier that a single attempt is not good enough, and an older
# flash model answers a grounded question just as well.
MODELS = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-flash-lite-latest"]
RETRY_CODES = {429, 500, 502, 503, 504}
RETRY_PAUSE_S = 1.2

TOP_K = 8                  # facts handed to the model
MIN_SCORE = 0.03           # below this a fact is noise, not a weak match

DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
DAY_WORDS = {d[:3]: i for i, d in enumerate(DAYS)} | {d: i for i, d in enumerate(DAYS)}

PRICE_WORDS = re.compile(
    r"\b(cost|costs|fare|fares|price|prices|priced|pricing|charge|charged|"
    r"how much|expensive|cheap|cheaper)\b", re.I)

SYSTEM = """You answer questions about NYCRide, a fare prediction project built on \
New York City taxi trip records.

Rules:
- Answer only from the facts given to you. They are the project's real numbers.
- If the facts do not cover the question, say so plainly and say what you can \
answer instead. Never invent a number.
- Quote numbers exactly as they are written. Do not add, subtract, convert, \
round or combine them into a new figure - a derived number is a wrong number \
here, even when the arithmetic is right.
- Write in simple, clear English for someone who is not a machine learning \
expert. Explain a term the first time you use it.
- Be brief: two to four sentences for most questions. Use a short list only when \
the question really asks for several items.
- Quote the project's actual numbers when they are relevant.
- Do not mention "facts", "context", "documents" or "retrieval" in your answer. \
Just answer the question.
- Never start with a greeting or a restatement of the question."""


class Ask:
    """Holds the fact base and the routing, so the API can stay thin."""

    def __init__(self, knowledge_path, zones, estimate):
        """zones: list of (zone_id, name, borough). estimate: a callable the
        router uses to price a trip, so this module never imports the API."""
        blob = json.loads(knowledge_path.read_text(encoding="utf-8"))
        self.facts = blob["facts"]
        self.built_from = blob.get("built_from", {})
        self.estimate = estimate

        # keywords are indexed but never shown; they exist so a question worded
        # differently from the fact still finds it
        corpus = [f"{f['title']}. {f['text']} {f.get('keywords', '')}"
                  for f in self.facts]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                          sublinear_tf=True, min_df=1)
        self.matrix = self.vectorizer.fit_transform(corpus)

        # longest names first, so "East Village" is not matched as "Village"
        self.zones = sorted(zones, key=lambda z: -len(z[1]))

    # ---------------------------------------------------------------- routing

    def find_zones(self, question):
        """Zone names mentioned in the question, in the order they appear."""
        low = question.lower()
        found, taken = [], []
        for zid, name, borough in self.zones:
            at = low.find(name.lower())
            if at < 0 or any(a <= at < b for a, b in taken):
                continue
            taken.append((at, at + len(name)))
            found.append((at, zid, name, borough))
        return [(z, n, b) for _, z, n, b in sorted(found)]

    @staticmethod
    def find_time(question):
        """An hour and a weekday if the question names them, else the defaults
        the rest of the site uses - Friday at 17:00."""
        low = question.lower()
        hour, dow = 17, 4

        m = re.search(r"\b(\d{1,2})\s*(?::(\d{2}))?\s*(am|pm)\b", low)
        if m:
            hour = int(m.group(1)) % 12 + (12 if m.group(3) == "pm" else 0)
        else:
            m = re.search(r"\b(\d{1,2}):(\d{2})\b", low)
            if m and int(m.group(1)) < 24:
                hour = int(m.group(1))
            elif re.search(r"\bmidnight\b", low):
                hour = 0
            elif re.search(r"\bnoon|midday\b", low):
                hour = 12

        for word, i in DAY_WORDS.items():
            if re.search(rf"\b{word}\b", low):
                dow = i
                break
        else:
            if re.search(r"\bweekend\b", low):
                dow = 5
            elif re.search(r"\bweekday\b", low):
                dow = 2
        return hour, dow

    def priced_trip(self, question):
        """If the question is asking what a named trip costs, price it."""
        zones = self.find_zones(question)
        if len(zones) < 2 or not PRICE_WORDS.search(question):
            return None
        (pu, pu_name, _), (do, do_name, _) = zones[0], zones[1]
        if pu == do:
            return None

        hour, dow = self.find_time(question)
        try:
            est = self.estimate(pu, do, hour, dow)
        except Exception:
            return None

        day = DAYS[dow].capitalize()
        return {
            "title": f"predicted fare: {pu_name} to {do_name}",
            "text": (
                f"The model prices a solo Uber trip from {pu_name} to {do_name} at "
                f"{day} {hour:02d}:00 at ${est['fare']:.2f}, likely between "
                f"${est['low']:.2f} and ${est['high']:.2f}. The trip is about "
                f"{est['miles']} miles ({est['miles_source']}), which works out at "
                f"${est['per_mile']:.2f} a mile. The same trip on Lyft prices at "
                f"${est['lyft']:.2f}, so {est['cheaper']} is cheaper on this route."),
            "detail": est,
        }

    # -------------------------------------------------------------- retrieval

    def retrieve(self, question, k=TOP_K):
        scores = (self.matrix @ self.vectorizer.transform([question]).T).toarray().ravel()
        order = np.argsort(scores)[::-1][:k]
        return [(self.facts[i], float(scores[i])) for i in order if scores[i] >= MIN_SCORE]

    def gather(self, question):
        """Everything worth showing the model, best source first."""
        picked, seen = [], set()

        trip = self.priced_trip(question)
        if trip:
            picked.append({"title": trip["title"], "text": trip["text"],
                           "topic": "live prediction", "score": 1.0})
            seen.add(trip["title"])

        # a named zone is worth including even if the wording hides it from TF-IDF
        for zid, name, _ in self.find_zones(question)[:2]:
            for f in self.facts:
                if f["topic"] == "zone" and f["title"] == f"zone: {name}" \
                        and f["title"] not in seen:
                    picked.append({**f, "score": 1.0})
                    seen.add(f["title"])

        for f, score in self.retrieve(question):
            if f["title"] not in seen:
                picked.append({**f, "score": round(score, 3)})
                seen.add(f["title"])
        return picked[:TOP_K + 2], trip

    # ------------------------------------------------------------- generation

    @staticmethod
    def _post(model, key, payload):
        """One attempt. Returns (text, error); error is None on success."""
        body = json.dumps({"model": model, **payload}).encode("utf-8")
        req = urllib.request.Request(
            API_URL, data=body,
            headers={"x-goog-api-key": key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                data = json.load(r)
        except urllib.error.HTTPError as e:
            return None, (e.code, e.read().decode("utf-8", "replace")[:160])
        except Exception as e:
            return None, (None, e.__class__.__name__)

        # Walk the steps rather than taking the last one: a thought step comes
        # first, and the text can arrive in several blocks.
        parts = [b["text"] for s in data.get("steps", []) if s.get("type") != "thought"
                 for b in (s.get("content") or [])
                 if b.get("type") == "text" and b.get("text")]
        text = "\n".join(parts).strip()
        return (text, None) if text else (None, (None, "empty_response"))

    def call_model(self, question, context):
        key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not key:
            return None, "no_key"

        payload = {
            "system_instruction": SYSTEM,
            "input": f"Facts about the project:\n\n{context}\n\nQuestion: {question}",
            # the thinking budget is the bulk of the latency, and this is a
            # grounded question rather than a puzzle
            "generation_config": {"thinking_level": "low", "max_output_tokens": 600},
        }

        last = "no attempt"
        for model in MODELS:
            for attempt in (1, 2):
                text, err = self._post(model, key, payload)
                if text:
                    return text, None
                code, detail = err
                last = f"{model}: {code or ''} {detail}".strip()
                if code not in RETRY_CODES:
                    break                      # a real failure; next model
                if attempt == 1:
                    time.sleep(RETRY_PAUSE_S)
        return None, last

    def answer(self, question):
        question = (question or "").strip()
        if not question:
            return {"ok": False, "answer": "Ask me something about the project.",
                    "sources": []}

        picked, trip = self.gather(question)
        if not picked:
            return {
                "ok": False,
                "answer": ("I do not have anything on that. I can answer questions "
                           "about the fare model and how it scores, the data and "
                           "how it was cleaned, any taxi zone, or what a particular "
                           "trip would cost."),
                "sources": [],
            }

        context = "\n\n".join(f"[{p['title']}] {p['text']}" for p in picked)
        text, err = self.call_model(question, context)

        if text is None:
            # The retrieval still worked, so show what was found rather than an
            # error page. This is also what the site does before a key is set.
            return {
                "ok": False, "error": err,
                "answer": ("I could not reach the language model, so here is what "
                           "the project's own records say:"),
                "sources": [{"title": p["title"], "text": p["text"],
                             "topic": p["topic"]} for p in picked[:4]],
                "priced": trip["detail"] if trip else None,
            }

        return {
            "ok": True,
            "answer": text,
            "sources": [{"title": p["title"], "text": p["text"], "topic": p["topic"]}
                        for p in picked[:5]],
            "priced": trip["detail"] if trip else None,
        }
