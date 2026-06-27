# 🎙️ Voice Intent Router

A speech-in, text-out agent that classifies spoken input into one of 6 fixed categories — factual question, time-sensitive question, opinion, coding request, small talk, or unclear/other — and routes each to a specialized handler. Text-first responses by default, with optional on-demand text-to-speech playback.

Built with LangGraph conditional routing, Groq Whisper for transcription, and Tavily for live search.

🚀 [Live Demo](your-streamlit-url) &nbsp;|&nbsp; 🔌 [API](your-railway-url)

---

## What it does

Record or upload a short voice clip saying anything — a fact-based question, something needing current information, an opinion, a coding request, or just small talk. The system:

1. **Transcribes** the audio via Groq's Whisper API
2. **Classifies** the transcript into exactly one of 6 fixed categories
3. **Routes** to a category-specific handler — factual answers go straight to the LLM, time-sensitive questions trigger a live Tavily search, opinions get multi-perspective framing instead of one asserted answer, coding requests render as syntax-highlighted code, small talk gets a brief conversational reply, and anything that doesn't clearly fit lands in a safety-net fallback that still attempts to be useful
4. **Displays text by default** — with an optional "🔊 Read aloud" button for browser-based TTS playback on text responses (not offered for code, since spoken code isn't useful)

---

## Architecture

```
Voice input (mic recording or file upload)
    ↓
transcribe_audio() — Groq Whisper, verbose_json + segment-level
                       no_speech_prob inspection for hallucination guarding
    ↓
classify_input node — LLM classifies into one of 6 fixed categories,
                        with strict normalization/validation against the
                        exact category set the graph's edges expect
    ↓
conditional routing (route_by_category)
    ├── factual_question        → direct LLM answer
    ├── time_sensitive_question → Tavily search → answer with sources
    ├── opinion_question        → multi-perspective framing
    ├── coding_request          → code generation, rendered as code
    ├── small_talk              → lightweight conversational reply
    └── unclear_or_other        → safety-net fallback, best-effort response
    ↓
Text response (default) + optional browser TTS on demand
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Agent Framework | LangGraph (conditional routing) |
| Speech-to-Text | Groq Whisper (whisper-large-v3) |
| LLM | LLaMA 3.3 70B via Groq API |
| Web Search | Tavily Search API |
| Text-to-Speech | Browser-native `speechSynthesis` (on-demand only) |
| Backend | FastAPI |
| Frontend | Streamlit (`streamlit-mic-recorder` for in-browser recording) |
| Deployment | Docker + Railway + Streamlit Cloud |

---

## Security and reliability — what's actually handled, and what isn't

Written the same way as the security section in my Customer Support Agent project: stating plainly what's genuinely fixed versus what's a deliberate, acknowledged scope limit, rather than implying full coverage.

**Genuinely handled:**

- **Audio file abuse** — file size capped at 15MB (stricter than Groq's 25MB API limit) and a minimum size floor, both checked *before* the file is sent to the transcription API, so oversized or empty uploads fail fast without wasting a network round trip.
- **Whisper hallucination on silence/noise** — Whisper is documented to sometimes "transcribe" silence or background noise into plausible-but-fake text. `transcribe.py` requests segment-level `no_speech_prob` metadata and flags the transcript as unreliable if most segments score above threshold, rather than blindly trusting any returned text.
- **Classification-to-routing mismatch** — LangGraph's conditional edges need an exact string match. The classifier's raw LLM output is never passed directly to the router; it's normalized and validated against the fixed category list first, with anything that doesn't match exactly forced to `unclear_or_other` instead of crashing the graph.
- **Prompt injection awareness** — a keyword-based pre-check flags common injection phrasings (e.g. "ignore previous instructions") and routes them to the fallback handler rather than normal classification. Stated honestly: this is a basic deterrent, not a solved problem — no prompt injection defense is fully reliable against a determined attacker, and this project doesn't claim otherwise.
- **Clear error surfaces** — missing API keys, transcription failures, and unexpected agent errors all return specific HTTP status codes and messages rather than generic crashes or silent failures.

**Deliberately out of scope, stated plainly rather than silently skipped:**

- **Real production rate limiting** — a simple in-memory per-window request counter exists as a basic safeguard against accidental loops, but it resets on restart and isn't distributed across instances. A real deployment would need Redis-backed limiting or a gateway-level solution, not in-process counters.
- **Full prompt injection resistance** — flagging known phrasings is a deterrent, not a guarantee; a sufficiently creative injection attempt could still get through.
- **Real-time/streaming conversation** — this is intentionally async (record → stop → process → respond), not a live back-and-forth voice call. That's a different, considerably more complex problem (voice activity detection, streaming transcription, WebSocket handling) that was scoped out from the start, not something that broke.

---

## Setup

1. Clone and install
```bash
git clone https://github.com/ankursingh0604/voice-intent-router
cd voice-intent-router
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

2. Create `.env`
```
GROQ_API_KEY=your-groq-api-key
TAVILY_API_KEY=your-tavily-api-key
LANGCHAIN_API_KEY=your-langsmith-api-key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=voice-intent-router
```

3. Run
```bash
# Terminal 1
uvicorn api:app --reload

# Terminal 2
streamlit run app.py
```

---

## Project Structure

```
voice-intent-router/
├── transcribe.py   ← Whisper STT with validation + hallucination guarding
├── agent.py        ← LangGraph classifier + 6 category handlers
├── api.py          ← FastAPI endpoint, rate limiting, injection pre-check
├── app.py           ← Streamlit frontend with mic recording + optional TTS
├── requirements.txt
└── Dockerfile
```

---

## Author

**Ankur Singh** — CS undergrad building RAG systems and AI agents

[GitHub](https://github.com/ankursingh0604)git • [LinkedIN](https://www.linkedin.com/in/ankur-singh-ai/) • [X](https://x.com/ankur_builds)
