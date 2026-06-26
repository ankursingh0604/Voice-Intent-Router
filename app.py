import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")
try:
    API_URL = st.secrets["API_URL"]
except Exception:
    pass

st.set_page_config(page_title="Voice Intent Router", page_icon="🎙️", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
    * { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Syne', sans-serif !important; }
    .block-container { padding: 2rem 2rem; max-width: 1100px; }

    .hero {
        background: linear-gradient(135deg, #0a0a14 0%, #10081a 50%, #0a0a14 100%);
        border: 1px solid #2a1e2a;
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1.5rem;
    }
    .hero-title { font-family: 'Syne', sans-serif !important; font-size: 2rem; font-weight: 800; color: #f0f0f8; margin: 0 0 0.5rem 0; }
    .hero-sub { font-size: 0.9rem; color: #9988aa; margin: 0; }

    .category-pills { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 1rem; }
    .pill { padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 500; border: 1px solid; }
    .pill-fact { background: #0f1a2a; color: #4488ee; border-color: #1a3a5a; }
    .pill-time { background: #1a2a0f; color: #88ee44; border-color: #3a5a1a; }
    .pill-opinion { background: #2a1a0f; color: #ee9944; border-color: #5a3a1a; }
    .pill-code { background: #1a0f2a; color: #aa66ee; border-color: #3a1a5a; }
    .pill-talk { background: #2a0f1a; color: #ee6699; border-color: #5a1a3a; }
    .pill-other { background: #1a1a1a; color: #999999; border-color: #3a3a3a; }

    .transcript-box { background: #0a0f1a; border: 1px solid #1a2a3e; border-radius: 10px; padding: 1rem 1.2rem; margin: 1rem 0; font-size: 14px; color: #8899aa; font-style: italic; }
    .warning-box { background: #2a1a0a; border: 1px solid #5a3a1a; border-radius: 10px; padding: 0.8rem 1.2rem; margin: 0.5rem 0; font-size: 13px; color: #ddaa66; }
    .response-box { background: #080c14; border: 1px solid #1a2a3e; border-radius: 10px; padding: 1.5rem; font-size: 15px; line-height: 1.7; color: #c0ccd8; }

    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #5a1a5a, #1a3a5a);
        color: white; border: none; border-radius: 8px;
        font-family: 'Syne', sans-serif; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="hero-title">🎙️ Voice Intent Router</div>
    <p class="hero-sub">Speak anything — a fact, a question, an opinion, code, or small talk. The agent classifies it and routes to the right handler.</p>
    <div class="category-pills">
        <span class="pill pill-fact">Factual Question</span>
        <span class="pill pill-time">Time-Sensitive</span>
        <span class="pill pill-opinion">Opinion</span>
        <span class="pill pill-code">Coding Request</span>
        <span class="pill pill-talk">Small Talk</span>
        <span class="pill pill-other">Unclear / Other</span>
    </div>
</div>
""", unsafe_allow_html=True)

CATEGORY_PILL_MAP = {
    "factual_question": "pill-fact",
    "time_sensitive_question": "pill-time",
    "opinion_question": "pill-opinion",
    "coding_request": "pill-code",
    "small_talk": "pill-talk",
    "unclear_or_other": "pill-other",
}
CATEGORY_LABEL_MAP = {
    "factual_question": "FACTUAL QUESTION",
    "time_sensitive_question": "TIME-SENSITIVE",
    "opinion_question": "OPINION",
    "coding_request": "CODING REQUEST",
    "small_talk": "SMALL TALK",
    "unclear_or_other": "UNCLEAR / OTHER",
}

# ── Input method ──────────────────────────────────────────────────────────────

st.markdown("#### 🎤 Record or upload your voice input")

tab_record, tab_upload = st.tabs(["🔴 Record (mic)", "📁 Upload audio file"])

audio_bytes = None
audio_filename = "recording.wav"

with tab_record:
    try:
        from streamlit_mic_recorder import mic_recorder
        rec = mic_recorder(
            start_prompt="🔴 Start recording",
            stop_prompt="⏹️ Stop recording",
            just_once=True,
            key="recorder"
        )
        if rec:
            st.session_state["audio_bytes"] = rec["bytes"]
            st.session_state["audio_filename"] = "recording.wav"
    except ImportError:
        st.warning(
            "Mic recording component not installed (`pip install streamlit-mic-recorder`). "
            "Use the Upload tab instead, or install it for in-browser recording."
        )

with tab_upload:
    uploaded = st.file_uploader(
        "Upload an audio file",
        type=["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"],
        label_visibility="collapsed"
    )
    if uploaded is not None:
        st.session_state["audio_bytes"] = uploaded.read()
        st.session_state["audio_filename"] = uploaded.name

audio_bytes = st.session_state.get("audio_bytes")
audio_filename = st.session_state.get("audio_filename", "recording.wav")

if audio_bytes:
    st.audio(audio_bytes)

process_btn = st.button("🚀 Process Voice Input", use_container_width=True, disabled=(audio_bytes is None))

if process_btn and audio_bytes:
    with st.spinner("Transcribing and routing..."):
        try:
            files = {"audio": (audio_filename, audio_bytes)}
            response = requests.post(f"{API_URL}/route", files=files, timeout=60)

            if response.status_code == 200:
                st.session_state["last_result"] = response.json()
            elif response.status_code == 400:
                st.error(f"⚠️ {response.json().get('detail', 'Bad request')}")
            elif response.status_code == 429:
                st.error("⏳ Rate limit reached — please wait a moment and try again.")
            else:
                st.error(f"Error: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Run: `uvicorn api:app --reload`")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# ── Result display ────────────────────────────────────────────────────────────

if "last_result" in st.session_state:
    r = st.session_state["last_result"]

    if r.get("flagged_as_possible_injection"):
        st.markdown(
            '<div class="warning-box">⚠️ This input was flagged as a possible '
            'prompt-injection attempt and handled via the safety-net path rather '
            'than normal classification.</div>',
            unsafe_allow_html=True
        )

    st.markdown(f'<div class="transcript-box">🗣️ "{r["transcript"]}"</div>', unsafe_allow_html=True)

    if not r.get("transcript_is_reliable"):
        st.markdown(
            f'<div class="warning-box">⚠️ Transcription confidence is low: {r.get("transcript_warning", "")}</div>',
            unsafe_allow_html=True
        )

    category = r.get("category", "unclear_or_other")
    pill_class = CATEGORY_PILL_MAP.get(category, "pill-other")
    pill_label = CATEGORY_LABEL_MAP.get(category, category.upper())
    st.markdown(f'<span class="pill {pill_class}">{pill_label}</span>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    response_type = r.get("response_type", "text")
    response_text = r.get("response", "")

    st.markdown('<div class="response-box">', unsafe_allow_html=True)
    if response_type == "code":
        # Deliberately NOT offering "Read aloud" for code responses — see
        # agent.py's handle_coding() docstring for why spoken code is a
        # poor fit for this output type.
        st.markdown("**Code response:**")
        st.code(response_text, language=None)
    else:
        st.markdown(response_text)
    st.markdown('</div>', unsafe_allow_html=True)

    if r.get("sources"):
        st.markdown("**Sources:**")
        for src in r["sources"]:
            st.markdown(f"- {src}")

    # Optional TTS — only offered for text responses, never for code.
    if response_type == "text":
        col_tts, _ = st.columns([1, 3])
        with col_tts:
            if st.button("🔊 Read aloud", key="tts_btn"):
                # Browser-native speechSynthesis — zero cost, no API call.
                # See project README for why this is the default TTS
                # approach (and Amazon Polly as the documented upgrade
                # path once an AWS account with billing exists).
                safe_text = response_text.replace("`", "").replace('"', "'").replace("\n", " ")
                st.components.v1.html(f"""
                    <script>
                    const utterance = new SpeechSynthesisUtterance("{safe_text}");
                    window.speechSynthesis.speak(utterance);
                    </script>
                """, height=0)
