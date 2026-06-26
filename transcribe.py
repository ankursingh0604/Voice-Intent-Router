import os
import io
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ── Limits and guards ─────────────────────────────────────────────────────────
# LOOPHOLE #1 (file size abuse): without a cap, someone can upload an
# arbitrarily large file. Groq's Whisper endpoint itself has a hard limit
# (25MB as of their current docs), but relying on the API to reject it
# means we've already spent bandwidth/time uploading a huge file before
# finding out. We enforce a stricter limit ourselves, BEFORE the network
# call, so bad input fails fast and cheap.
MAX_AUDIO_BYTES = 15 * 1024 * 1024  # 15MB — Groq's documented hard limit
                                      # is 25MB; we cap stricter than that
                                      # so we fail fast on oversized files
                                      # before spending time on the upload,
                                      # while still allowing 10+ minutes of
                                      # compressed speech, far more than
                                      # any single voice query needs

MIN_AUDIO_BYTES = 1000  # reject near-empty files (e.g. a 0-byte or
                          # near-instant accidental recording) before
                          # even calling the API

# Matches Groq's documented supported formats exactly (per their API
# reference) rather than guessing at a broader list — submitting an
# unsupported format wastes a round trip only to get rejected anyway.
ALLOWED_EXTENSIONS = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"}

# LOOPHOLE #2 (Whisper hallucination on non-speech): Whisper is known to
# "transcribe" silence or noise into plausible-sounding but fake text
# (this is documented behavior of the underlying model architecture, not
# a bug specific to this project — it's trained to always produce SOME
# text). A common heuristic signal for this is segments with very low
# average log-probability / high "no_speech_prob". Groq's API exposes
# verbose_json output with segment-level data we can inspect, rather than
# blindly trusting the plain text transcript.
NO_SPEECH_PROB_THRESHOLD = 0.6  # if Whisper itself thinks there's a >60%
                                  # chance a segment is NOT speech, we
                                  # don't trust that segment's text


class TranscriptionResult:
    def __init__(self, text: str, is_likely_valid: bool, reason: str = ""):
        self.text = text
        self.is_likely_valid = is_likely_valid
        self.reason = reason


def validate_audio_file(file_bytes: bytes, filename: str) -> tuple[bool, str]:
    """
    Pre-flight checks BEFORE calling the Whisper API at all.
    Returns (is_valid, error_message).
    """
    if len(file_bytes) > MAX_AUDIO_BYTES:
        return False, f"Audio file too large ({len(file_bytes) / 1_000_000:.1f}MB). Max allowed: {MAX_AUDIO_BYTES / 1_000_000:.0f}MB."

    if len(file_bytes) < MIN_AUDIO_BYTES:
        return False, "Audio file is empty or too short to contain speech."

    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported audio format '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"

    return True, ""


def transcribe_audio(file_bytes: bytes, filename: str) -> TranscriptionResult:
    """
    Transcribes audio via Groq's Whisper endpoint, with hallucination
    guarding via segment-level no_speech_prob inspection.

    Returns a TranscriptionResult — callers should check is_likely_valid
    before trusting `.text`, rather than assuming any returned text is
    real speech.
    """
    is_valid, error = validate_audio_file(file_bytes, filename)
    if not is_valid:
        return TranscriptionResult(text="", is_likely_valid=False, reason=error)

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        # LOOPHOLE #7: fail with a clear, specific error rather than
        # letting the Groq client raise its own less-obvious exception
        # several layers down.
        return TranscriptionResult(
            text="", is_likely_valid=False,
            reason="Server misconfiguration: GROQ_API_KEY is not set."
        )

    client = Groq(api_key=api_key)

    try:
        result = client.audio.transcriptions.create(
            file=(filename, io.BytesIO(file_bytes)),
            model="whisper-large-v3",
            response_format="verbose_json",  # gives us segment-level data,
                                               # not just plain text
            temperature=0.0,  # deterministic transcription — we don't
                                # want random variation on the same audio
            timestamp_granularities=["segment"],  # explicitly requested
                                # rather than relying on an implicit
                                # default, since this is what populates
                                # the segments/no_speech_prob data the
                                # hallucination guard below depends on
        )
    except Exception as e:
        return TranscriptionResult(
            text="", is_likely_valid=False,
            reason=f"Transcription service error: {str(e)}"
        )

    text = (result.text or "").strip()

    if not text:
        return TranscriptionResult(
            text="", is_likely_valid=False,
            reason="No speech detected in audio."
        )

    # Hallucination guard — inspect segments if the API returned them.
    # Not every Groq SDK version/response guarantees `segments` is
    # present, so this degrades gracefully rather than crashing if it's
    # missing — we just skip the extra check and trust the plain text in
    # that case, rather than failing the whole transcription over a
    # missing optional field.
    segments = getattr(result, "segments", None)
    if segments:
        high_no_speech_segments = [
            s for s in segments
            if getattr(s, "no_speech_prob", 0) > NO_SPEECH_PROB_THRESHOLD
        ]
        # If MOST segments are flagged as likely non-speech, treat the
        # whole transcript as suspect rather than confidently wrong.
        if len(high_no_speech_segments) >= max(1, len(segments) * 0.7):
            return TranscriptionResult(
                text=text, is_likely_valid=False,
                reason="Audio appears to contain mostly silence or non-speech sound — transcription may be unreliable."
            )

    return TranscriptionResult(text=text, is_likely_valid=True)
