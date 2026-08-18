"""
run_voice.py
-------------
Full real-time voice entry point for the Nexatel multilingual voice assistant.

Pipeline flow
-------------
  Microphone
    └─► Silero VAD  (utterance boundary detection)
          └─► ASRRouter
                ├─ Indic language → IndicConformer (Tier 1)
                └─ Other language → Whisper (Tier 2)
                      └─► ASRResult  { raw_text, detected_language }
                            └─► [pipeline.begin_tts() — mic stays live, barge-in armed]
                                  └─► LangGraph  (orchestrator → sub-agent → RAG → TTS)
                                        └─► edge-tts speaks reply in caller's language
                                              ├─ caller talks over it → on_barge_in() fires
                                              │  instantly, stop_event cuts playback short,
                                              │  their utterance becomes the next turn
                                              └─ TTS finishes normally → residual queue drained
                                  └─► [pipeline.end_tts()] └─► loop for next utterance

Barge-in requires a headset/earbuds for reliable results — there is no
acoustic echo cancellation, so on a speaker-only setup the assistant's own
voice can trigger a false interrupt. Pass --no_barge_in to fall back to the
old behaviour (microphone fully muted while the assistant speaks).

Usage
-----
    python scripts/run_voice.py
    python scripts/run_voice.py --phone 9876543210 --show_debug
    python scripts/run_voice.py --min_similarity 0.4
    python scripts/run_voice.py --no_barge_in

    # Ctrl+C to end the call at any point.

Startup sequence
----------------
  1. Phone number is prompted (10 digits).  Can be bypassed with --phone.
  2. VAD/ASR pipeline starts listening.
  3. Each detected utterance is transcribed, language auto-detected, and fed
     into LangGraph.  The assistant replies in the caller's detected language.
  4. The loop ends when:
       - The customer requests call termination (LangGraph sets call_end_requested)
       - The call is escalated to a human agent (handoff=True)
       - The user presses Ctrl+C
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import threading
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the package root is on sys.path when run as a script
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from dotenv import load_dotenv

load_dotenv()

import os
from langchain_core.messages import AIMessage, HumanMessage

from vay.audio.pipeline import STTPipeline
from vay.graph.workflow import build_graph
from vay.graph.state import GraphState
from vay.graph.utils import localized, trim_history
from vay.tools.session import SessionContext, load_customer_preferred_language
from vay.types import ASRResult, LanguageTier
from vay.config import settings as asr_settings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HANDOFF_LOG_PATH = "handoff_log.jsonl"
DEFAULT_MIN_SIMILARITY = 0.3
DEFAULT_MAX_HISTORY_TURNS = 6

# Post-TTS unmute delay (seconds).  Gives the speaker/room a moment to go
# quiet so the first VAD chunk after unmute doesn't capture speaker tail-ring.
_POST_TTS_SILENCE_S: float = 0.4

HANDOFF_MESSAGE_TEMPLATES = {
    "en": "I'm transferring you to a live Nexatel agent now. Please hold.",
    "hi": "मैं आपको अभी एक लाइव Nexatel एजेंट से जोड़ रहा हूँ। कृपया प्रतीक्षा करें।",
    "ta": "நான் உங்களை இப்போது ஒரு நேரடி Nexatel முகவரிடம் இணைக்கிறேன். தயவுசெய்து காத்திருங்கள்.",
}

# Transcripts that consist entirely of punctuation / whitespace are Whisper
# hallucinations on very short noise bursts.  Reject them before hitting the
# LangGraph to avoid spurious chitchat turns.
_NOISE_TRANSCRIPT_RE = re.compile(r'^[\s.,!?…\-–—\'\"]+$')


# ---------------------------------------------------------------------------
# Phone-number prompt
# ---------------------------------------------------------------------------

def _prompt_phone_number() -> str:
    """Prompt until a valid 10-digit phone number is entered."""
    while True:
        try:
            phone = input("\nCaller phone number (10 digits): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)
        if re.fullmatch(r"\d{10}", phone):
            return phone
        print("  Please enter exactly 10 digits (no spaces or dashes).")


# ---------------------------------------------------------------------------
# Voice call session
# ---------------------------------------------------------------------------

class VoiceCallSession:
    """Holds all mutable state for a single voice call.

    Thread-safety note: ``on_asr_result`` is called from the STTPipeline's
    consumer background thread sequentially (one utterance at a time), so
    ``conversation_history`` mutations are safe without additional locks.
    ``graph.invoke`` is safe to call from a background thread.
    """

    def __init__(
        self,
        phone_number: str,
        graph,
        pipeline: STTPipeline,
        min_similarity: float,
        max_history_turns: int,
        show_debug: bool,
    ) -> None:
        self.phone_number = phone_number
        self.graph = graph
        self.pipeline = pipeline          # kept so on_asr_result can mute/unmute
        self.min_similarity = min_similarity
        self.max_history_turns = max_history_turns
        self.show_debug = show_debug

        # Load the customer's registered language preference from the DB once at call
        # startup.  Used as an ASR override hint when Whisper auto-detection is low-confidence.
        preferred_lang = load_customer_preferred_language(phone_number)

        self.language = preferred_lang
        self.session = SessionContext(
            phone_number=phone_number,
            verified=True,
            language=preferred_lang,
            preferred_language=preferred_lang,
        )
        self.conversation_history: list = []
        self.is_active = True
        self._stop_event = threading.Event()

        # Barge-in: the stop_event for the TTS currently playing (if any).
        # Created fresh per turn in on_asr_result(); on_barge_in() sets it
        # the instant the pipeline detects the caller talking over the
        # assistant, which cuts tts.speak() short (see tts_node/tts/engine.py).
        self._tts_stop_event: threading.Event | None = None
        pipeline.on_barge_in = self._on_barge_in

    # ------------------------------------------------------------------
    # Barge-in callback — invoked on the STTPipeline's VAD producer thread
    # ------------------------------------------------------------------

    def _on_barge_in(self) -> None:
        """Fired the instant the caller starts talking over the assistant."""
        if self._tts_stop_event is not None:
            self._tts_stop_event.set()

    # ------------------------------------------------------------------
    # ASR callback — invoked by STTPipeline consumer thread
    # ------------------------------------------------------------------

    def on_asr_result(self, result: ASRResult) -> None:
        """Process one transcribed utterance through the LangGraph pipeline."""
        if not self.is_active:
            return

        transcript = result.raw_text.strip()
        lang = result.detected_language
        confidence = result.confidence

        # ------------------------------------------------------------------
        # Filter 1: empty transcript
        # ------------------------------------------------------------------
        if not transcript:
            print("[VoiceCall] Empty transcript — skipping.")
            return

        # ------------------------------------------------------------------
        # Filter 2: punctuation-only (Whisper hallucination on noise/echo)
        # A transcript that is entirely punctuation / whitespace (e.g. ".",
        # "...", "?") is not real speech — discard it silently.
        # ------------------------------------------------------------------
        if _NOISE_TRANSCRIPT_RE.fullmatch(transcript):
            print(f"[VoiceCall] Noise transcript ({transcript!r}) — skipping.")
            return

        # ------------------------------------------------------------------
        # Low-confidence Tier-2 → re-run with customer's registered language
        #
        # Whisper occasionally misidentifies Indian languages as South-East
        # Asian ones (e.g. Tamil → Indonesian) with low confidence (<0.5).
        # When this happens AND the customer's DB-registered language is a
        # Tier-1 Indic language, we re-run ASR with IndicConformer using the
        # registered language as a hard hint instead of trusting the Whisper
        # guess.  The result transcript + language replaces the original.
        # ------------------------------------------------------------------
        preferred_lang = self.session.preferred_language
        LOW_CONF_THRESHOLD = 0.5
        if (
            confidence < LOW_CONF_THRESHOLD
            and result.language_tier == LanguageTier.TIER_2
            and preferred_lang in asr_settings.tier1_languages
        ):
            print(
                f"[VoiceCall] Low Whisper confidence ({confidence:.2f}) on Tier-2 code '{lang}' "
                f"but customer's registered language is '{preferred_lang}' (Tier-1). "
                f"Re-running ASR with IndicConformer override."
            )
            # pipeline.router is the ASRRouter instance; re-transcribe the same
            # audio with the Tier-1 override.  We don't have the raw tensor here —
            # that lives inside the pipeline — so we use the last_audio_tensor
            # stored on the pipeline after each utterance.
            try:
                if hasattr(self.pipeline, 'router') and hasattr(self.pipeline, 'last_audio_tensor') \
                        and self.pipeline.last_audio_tensor is not None:
                    corrected = self.pipeline.router.route_and_transcribe(
                        self.pipeline.last_audio_tensor,
                        override_language=preferred_lang,
                    )
                    if corrected.raw_text.strip():
                        print(
                            f"[VoiceCall] IndicConformer override result: \"{corrected.raw_text}\" "
                            f"(lang={corrected.detected_language})"
                        )
                        result = corrected
                        transcript = corrected.raw_text.strip()
                        lang = corrected.detected_language
            except Exception as e:
                print(f"[VoiceCall] IndicConformer override failed ({e}) — using Whisper result.")

        print(f"\n[VoiceCall] Utterance  : \"{transcript}\"")
        print(f"[VoiceCall] Language   : {lang} (detected by ASR)")

        self.language = lang
        self.session.language = lang

        # Fresh stop_event for this turn's TTS — on_barge_in() sets it if the
        # caller starts talking while the assistant is still speaking.
        self._tts_stop_event = threading.Event()

        state: GraphState = {
            "phone_number": self.phone_number,
            "language": lang,
            "transcript": transcript,
            "conversation_history": self.conversation_history,
            "show_debug": self.show_debug,
            "min_similarity": self.min_similarity,
            "session": self.session,
            "barge_in_event": self._tts_stop_event,
        }

        # ------------------------------------------------------------------
        # Guard the microphone around TTS playback (edge-tts, inside
        # graph.invoke via tts_node) so the speaker output isn't fed back in
        # as the next user utterance.
        #
        # With barge-in enabled the mic is NOT muted: begin_tts() arms
        # barge-in detection instead, so genuine speech from the caller
        # interrupts tts.speak() (via _tts_stop_event) almost immediately and
        # that utterance flows straight through as the next query once this
        # graph.invoke() call returns. Without barge-in (or on a speaker-only
        # setup with no headset / no AEC) fall back to the old hard mute.
        # ------------------------------------------------------------------
        if self.pipeline.barge_in_enabled:
            self.pipeline.begin_tts()
        else:
            self.pipeline.mute()
        try:
            result_state = self.graph.invoke(state)
        except Exception as e:
            print(f"[VoiceCall] LangGraph error: {e}")
            return
        finally:
            barged_in = self._tts_stop_event.is_set()
            self._tts_stop_event = None
            if self.pipeline.barge_in_enabled:
                self.pipeline.end_tts()
                if not barged_in:
                    # TTS finished on its own — a residual capture in the
                    # queue would be leftover echo/noise, not a real query.
                    # Drain it (unmute() is a no-op on mute state here, but
                    # still does the draining). If barge-in DID fire, the
                    # queue instead holds the caller's real interrupting
                    # utterance — keep it, it becomes the next turn once
                    # this call returns.
                    time.sleep(_POST_TTS_SILENCE_S)
                    self.pipeline.unmute()
            else:
                time.sleep(_POST_TTS_SILENCE_S)
                self.pipeline.unmute()

        reply = result_state.get("final_reply") or localized(
            HANDOFF_MESSAGE_TEMPLATES, result_state.get("language", lang)
        )

        self.conversation_history.append(HumanMessage(content=transcript))
        self.conversation_history.append(AIMessage(content=reply))
        self.conversation_history = trim_history(
            self.conversation_history, self.max_history_turns
        )

        print(f"[VoiceCall] Assistant  : {reply}\n")

        if result_state.get("call_end_requested"):
            print("--- Call ended by customer. Session terminated. ---")
            self.is_active = False
            self._stop_event.set()

        elif result_state.get("handoff"):
            reason = result_state.get("handoff_reason", "n/a")
            print(
                f"--- Call transferred to a human agent ({reason}). "
                f"Session terminated. See {HANDOFF_LOG_PATH} for the escalation packet. ---"
            )
            self.is_active = False
            self._stop_event.set()

    def wait_for_stop(self) -> None:
        """Block until the session signals it should end.
        Uses a 200ms polling loop instead of a bare ``_stop_event.wait()``
        because ``threading.Event.wait()`` without a timeout can swallow
        ``SIGINT``/``KeyboardInterrupt`` on Windows — the GIL doesn't release
        to the interrupt handler during a blocking C-level wait.  With the
        short timeout the main thread wakes up every 200 ms, which is fast
        enough to feel instant and slow enough to be CPU-free.
        """
        while not self._stop_event.is_set():
            # 200 ms timeout gives the Python SIGINT handler a chance to fire
            self._stop_event.wait(timeout=0.2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nexatel real-time voice assistant — VAD → ASR → LangGraph → TTS loop."
    )
    parser.add_argument(
        "--phone",
        default=None,
        help="10-digit caller phone number (skips the interactive prompt).",
    )
    parser.add_argument(
        "--min_similarity",
        type=float,
        default=DEFAULT_MIN_SIMILARITY,
        help="RAG confidence gate threshold (default: 0.3).",
    )
    parser.add_argument(
        "--max_history_turns",
        type=int,
        default=DEFAULT_MAX_HISTORY_TURNS,
        help="Conversation history window in turns (default: 6).",
    )
    parser.add_argument(
        "--show_debug",
        action="store_true",
        help="Print orchestrator JSON and tool-call traces.",
    )
    parser.add_argument(
        "--no_barge_in",
        action="store_true",
        help=(
            "Disable barge-in: fall back to fully muting the microphone while "
            "the assistant speaks. Use this on speaker-only setups (no "
            "headset) where TTS echo could otherwise misfire as barge-in."
        ),
    )
    args = parser.parse_args()

    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY environment variable is not set.")
        sys.exit(1)

    # ----------------------------------------------------------------
    # Step 1: Collect phone number
    # ----------------------------------------------------------------
    phone_number = (
        args.phone
        if args.phone and re.fullmatch(r"\d{10}", args.phone)
        else _prompt_phone_number()
    )

    print(f"\n[VoiceCall] Starting session for phone: {phone_number}")
    print("[VoiceCall] Speak into the microphone. Press Ctrl+C to end the call.\n")

    # ----------------------------------------------------------------
    # Step 2: Build LangGraph
    # ----------------------------------------------------------------
    print("[VoiceCall] Building LangGraph...")
    graph = build_graph()

    # ----------------------------------------------------------------
    # Step 3: Create pipeline FIRST (session needs the reference)
    # ----------------------------------------------------------------
    # Callback is set after session creation (chicken-and-egg); we use
    # a wrapper lambda so the pipeline exists before the session does.
    pipeline = STTPipeline(callback=None, barge_in=not args.no_barge_in)   # callback wired in step 4

    # ----------------------------------------------------------------
    # Step 4: Create voice session (needs the pipeline for mute/unmute)
    # ----------------------------------------------------------------
    call_session = VoiceCallSession(
        phone_number=phone_number,
        graph=graph,
        pipeline=pipeline,
        min_similarity=args.min_similarity,
        max_history_turns=args.max_history_turns,
        show_debug=args.show_debug,
    )
    pipeline.callback = call_session.on_asr_result   # wire callback

    # ----------------------------------------------------------------
    # Step 5: Run pipeline in a daemon thread
    # ----------------------------------------------------------------
    pipeline_thread = threading.Thread(
        target=pipeline.start, daemon=True, name="stt-pipeline"
    )
    pipeline_thread.start()

    # ----------------------------------------------------------------
    # Step 6: Main thread waits for call termination or Ctrl+C
    # ----------------------------------------------------------------
    try:
        call_session.wait_for_stop()
    except KeyboardInterrupt:
        print("\n[VoiceCall] Interrupted by user.")
    finally:
        pipeline.stop()
        print("[VoiceCall] Session ended.")


if __name__ == "__main__":
    main()
