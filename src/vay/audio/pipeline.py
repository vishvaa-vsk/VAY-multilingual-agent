"""Audio STT pipeline manager (Producer-Consumer architecture).

The pipeline captures microphone audio via Silero VAD, detects the utterance
boundary, then routes the complete utterance to the correct ASR model
(IndicConformer for Indic languages, Whisper for everything else).

Usage — standalone demo (prints results):
    python -m vay.audio.pipeline

Usage — with a downstream callback (wires into LangGraph):
    def on_result(result: ASRResult) -> None:
        ...  # feed into graph.invoke(...)

    pipeline = STTPipeline(callback=on_result)
    pipeline.start()          # blocks until KeyboardInterrupt
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Callable

import torch

from vay.asr.router import ASRRouter
from vay.audio.vad import SileroVADStreamer
from vay.config import settings
from vay.types import ASRResult

# Minimum number of audio samples an utterance must contain before it is sent
# to the ASR model.  Short captures (~<0.5 s at 16 kHz = 8 000 samples) are
# almost always noise bursts, mic clicks, or the tail-end of TTS playback
# bleeding into the microphone — Whisper reliably hallucinates "." or a random
# word on them.  The value here is conservative; increase if false positives
# persist.
_MIN_UTTERANCE_SAMPLES: int = 8_000   # ~0.5 s at 16 kHz


class STTPipeline:
    """Manages the lifecycle of real-time audio chunking, language ID, and STT routing.

    Args:
        callback: Optional callable that receives each :class:`ASRResult` after
            transcription.  If *None*, results are printed to stdout.  The
            callback is invoked on the consumer background thread — keep it
            thread-safe (LangGraph's ``graph.invoke`` is safe to call from a
            background thread as long as no shared mutable state is accessed
            concurrently outside the graph).
        barge_in: If True (default), the microphone stays live during TTS
            playback instead of being muted, and speech detected while TTS is
            speaking fires ``on_barge_in`` immediately (see below) so the
            caller can cut playback short. Set False to restore the old
            fully-muted-during-TTS behaviour.
    """

    def __init__(
        self,
        callback: Callable[[ASRResult], None] | None = None,
        barge_in: bool = True,
    ) -> None:
        self.vad_streamer = SileroVADStreamer(
            sample_rate=settings.sample_rate,
            min_silence_duration_ms=settings.silence_duration_ms,
            on_speech_start=self._on_speech_start,
        )
        self.router = ASRRouter()
        self.utterance_queue: queue.Queue = queue.Queue()
        self.is_running = False
        self._consumer_thread: threading.Thread | None = None

        # Downstream callback — set to None to use the default stdout printer
        self.callback: Callable[[ASRResult], None] | None = callback

        # Mute flag — when set the VAD producer discards utterances instead of
        # queuing them.  Use mute()/unmute() to pause listening during TTS
        # playback so the speaker output is never fed back as a user utterance.
        # Only used as a hard fallback (barge_in=False) — see begin_tts()/
        # end_tts() for the barge-in-aware path.
        self._muted = threading.Event()

        # --- Barge-in support ---------------------------------------------
        # NOTE: there is no acoustic echo cancellation here — with no headset
        # the assistant's own speaker output can itself trigger the VAD and
        # look like a barge-in. Reliable barge-in assumes a headset/earbuds,
        # same as most barge-in-capable voice assistants without hardware
        # AEC. mute()/unmute() (barge_in=False) remains available as a safe
        # fallback for speaker-only setups.
        self.barge_in_enabled = barge_in
        # Set for the duration of a TTS turn; the VAD producer thread checks
        # this (via _on_speech_start) to know whether new speech counts as a
        # barge-in.
        self.tts_active = threading.Event()
        # Caller-supplied hook invoked (from the VAD producer thread) the
        # instant speech is detected while tts_active is set. Should be cheap
        # and non-blocking — typically just `threading.Event.set()` on the
        # TTS stop_event to cut playback short.
        self.on_barge_in: Callable[[], None] | None = None
        # Guards against firing on_barge_in more than once per TTS turn.
        self._barge_in_fired = threading.Event()

        # Last utterance audio tensor — cached so the callback (run_voice.py) can
        # re-invoke the router with a different language override without re-reading
        # the microphone.  Set to None until the first utterance is processed.
        self.last_audio_tensor: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Barge-in (TTS-active window)
    # ------------------------------------------------------------------

    def begin_tts(self) -> None:
        """Call right before starting TTS playback for a turn.

        Arms barge-in detection: any speech onset from here until
        :meth:`end_tts` is reported via ``on_barge_in``.
        """
        self._barge_in_fired.clear()
        self.tts_active.set()

    def end_tts(self) -> None:
        """Call after TTS playback ends (naturally or via barge-in)."""
        self.tts_active.clear()

    def _on_speech_start(self) -> None:
        """VAD callback: fired the moment speech crosses the threshold.

        Runs on the VAD producer thread. Only acts while a TTS turn is
        active and barge-in is enabled; fires at most once per TTS turn.
        """
        if not (self.barge_in_enabled and self.tts_active.is_set()):
            return
        if self._barge_in_fired.is_set():
            return
        self._barge_in_fired.set()
        print("[Pipeline] Barge-in: speech detected during TTS playback — interrupting.")
        if self.on_barge_in is not None:
            try:
                self.on_barge_in()
            except Exception as e:
                print(f"[Pipeline] on_barge_in callback error: {e}")

    # ------------------------------------------------------------------
    # Mute / unmute (TTS playback guard)
    # ------------------------------------------------------------------

    def mute(self) -> None:
        """Stop accepting new utterances (call before TTS playback starts).

        Any utterance that has *already* been queued before mute() is called
        will still be processed by the consumer; the mute only blocks new ones
        from entering the queue.
        """
        self._muted.set()
        print("[Pipeline] Microphone muted (TTS playing).")

    def unmute(self) -> None:
        """Resume accepting utterances (call after TTS playback ends).

        Drains the utterance queue of any captures that slipped in during
        the muted window (e.g. TTS audio echo, room noise after the speaker
        stopped) before re-enabling the consumer.
        """
        # Drain residual captures accumulated while muted
        drained = 0
        while True:
            try:
                self.utterance_queue.get_nowait()
                self.utterance_queue.task_done()
                drained += 1
            except queue.Empty:
                break
        if drained:
            print(f"[Pipeline] Drained {drained} residual utterance(s) after TTS.")
        self._muted.clear()
        print("[Pipeline] Microphone unmuted (listening).")

    # ------------------------------------------------------------------
    # Consumer (background thread)
    # ------------------------------------------------------------------

    def _consumer_loop(self) -> None:
        """Background thread: consume utterances from the VAD queue and route to ASR."""
        while self.is_running:
            try:
                utterance = self.utterance_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                # ----------------------------------------------------------
                # Minimum-length guard
                # Short audio bursts (<~0.5 s) are almost always noise or
                # TTS echo that slipped through before the mute was applied.
                # Whisper hallucinates "." or random words on them — skip.
                # ----------------------------------------------------------
                if len(utterance) < _MIN_UTTERANCE_SAMPLES:
                    print(
                        f"[Pipeline] Utterance too short "
                        f"({len(utterance)} samples < {_MIN_UTTERANCE_SAMPLES}) — skipping."
                    )
                    continue

                audio_tensor = torch.from_numpy(utterance)
                # Cache for potential IndicConformer retry in run_voice.py
                self.last_audio_tensor = audio_tensor
                print(f"\n[Pipeline] Processing utterance of {len(utterance)} samples...")

                start_time = time.time()
                result = self.router.route_and_transcribe(audio_tensor)
                duration = time.time() - start_time

                print(f"[STT Result - {duration:.2f}s]")
                print(f"  Language : {result.detected_language}")
                print(f"  Model    : {result.model_used}")
                print(f'  Text     : "{result.raw_text}"')

                # Dispatch to downstream consumer (LangGraph or stdout)
                if self.callback is not None:
                    self.callback(result)

            except Exception as e:
                print(f"[Pipeline] Error transcribing utterance: {e}")
            finally:
                self.utterance_queue.task_done()


    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the STT pipeline.  Blocks on the calling thread (VAD producer loop)."""
        self.is_running = True

        self._consumer_thread = threading.Thread(
            target=self._consumer_loop, daemon=True, name="stt-consumer"
        )
        self._consumer_thread.start()

        print("[Pipeline] Starting VAD Producer on main thread...")
        try:
            for utterance in self.vad_streamer.stream():
                # Discard utterances while muted (TTS is playing)
                if self._muted.is_set():
                    continue
                self.utterance_queue.put(utterance)
        except KeyboardInterrupt:
            self.stop()

    def stop(self) -> None:
        """Stop the STT pipeline and wait for the consumer thread to exit."""
        print("\n[Pipeline] Stopping STT Pipeline...")
        self.is_running = False
        if self._consumer_thread:
            self._consumer_thread.join(timeout=2.0)
        self.router.reset_session()


# ---------------------------------------------------------------------------
# Standalone demo — run with: python -m vay.audio.pipeline
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pipeline = STTPipeline()   # no callback → results printed to stdout
    pipeline.start()
