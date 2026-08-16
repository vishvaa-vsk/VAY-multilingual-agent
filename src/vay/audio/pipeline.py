"""Audio STT pipeline manager (Producer-Consumer architecture)."""

import queue
import threading
import time

import torch

from vay.asr.router import ASRRouter
from vay.audio.vad import SileroVADStreamer
from vay.config import settings


class STTPipeline:
    """Manages the lifecycle of real-time audio chunking, language ID, and STT routing."""

    def __init__(self) -> None:
        self.vad_streamer = SileroVADStreamer(
            sample_rate=settings.sample_rate,
            min_silence_duration_ms=settings.silence_duration_ms,
        )
        self.router = ASRRouter()
        self.utterance_queue: queue.Queue = queue.Queue()
        self.is_running = False
        self._consumer_thread: threading.Thread | None = None

    def _consumer_loop(self) -> None:
        """Background thread consuming utterances from VAD and routing to ASR."""
        while self.is_running:
            try:
                # Wait for utterance with a timeout to allow graceful shutdown
                utterance = self.utterance_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            # We got an utterance. Send it to the router!
            try:
                # Convert to torch tensor
                tensor_chunk = torch.from_numpy(utterance)
                
                print(f"[Pipeline] Processing utterance of {len(utterance)} samples...")
                
                # Execute STT route
                start_time = time.time()
                result = self.router.route_and_transcribe(tensor_chunk)
                duration = time.time() - start_time
                
                print(f"\n[STT Result - {duration:.2f}s]")
                print(f"Language: {result.detected_language}")
                print(f"Model: {result.model_used}")
                print(f"Text: \"{result.raw_text}\"\n")
                
            except Exception as e:
                print(f"[Pipeline] Error transcribing utterance: {e}")
            finally:
                self.utterance_queue.task_done()

    def start(self) -> None:
        """Start the STT pipeline."""
        self.is_running = True
        
        # Start Consumer Thread
        self._consumer_thread = threading.Thread(target=self._consumer_loop, daemon=True)
        self._consumer_thread.start()
        
        # Run Producer Loop (blocking on main thread)
        print("[Pipeline] Starting VAD Producer on main thread...")
        try:
            for utterance in self.vad_streamer.stream():
                # Push utterance to queue without blocking
                self.utterance_queue.put(utterance)
        except KeyboardInterrupt:
            self.stop()

    def stop(self) -> None:
        """Stop the STT pipeline."""
        print("\n[Pipeline] Stopping STT Pipeline...")
        self.is_running = False
        if self._consumer_thread:
            self._consumer_thread.join(timeout=2.0)
        self.router.reset_session()


if __name__ == "__main__":
    pipeline = STTPipeline()
    pipeline.start()
