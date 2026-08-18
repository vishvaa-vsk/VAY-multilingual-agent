"""Voice Activity Detection (VAD) streaming using Silero VAD and sounddevice."""

import queue
import time
from typing import Any, Callable, Generator

import numpy as np
import sounddevice as sd
import torch


class SileroVADStreamer:
    """Streams audio from microphone and yields complete utterances based on VAD."""

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 512,
        threshold: float = 0.5,
        min_silence_duration_ms: int = 700,
        pre_speech_buffer_ms: int = 300,
        on_speech_start: "Callable[[], None] | None" = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.threshold = threshold
        self.min_silence_duration_ms = min_silence_duration_ms
        self.pre_speech_buffer_ms = pre_speech_buffer_ms
        # Fired the instant speech crosses the VAD threshold — i.e. *before*
        # the full utterance (and its trailing silence) has been captured.
        # Used for barge-in: the caller needs to know "user started talking"
        # immediately, not ~700ms+ later when the utterance finally yields.
        self.on_speech_start = on_speech_start

        # Load the model
        print("Loading Silero VAD model...")
        self.model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False
        )
        self.model.eval()
        
        self.q: queue.Queue = queue.Queue()
        self.is_speaking = False
        
        # Buffers
        self.ring_buffer_size = int((self.pre_speech_buffer_ms / 1000.0) * self.sample_rate / self.chunk_size)
        self.ring_buffer: list[np.ndarray] = []
        self.current_utterance: list[np.ndarray] = []
        self.silence_start_time = 0.0

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: Any, status: sd.CallbackFlags) -> None:
        """Callback for sounddevice to capture audio chunks."""
        if status:
            print(f"SoundDevice Status: {status}")
        # Copy the data and put it in the queue
        self.q.put(indata.copy())

    def stream(self) -> Generator[np.ndarray, None, None]:
        """Start microphone stream and yield complete utterances."""
        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            blocksize=self.chunk_size,
            callback=self._audio_callback
        )
        
        print("Starting VAD stream. Speak into the microphone...")
        with stream:
            while True:
                # Get audio chunk from queue
                chunk = self.q.get()
                
                # Convert to torch tensor for Silero
                tensor_chunk = torch.from_numpy(chunk).squeeze()
                
                # Get speech probability
                with torch.no_grad():
                    speech_prob = self.model(tensor_chunk, self.sample_rate).item()

                if not self.is_speaking:
                    # WAITING STATE
                    self.ring_buffer.append(chunk)
                    if len(self.ring_buffer) > self.ring_buffer_size:
                        self.ring_buffer.pop(0)
                        
                    if speech_prob > self.threshold:
                        # Speech started!
                        print("\n[VAD] Speech detected! Recording...")
                        self.is_speaking = True
                        self.current_utterance = self.ring_buffer.copy()
                        self.ring_buffer.clear()
                        if self.on_speech_start is not None:
                            try:
                                self.on_speech_start()
                            except Exception as e:
                                print(f"[VAD] on_speech_start callback error: {e}")
                else:
                    # RECORDING STATE
                    self.current_utterance.append(chunk)
                    
                    if speech_prob < self.threshold:
                        # Silence detected, but we wait for min_silence_duration
                        if self.silence_start_time == 0.0:
                            self.silence_start_time = time.time()
                        elif (time.time() - self.silence_start_time) * 1000 > self.min_silence_duration_ms:
                            # Silence duration exceeded, utterance complete
                            print("[VAD] Utterance complete!")
                            utterance_audio = np.concatenate(self.current_utterance)
                            
                            # Reset state
                            self.is_speaking = False
                            self.current_utterance = []
                            self.silence_start_time = 0.0
                            
                            # MUST reset model state after utterance, otherwise GRU memory gets stuck
                            self.model.reset_states()
                            
                            yield utterance_audio
                    else:
                        # Still speaking, reset silence timer
                        self.silence_start_time = 0.0

