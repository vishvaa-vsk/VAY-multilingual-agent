# Speech-to-Text (STT) and ASR Pipeline

This document is a technical study and reference guide for the Speech-to-Text (STT), Automatic Speech Recognition (ASR), Voice Activity Detection (VAD), and real-time Barge-In (Interruption Handling) subsystems in VAY.

---

## 1. Subsystem Architecture

The voice intake pipeline coordinates real-time microphone capture, voice activity boundary detection, early speech interruption (barge-in), and dual-tier ASR transcription.

```mermaid
flowchart TD
    subgraph CaptureAndVAD ["1. Capture & VAD (vad.py)"]
        Mic([Microphone Audio 16kHz]) --> Streamer[SileroVADStreamer: blocksize=512]
        Streamer --> ModelInfer[Silero VAD snakers4/silero-vad: speech_prob]
        
        ModelInfer --> PreBuffer[Ring Buffer: 300ms pre-speech audio]
        ModelInfer --> SpeechDetect{speech_prob > threshold 0.5?}
        
        SpeechDetect -->|Speech Onset| OnSpeechStart[on_speech_start Hook]
        OnSpeechStart --> TTSActiveCheck{tts_active is set & barge_in enabled?}
        TTSActiveCheck -->|Yes: Barge-In Event| FireBargeIn["on_barge_in() -> set TTS stop_event<br/>(Immediately cuts playsound3)"]
        TTSActiveCheck -->|No| Accumulate[Accumulate Utterance Frames]
        FireBargeIn --> Accumulate
        
        ModelInfer --> SilenceCheck{Silence Duration > 700ms?}
        SilenceCheck -->|Utterance Boundary| YieldUtterance[Concatenate Utterance & Reset Model States]
    end

    subgraph PipelineQueue ["2. Producer-Consumer Pipeline (pipeline.py)"]
        YieldUtterance --> UtteranceQueue[utterance_queue.put]
        UtteranceQueue --> ConsumerLoop[Background Thread: _consumer_loop]
        
        ConsumerLoop --> LenCheck{Samples >= _MIN_UTTERANCE_SAMPLES<br/>len >= 8000 ~0.5s?}
        LenCheck -->|No: Mic Click / Spurious Noise| SkipUtterance[Discard Utterance]
        LenCheck -->|Yes| CacheTensor["Cache self.last_audio_tensor<br/>Convert to torch.Tensor"]
    end

    subgraph RoutingAndASR ["3. Dual-Tier Routing (router.py, whisper.py, indic.py)"]
        CacheTensor --> WhisperAuto["WhisperASR.transcribe_auto(audio_tensor)<br/>Single Groq API call (verbose_json)"]
        WhisperAuto --> ParseResp["Extract detected_language & raw_text<br/>Calculate confidence from segment logprobs<br/>Filter hallucinations & dedup repeated words"]
        
        ParseResp --> TierCheck{detected_language in tier1_languages?<br/>22 Indian Languages}
        
        TierCheck -->|Yes: Indic Language| IndicRun["IndicConformerASR.transcribe(audio_tensor, lang)<br/>AutoModel rnnt decoding"]
        TierCheck -->|No: English / Global| ReturnWhisper[Use Whisper ASRResult]
        
        IndicRun --> IndicEmptyCheck{IndicConformer raw_text empty?}
        IndicEmptyCheck -->|Yes / Fallback| ReturnWhisper
        IndicEmptyCheck -->|No| ReturnIndic[Use IndicConformer ASRResult]
    end

    subgraph DownstreamSession ["4. Voice Session Dispatch (run_voice.py)"]
        ReturnWhisper --> Callback[STTPipeline.callback -> on_asr_result]
        ReturnIndic --> Callback
        
        Callback --> PunctuationFilter{_NOISE_TRANSCRIPT_RE match?<br/>Pure whitespace/punctuation}
        PunctuationFilter -->|Yes| DropNoise[Discard Hallucinated Noise Turn]
        
        PunctuationFilter -->|No| LowConfRetry{Whisper Conf < 0.50 & Tier-2<br/>AND session.preferred_language in Tier-1?}
        LowConfRetry -->|Yes| RetryIndic["Re-invoke router.route_and_transcribe<br/>(last_audio_tensor, override_language=pref)"]
        RetryIndic --> InvokeGraph[Build GraphState & Invoke LangGraph]
        LowConfRetry -->|No| InvokeGraph
    end
```

---

## 2. Voice Activity Detection (VAD) & Real-Time Barge-In

### 2.1 Implementation Details
**Primary Code Reference:** [`src/vay/audio/vad.py`](file:///home/vishvaa/Projects/VAY-multilingual-agent/src/vay/audio/vad.py)

The [`SileroVADStreamer`](file:///home/vishvaa/Projects/VAY-multilingual-agent/src/vay/audio/vad.py#L12-L126) class reads continuous audio from the microphone and yields complete speech segments.

```python
# Code snippet from src/vay/audio/vad.py
class SileroVADStreamer:
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 512,
        threshold: float = 0.5,
        min_silence_duration_ms: int = 700,
        pre_speech_buffer_ms: int = 300,
        on_speech_start: "Callable[[], None] | None" = None,
    ) -> None:
        self.model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False
        )
        self.model.eval()
```

- **Audio Specs**: 16,000 Hz, 1-channel mono, 32-bit floating point PCM via `sounddevice.InputStream`.
- **Chunk Evaluation**: Evaluates non-overlapping chunks of 512 samples (~32 ms at 16 kHz).
- **Pre-Speech Buffer**: A 300 ms circular ring buffer preserves vocal onset frames preceding the threshold trigger.
- **Utterance Boundary Detection**: An active utterance completes when speech probability remains below `threshold = 0.50` for `min_silence_duration_ms = 700 ms`.
- **GRU State Reset**: Calls `self.model.reset_states()` after every utterance to prevent recurrent hidden-state drift across conversational turns.

### 2.2 Real-Time Barge-In Mechanics
**Primary Code Reference:** [`src/vay/audio/pipeline.py`](file:///home/vishvaa/Projects/VAY-multilingual-agent/src/vay/audio/pipeline.py#L109-L142)

```python
# Code snippet from src/vay/audio/pipeline.py
def _on_speech_start(self) -> None:
    if not (self.barge_in_enabled and self.tts_active.is_set()):
        return
    if self._barge_in_fired.is_set():
        return
    self._barge_in_fired.set()
    print("[Pipeline] Barge-in: speech detected during TTS playback — interrupting.")
    if self.on_barge_in is not None:
        self.on_barge_in()
```

- **Early Vocal Trigger**: `on_speech_start` fires immediately when `speech_prob > 0.50` without waiting for trailing silence.
- **Armed Window**: `begin_tts()` sets `tts_active = True` during TTS synthesis and playback; `end_tts()` clears it.
- **Playback Interruption**: When speech begins during playback, `on_barge_in()` sets the `stop_event` on the running TTS thread, terminating the `playsound3` audio playback process within ~50 ms.
- **Acoustic Fallback**: For speaker setups without headsets, pass `--no_barge_in` in [`scripts/run_voice.py`](file:///home/vishvaa/Projects/VAY-multilingual-agent/scripts/run_voice.py) to enable hard `mute()`/`unmute()` with 400 ms room-settling delay.

---

## 3. Producer-Consumer Pipeline Execution

**Primary Code Reference:** [`src/vay/audio/pipeline.py`](file:///home/vishvaa/Projects/VAY-multilingual-agent/src/vay/audio/pipeline.py#L41-L267)

The [`STTPipeline`](file:///home/vishvaa/Projects/VAY-multilingual-agent/src/vay/audio/pipeline.py#L41) manages asynchronous audio capture and worker transcription:

```python
# Code snippet from src/vay/audio/pipeline.py
def _consumer_loop(self) -> None:
    while self.is_running:
        utterance = self.utterance_queue.get(timeout=1.0)
        if len(utterance) < _MIN_UTTERANCE_SAMPLES: # 8,000 samples (~0.5s)
            continue
        audio_tensor = torch.from_numpy(utterance)
        self.last_audio_tensor = audio_tensor
        result = self.router.route_and_transcribe(audio_tensor)
        if self.callback is not None:
            self.callback(result)
```

- **Noise Filtering (`_MIN_UTTERANCE_SAMPLES = 8000`)**: Discards short noise clicks and echo bleeds before model invocation.
- **Audio Tensor Caching**: Retains `self.last_audio_tensor` to support downstream model retry passes.

---

## 4. Dual-Tier ASR Router & Zero-Overhead LID

**Primary Code Reference:** [`src/vay/asr/router.py`](file:///home/vishvaa/Projects/VAY-multilingual-agent/src/vay/asr/router.py)

```python
# Code snippet from src/vay/asr/router.py
def route_and_transcribe(self, audio_tensor: torch.Tensor, override_language: str | None = None) -> ASRResult:
    # Step 1: Single-pass transcription & LID via Whisper
    whisper_result = self.whisper_asr.transcribe_auto(audio_tensor)
    detected_lang = whisper_result.detected_language

    # Step 2: Route according to language tier
    if detected_lang in settings.tier1_languages:
        indic_result = self.indic_asr.transcribe(audio_tensor, language=detected_lang)
        if not indic_result.raw_text.strip():
            return whisper_result # Fallback if IndicConformer returns empty
        return indic_result
    else:
        return whisper_result
```

### 4.1 Language Tier Matrix

| Tier | Supported Languages | Model Identifier | Execution Backend |
|---|---|---|---|
| **Tier 1** | 22 Scheduled Indian Languages (`ta`, `hi`, `te`, `kn`, `ml`, `mr`, `gu`, `bn`, `pa`, `or`, `as`, `ur`, etc.) | `ai4bharat/indic-conformer-600m-multilingual` | PyTorch `AutoModel` (RNN-T Decoding) |
| **Tier 2** | English (`en`) + 90 Global Languages | `openai/whisper-large-v3-turbo` | Groq Cloud API (`verbose_json`) |

---

## 5. Model Wrappers & Execution Specifics

### 5.1 IndicConformer Wrapper
**Primary Code Reference:** [`src/vay/asr/indic.py`](file:///home/vishvaa/Projects/VAY-multilingual-agent/src/vay/asr/indic.py)

```python
# Code snippet from src/vay/asr/indic.py
from transformers import AutoModel

self.model = AutoModel.from_pretrained(
    "ai4bharat/indic-conformer-600m-multilingual",
    trust_remote_code=True,
    token=os.environ.get("HF_TOKEN")
)
output = self.model(audio_tensor.view(1, -1), language, "rnnt")
```

> [!IMPORTANT]
> **HuggingFace Pipeline Gotcha**: Do not use `transformers.pipeline()` with IndicConformer because its custom configuration classes fail under generic pipeline factories. Direct `AutoModel` invocation is required.

### 5.2 Whisper Wrapper & Zero-Overhead LID
**Primary Code Reference:** [`src/vay/asr/whisper.py`](file:///home/vishvaa/Projects/VAY-multilingual-agent/src/vay/asr/whisper.py)

```python
# Code snippet from src/vay/asr/whisper.py
response = self.client.audio.transcriptions.create(
    file=("audio.wav", wav_bytes),
    model="whisper-large-v3-turbo",
    response_format="verbose_json", # Returns text + language in one call
)
```

- **Single-Pass Auto-Transcription**: `transcribe_auto()` eliminates the separate LID API call, reducing turn latency from ~1.8s to ~0.6s.
- **Language Normalization (`_normalise_language`)**: Maps 60+ English language names (e.g. `"tamil"`, `"hindi"`) to validated ISO 639-1 tags (`_GROQ_VALID_LANGUAGE_CODES`).
- **Hallucination Blacklist (`src/vay/asr/hallucinations.py`)**: Filters repeated tokens and known subtitle hallucinations (`"Thank you for watching"`, `"Subtitles by"`).

---

## 6. Downstream Voice Session Recovery

**Primary Code Reference:** [`scripts/run_voice.py`](file:///home/vishvaa/Projects/VAY-multilingual-agent/scripts/run_voice.py)

1. **Noise Transcript Filter**: Evaluates `_NOISE_TRANSCRIPT_RE = r'^[\s.,!?…\-–—\'\"]+$'` to drop pure punctuation transcripts.
2. **Account-Aware Low-Confidence Fallback**:
   If Whisper detects a Tier-2 language with confidence < 0.50, but the customer profile ([`SessionContext.preferred_language`](file:///home/vishvaa/Projects/VAY-multilingual-agent/src/vay/tools/session.py)) is an Indic Tier-1 language, the router re-runs `router.route_and_transcribe(last_audio_tensor, override_language=pref_lang)` through IndicConformer.

---

## 7. Performance & Accuracy Benchmarks

Evaluated on 200 Mozilla Common Voice test samples per language:

| Language | Test Samples | Whisper WER (%) | Whisper CER (%) | Whisper Avg Time (s) | IndicConformer WER (%) | IndicConformer CER (%) | IndicConformer Avg Time (s) |
|---|---|---|---|---|---|---|---|
| **Tamil** | 200 | 62.44% | 17.87% | 0.88s | **26.06%** | **5.52%** | 1.65s |
| **Hindi** | 200 | 35.10% | 17.60% | 0.60s | **12.00%** | **6.30%** | 1.12s |
| **English** | 200 | **3.79%** | **7.09%** | **0.32s** | N/A | N/A | N/A |

### Empirical Insights:
- **IndicConformer Superiority on Indian Languages**: Delivers a **58% relative WER reduction in Tamil** (26.06% vs. 62.44%) and **66% relative WER reduction in Hindi** (12.00% vs. 35.10%).
- **Whisper Superiority on English**: Fast API inference (0.32s) with 3.79% WER.
