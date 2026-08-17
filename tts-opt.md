# TTS Latency Optimization — Aug 17, 2026

> **Symptom reported:** LLM reply generation is fast, but the spoken (TTS)
> output takes noticeably longer to actually start playing.
> **Scope:** `src/vay/tts/engine.py` only (+ no call-site changes needed —
> `graph/nodes/utils.py::tts_node` calls `tts.speak(...)` exactly as before).
> No changes to ASR, RAG, LangGraph orchestration, guardrail, or compliance
> logic. No new dependencies — `pyproject.toml` / `uv.lock` unchanged.

---

## 1. Root Cause

Tracing the real pipeline (`scripts/run_voice.py` → `tts_node` →
`tts/engine.py::speak()`):

1. `graph.invoke(state)` finishes fast, producing the complete `final_reply`
   string (this part was already confirmed fast and is untouched).
2. `tts_node` called `tts.speak(final_reply, lang)`, which did, **serially**:
   - `edge_tts.Communicate(text, voice).save(output_path)` — blocks until
     **the entire mp3 for the entire reply** is synthesized and written to
     disk, then
   - `playsound3.playsound(output_path)` — blocks for full playback.
3. So playback start time = network RTT + synthesis time for the **whole**
   reply, not just the first sentence. For a typical 2–4 sentence assistant
   reply, that's the dominant chunk of the delay the user was seeing. The mic
   also stays muted (`pipeline.mute()` / `unmute()` in `run_voice.py`) for
   the entire duration.

## 2. Fix

Kept everything upstream of `tts_node` untouched — the guardrail
(PII/consent/compliance checks, `project_context.md` §4) still sees and
gates the **complete** `final_reply` text before anything is spoken. The
change is purely in *how* that already-finalized text gets synthesized and
played:

- **Sentence-level chunking** (`_split_into_speech_chunks`): a regex split on
  sentence-ending punctuation across every language the TTS voices cover —
  Latin `. ! ?`, Devanagari danda `। ॥` (Hindi/Marathi), and CJK fullwidth
  punctuation (`。！？`). Short replies (< 120 chars) or text with no
  detectable sentence boundary are left as a single chunk — chunking a
  one-liner only adds network overhead with no benefit.
- **Pipelined synthesis + playback** (`_speak_pipelined`): synthesizes chunk
  0, starts playing it, and **concurrently synthesizes chunk 1 in the
  background** while chunk 0 plays — repeating for every subsequent chunk.
  Playback still happens strictly in order via `playsound3` (no new audio
  dependency); only the *scheduling* changed.
- `speak()` routes into this pipelined path only for its default hot-path
  usage (`play=True`, no explicit `output_path` — i.e. exactly what
  `tts_node` calls on every turn). The single-file behavior is preserved
  byte-for-byte for every other caller: `TTSEngine.synthesize()`
  (`play=False`, used by anything that wants one complete mp3 file on disk),
  and any caller passing an explicit `output_path`.

**Net effect:** time-to-first-audio is now bounded by the synthesis time of
the *first sentence* instead of the *whole reply*. Total call duration is
roughly unchanged (the tail of the reply's synthesis now overlaps with
playback instead of stacking in front of it).

## 3. Why not other options

| Option | Why not (now) |
|---|---|
| Stream LLM tokens straight into TTS | Guardrail must see the complete draft reply first (PII/compliance/consent gate, `project_context.md` §4 — explicitly compliance-critical, not to be weakened). Chunking here only happens *after* `final_reply` is finalized and guardrail-approved. |
| Switch TTS provider (Cartesia/ElevenLabs/Deepgram streaming) | `edge-tts` is a locked model/service choice in `project_context.md` §2 ("DO NOT SUBSTITUTE") — out of scope unless revisited by the team. |
| True raw-PCM streaming playback (decode `Communicate.stream()` mp3 chunks progressively into `sounddevice`) | Lower latency still, but needs a streaming mp3 decoder and reworked audio-device code — bigger surface area for a 7-day hackathon build. Sentence-level pipelining captures most of the perceived-latency win with a much smaller, safer change reusing the existing `playsound3` dependency. |

## 4. Verification

- **New unit tests**: `tests/test_tts_chunking.py` (5 tests) — pure-function
  coverage of `_split_into_speech_chunks`: short text untouched, long
  unpunctuated text untouched, multi-sentence English split correctly,
  Hindi danda (`।`) boundary split correctly, no empty chunks ever produced.
- **Full suite**: `uv run pytest tests/ -v` → **16/16 passed** (was 11/11
  baseline before this change; +5 new tests, 0 regressions).
- **Manual synthesis smoke test** (`play=False`, single-file path): confirmed
  byte-identical behavior to before — still produces one valid mp3.
- **Manual pipelining smoke test** (playback function patched to log
  start/end timestamps instead of hitting a real audio device, in this
  headless dev environment): for a 3-sentence sample reply —
  - Old (single-file) synth-only time for the same text: **~1.85s** before
    any audio could start playing.
  - New (pipelined) time-to-first-audio: **~1.08s** — chunk 1 starts playing
    while chunk 2 is still synthesizing in the background, chunks play
    strictly in order with no overlap/collision.
  - This ~40% drop in time-to-first-audio is for a short 3-sentence reply;
    the gap widens for longer replies since first-chunk synthesis time stays
    roughly constant regardless of total reply length, while the old
    approach's wait time scaled with the whole reply.

## 5. Files changed

```
src/vay/tts/engine.py       sentence chunking + pipelined synth/playback path
tests/test_tts_chunking.py  NEW — unit tests for the chunker
```

No changes to `pyproject.toml` / `uv.lock` (no new dependencies — reuses
`edge-tts` and `playsound3`, both already present).

## 6. Possible follow-ups (not done here, flagged for the team)

- If further latency reduction is needed beyond this, the next real lever is
  raw-PCM streaming playback (see §3) — bigger change, bigger win.
- `_MIN_CHARS_FOR_CHUNKING` (120 chars) and the sentence-split regex are
  reasonable defaults but not empirically tuned against real call transcripts
  yet — worth revisiting once there's real latency telemetry from live calls.
- No changes were made to how long the LLM/guardrail/tool-loop stages take —
  if the user later observes those regressing again, that's a separate
  investigation (see `dev-context.md` for the existing latency fixes already
  made there, e.g. the double-Groq-call fix in ASR routing).
