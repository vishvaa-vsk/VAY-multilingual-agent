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

## 6. Addendum (same day) — language coverage + prompt alignment

Follow-up questions after the initial fix: does the chunker work for *all* 18
TTS languages, and does the LLM's `final_reply` text actually produce output
the chunker (and TTS in general) handles well?

**Chunker coverage gap found and fixed** (`src/vay/tts/engine.py`):
`_SENTENCE_SPLIT_RE` only recognized Latin `.!?`, Devanagari danda `। ॥`, and
CJK fullwidth punctuation — missing Arabic's `؟` (question mark) and Urdu's
own full stop `۔` (distinct from the Arabic one), both used by the `ar`/`ur`
voices. Added both. New tests: `test_arabic_question_mark_boundary_is_split`,
`test_urdu_full_stop_boundary_is_split`. Every one of the 18 `VOICES`
languages now has its natural sentence-ending punctuation recognized. (When a
reply genuinely has no recognizable boundary, the whole text is still spoken
correctly as one chunk — just without the pipelining benefit, never a
failure.)

**LLM output alignment fixed** (`src/vay/graph/core_utils.py`,
`SUBAGENT_SYSTEM_PROMPT_TEMPLATE` — the single prompt every spoken reply is
generated from, including `closing_node`'s goodbye line, which reuses the
same template): the LLM's replies weren't guaranteed to produce text that's
good to *speak*, in three concrete ways a caller would actually notice:

1. **Raw table pipes could leak into speech.** The KB documents
   (`data/kb/*.md`) are markdown tables; a sub-agent could echo a `|`-
   delimited row instead of summarizing it. Rule 11 now explicitly forbids
   the `|` character and raw tables/bullets, with a worked example showing
   the required rephrasing.
2. **Rates were written with a slash** (e.g. "2GB/day" — the plan-listing
   instruction's own worked example did this). Rule 11 now requires "2 GB
   per day" / "per month" / "per line" instead, and the `/` character is
   banned from the spoken reply outright. Fixed both the new rule and the
   pre-existing example text at the plan-listing instruction.
3. **Dates could come out in raw ISO/tool format** (e.g. `2025-08-15` from a
   DB column) instead of a natural spoken date. New Rule 12 requires
   converting any numeric/ISO date to a natural spoken form per language —
   "15th August 2025" in English, and the equivalent non-ordinal form in
   other languages (e.g. Hindi "15 अगस्त 2025", Tamil "15 ஆகஸ்ட் 2025").
4. **Tone strengthened + terminal-punctuation requirement added.** Rule 11
   now also requires every sentence to end with proper terminal punctuation
   for the reply's language — this isn't just cosmetic, it's what the
   sentence-chunker above actually keys off of, so a reply that respects
   this rule gets the full pipelining benefit; one that doesn't still plays
   correctly, just as a single unsplit chunk.

Renumbered the old "12. ANTI-REPETITION" rule to 13 to make room; its content
is unchanged.

**Not touched:** `ORCHESTRATOR_SYSTEM_PROMPT` (JSON-only, never spoken), and
every fixed/deterministic template (`HANDOFF_MESSAGE_TEMPLATES`,
`CLARIFY_TEMPLATES`, `CHITCHAT_TEMPLATES`, `AGGRESSIVE_WARNING_TEMPLATES`,
`CALL_CUT_TEMPLATES`, consent scripts) — these are hand-written per-language
strings by design, not LLM output, so there's nothing for a prompt rule to
fix there.

**Verification:** `uv run pytest tests/ -v` → **18/18 passed** (16 baseline
+ 2 new punctuation-coverage tests). `ruff check` on the touched files shows
only pre-existing long-line (E501) warnings already present in
`core_utils.py` before this change (the file's system-prompt strings were
already over the 100-char line limit throughout) — no new lint issues
introduced. The prompt-wording changes themselves need a live
`GROQ_API_KEY`-backed run to confirm the model actually complies (prompt
instructions are a strong nudge, not a hard guarantee) — recommend spot-
checking a plan-listing question (`/day` phrasing) and a date-bearing
question (ticket/due-date lookup) in a couple of languages via
`uv run python scripts/run_assistant.py --show_debug`.

## 7. Possible follow-ups (not done here, flagged for the team)

- If further latency reduction is needed beyond this, the next real lever is
  raw-PCM streaming playback (see §3) — bigger change, bigger win.
- `_MIN_CHARS_FOR_CHUNKING` (120 chars) and the sentence-split regex are
  reasonable defaults but not empirically tuned against real call transcripts
  yet — worth revisiting once there's real latency telemetry from live calls.
- No changes were made to how long the LLM/guardrail/tool-loop stages take —
  if the user later observes those regressing again, that's a separate
  investigation (see `dev-context.md` for the existing latency fixes already
  made there, e.g. the double-Groq-call fix in ASR routing).
