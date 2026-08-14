"""
content_manager.py

CRUD layer on top of chroma_setup.py.

  CREATE  create(source)   -> admin gives a URL or a PDF path.
                              Auto-detected, converted to Markdown, saved to
                              disk, chunked, embedded, and stored in ChromaDB.
  READ    read(query)      -> semantic search over stored chunks.
          list_sources()   -> every distinct source currently in the DB.
  UPDATE  update(source)   -> deletes the source's old chunks, re-ingests it.
                              Use after the underlying page/PDF has changed.
  DELETE  delete(source)   -> removes every chunk belonging to a source.

CLI:
    python content_manager.py create "https://example.com/article"
    python content_manager.py create "./reports/document.pdf"
    python content_manager.py create "https://example.com" --labels billing plans network
    python content_manager.py read "how do I reset my sim" --n-results 5
    python content_manager.py update "https://example.com/article"
    python content_manager.py delete "https://example.com/article"
    python content_manager.py list
    python content_manager.py inspect "https://example.com/article"
    python content_manager.py categories

Domain-agnostic design: works on any knowledge base (telecom, medical, legal,
HR, finance, …) without editing source code. Category labels are either
auto-discovered from the content (default) or supplied by the caller via
--labels / the `category_labels` argument.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Dynamic import for 'chroma_setup (1).py' — the (1) suffix prevents normal
# `import chroma_setup` from working, so we load it explicitly by file path
# and register it under the canonical name so downstream `from chroma_setup
# import …` statements work without modification.
# ---------------------------------------------------------------------------


def chunk_markdown(
    markdown: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[tuple[str, str]]:
    """
    Structure-aware, sentence-boundary chunking.

    Returns a list of (chunk_text, heading_context) tuples.

    Fixes applied:
      - FIX 5: Splits are at sentence boundaries (nltk.sent_tokenize) — no mid-sentence cuts.
      - FIX 6: Last seen Markdown heading is propagated into orphan chunks.
      - FIX 7: Overlap is applied via sentence carry-over, never exceeding chunk_size.
    """
    # --- Step 1: Split into structural blocks (headings / paragraphs) ---
    blocks = re.split(r"\n(?=#{1,6}\s)|\n\n+", markdown)
    blocks = [b.strip() for b in blocks if b.strip()]

    # --- Step 2: For each block, segment into sentences ---
    # Track headings so every chunk knows its section context
    all_sentences: list[str] = []
    sentence_headings: list[str] = []  # heading in effect when sentence was added
    current_heading = ""

    for block in blocks:
        h = _extract_heading(block)
        if h:
            current_heading = h

        try:
            sents = nltk.sent_tokenize(block)
        except Exception:
            sents = [block]

        for s in sents:
            s = s.strip()
            if s:
                all_sentences.append(s)
                sentence_headings.append(current_heading)

    if not all_sentences:
        return []

    # --- Step 3: Greedy pack sentences into chunks (FIX 7 — size-safe) ---
    chunks: list[tuple[str, str]] = []
    cur_sents: list[str] = []
    cur_heading = ""
    cur_len = 0

    def _flush(sents: list[str], hdg: str) -> None:
        if sents:
            chunks.append((" ".join(sents), hdg))

    for sent, hdg in zip(all_sentences, sentence_headings):
        addition = len(sent) + (1 if cur_sents else 0)  # +1 for space

        if cur_len + addition <= chunk_size:
            cur_sents.append(sent)
            cur_len += addition
            if not cur_heading:
                cur_heading = hdg
        else:
            _flush(cur_sents, cur_heading)

            # If a SINGLE sentence exceeds chunk_size, hard-split it by words
            if len(sent) > chunk_size:
                words = sent.split()
                part, part_len = [], 0
                for w in words:
                    if part_len + len(w) + 1 > chunk_size and part:
                        chunks.append((" ".join(part), hdg))
                        part, part_len = [w], len(w)
                    else:
                        part.append(w)
                        part_len += len(w) + 1
                if part:
                    chunks.append((" ".join(part), hdg))
                cur_sents, cur_heading, cur_len = [], hdg, 0
            else:
                cur_sents = [sent]
                cur_heading = hdg
                cur_len = len(sent)

    _flush(cur_sents, cur_heading)

    if not chunks:
        return []

    # --- Step 4: Sentence-boundary overlap (FIX 7 continued) ---
    if chunk_overlap <= 0 or len(chunks) <= 1:
        return chunks

    overlapped: list[tuple[str, str]] = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_text, _ = chunks[i - 1]
        curr_text, curr_hdg = chunks[i]

        # Take sentences from the END of the previous chunk until we hit overlap budget
        prev_sents = nltk.sent_tokenize(prev_text)
        tail_sents: list[str] = []
        tail_len = 0
        for s in reversed(prev_sents):
            if tail_len + len(s) + 1 > chunk_overlap:
                break
            tail_sents.insert(0, s)
            tail_len += len(s) + 1

        if tail_sents:
            new_text = " ".join(tail_sents) + " " + curr_text
        else:
            new_text = curr_text

        overlapped.append((new_text, curr_hdg))

    return overlapped


# ============================================================================
# LANGUAGE DETECTION
# ============================================================================


def detect_language(text: str) -> str:
    """
    Detect the primary language of a text using langdetect.
    Returns an ISO 639-1 code (e.g. 'en', 'ta', 'hi') or 'unknown'.
    Uses the first ~500 chars for speed — enough for reliable detection.
    """
    try:
        return _langdetect(text[:500])
    except (LangDetectException, Exception):
        return "unknown"


# ============================================================================
# CONVERSION: URL -> Markdown
# ============================================================================
