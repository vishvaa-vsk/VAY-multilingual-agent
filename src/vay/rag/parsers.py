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
import importlib.util as _ilu
import sys as _sys
from pathlib import Path as _Path

_chroma_setup_path = _Path(__file__).parent / "chroma_setup (1).py"
if not _chroma_setup_path.exists():
    _chroma_setup_path = _Path(__file__).parent / "chroma_setup.py"
_spec = _ilu.spec_from_file_location("chroma_setup", str(_chroma_setup_path))
_mod = _ilu.module_from_spec(_spec)
_sys.modules.setdefault("chroma_setup", _mod)
_spec.loader.exec_module(_mod)

import re

# ---------------------------------------------------------------------------
# Force UTF-8 on Windows consoles (cp1252 chokes on Unicode in rich content)
# ---------------------------------------------------------------------------
import sys as _sys_utf8
from pathlib import Path

import html2text
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

if hasattr(_sys_utf8.stdout, "reconfigure"):
    try:
        _sys_utf8.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def fetch_html(url: str, timeout: int = 15) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def url_to_markdown(url: str) -> tuple[str, str]:
    """Fetch a URL and convert its readable content to Markdown."""
    html = fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(
        ["script", "style", "noscript", "nav", "footer", "header", "aside", "form", "iframe"]
    ):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else url

    converter = html2text.HTML2Text()
    converter.ignore_images = False
    converter.ignore_links = False
    converter.body_width = 0
    converter.baseurl = url

    markdown = converter.handle(str(soup))
    markdown = re.sub(r"\n{3,}", "\n\n", markdown).strip()
    if title:
        markdown = f"# {title}\n\n{markdown}"
    return markdown, title


# ============================================================================
# CONVERSION: PDF -> Markdown
# ============================================================================


def pdf_to_markdown(pdf_path: str) -> tuple[str, str]:
    """Extract text page-by-page from a PDF and format it as Markdown."""
    reader = PdfReader(pdf_path)
    title = Path(pdf_path).stem
    meta_title = reader.metadata.title if reader.metadata else None
    if meta_title and meta_title.strip() and meta_title.strip().lower() not in ("untitled", ""):
        title = meta_title.strip()

    parts = [f"# {title}\n"]
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        parts.append(f"## Page {i}\n\n{text}")

    markdown = re.sub(r"\n{3,}", "\n\n", "\n\n".join(parts)).strip()
    return markdown, title


# ============================================================================
# SHARED: save markdown, slugify
# ============================================================================
