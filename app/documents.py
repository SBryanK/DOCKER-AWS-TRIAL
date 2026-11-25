from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}

_WS = re.compile(r"[ \t\u00a0]+")
_BLANK_LINES = re.compile(r"\n\s*\n+")


@dataclass(frozen=True, slots=True)
class Document:
    """One text unit with provenance: a PDF page or a whole text file."""

    source: str
    page: int
    text: str


def clean_text(raw: str) -> str:
    text = raw.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = _WS.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


def load_document(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type: {path.name} (use PDF, TXT or MD)")

    if suffix == ".pdf":
        return _load_pdf(path)
    text = clean_text(path.read_text(encoding="utf-8", errors="replace"))
    if len(text) < 20:
        raise ValueError(f"No extractable text in {path.name}")
    return [Document(source=path.name, page=1, text=text)]


def _load_pdf(path: Path) -> list[Document]:
    try:
        reader = PdfReader(str(path))
    except Exception as exc:  # pypdf raises several different exception types
        raise ValueError(f"Cannot read PDF {path.name}: {exc}") from exc

    if reader.is_encrypted:
        reader.decrypt("")

    documents: list[Document] = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = clean_text(page.extract_text() or "")
        except Exception:
            text = ""
        if len(text) >= 20:
            documents.append(Document(source=path.name, page=number, text=text))

    if not documents:
        raise ValueError(
            f"No extractable text in {path.name}. Scanned PDFs need OCR before ingestion."
        )
    return documents
