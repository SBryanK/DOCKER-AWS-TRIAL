"""CLI: ingest policy PDFs into the vector store (no HTTP server needed).

Examples:
    python -m scripts.ingest                  # ingest data/policies
    python -m scripts.ingest path/to/a.pdf    # ingest specific files
    python -m scripts.ingest --reset          # replace the existing index
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from app.config import Settings
from app.documents import SUPPORTED_SUFFIXES
from app.rag import RAGService


def _gather(raw_paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in raw_paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(
                sorted(
                    child
                    for child in path.rglob("*")
                    if child.is_file() and child.suffix.lower() in SUPPORTED_SUFFIXES
                )
            )
        elif path.is_file():
            files.append(path)
        else:
            raise SystemExit(f"Path not found: {path}")
    if not files:
        raise SystemExit("No PDF/TXT/MD files found to ingest.")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        default=None,
        help="Files or directories (default: data/policies from settings)",
    )
    parser.add_argument("--reset", action="store_true", help="Replace the existing index")
    args = parser.parse_args()

    settings = Settings()
    logging.basicConfig(
        level=settings.log_level.upper(), format="%(levelname)s %(name)s: %(message)s"
    )

    raw_paths = args.paths or [str(settings.policy_dir)]
    files = _gather(raw_paths)
    print(f"Ingesting {len(files)} file(s) with {settings.embeddings_provider} embeddings")

    service = RAGService(settings)
    summary = service.ingest_files(files, reset=args.reset)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
