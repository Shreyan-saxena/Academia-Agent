"""Document loading, metadata parsing, and privacy-preserving in-memory storage."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Iterable, List, Optional

from .schemas import DocumentSummary, UserRole


_FRONT_MATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_HEADING = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)


@dataclass(frozen=True)
class AcademicDocument:
    id: str
    title: str
    source: str
    text: str
    allowed_roles: tuple[UserRole, ...]


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    title: str
    section: str
    text: str
    parent_text: str
    allowed_roles: tuple[UserRole, ...]


def _document_id(source: str, text: str) -> str:
    digest = hashlib.sha256(f"{source}:{text}".encode("utf-8")).hexdigest()[:12]
    return f"doc_{digest}"


def _parse_front_matter(raw: str, fallback_title: str, source: str) -> AcademicDocument:
    metadata: dict[str, str] = {}
    body = raw.strip()
    match = _FRONT_MATTER.match(raw)
    if match:
        for line in match.group(1).splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                metadata[key.strip().lower()] = value.strip()
        body = raw[match.end():].strip()
    title = metadata.get("title", fallback_title)
    raw_roles = metadata.get("roles", "student,faculty,staff,administrator")
    roles: list[UserRole] = []
    for candidate in raw_roles.split(","):
        try:
            roles.append(UserRole(candidate.strip()))
        except ValueError:
            continue
    if not roles:
        roles = list(UserRole)
    return AcademicDocument(
        id=_document_id(source, body),
        title=title,
        source=source,
        text=body,
        allowed_roles=tuple(roles),
    )


def load_markdown(path: Path, source: Optional[str] = None) -> AcademicDocument:
    return _parse_front_matter(path.read_text(encoding="utf-8"), path.stem.replace("_", " ").title(), source or str(path))


def chunk_document(document: AcademicDocument, words_per_chunk: int = 180, overlap_words: int = 35) -> list[Chunk]:
    """Create child retrieval chunks while retaining a larger section parent payload."""
    sections = re.split(r"(?=^#{1,6}\s+)", document.text, flags=re.MULTILINE)
    chunks: list[Chunk] = []
    for section_index, section_text in enumerate(s for s in sections if s.strip()):
        heading = _HEADING.search(section_text)
        section_name = heading.group(1).strip() if heading else "Overview"
        clean = _HEADING.sub("", section_text, count=1).strip()
        words = clean.split()
        if not words:
            continue
        start = 0
        part = 0
        while start < len(words):
            end = min(len(words), start + words_per_chunk)
            child_text = " ".join(words[start:end])
            parent_start = max(0, start - overlap_words)
            parent_end = min(len(words), end + overlap_words * 2)
            parent_text = " ".join(words[parent_start:parent_end])
            chunks.append(Chunk(
                id=f"{document.id}_{section_index}_{part}",
                document_id=document.id,
                title=document.title,
                section=section_name,
                text=child_text,
                parent_text=parent_text,
                allowed_roles=document.allowed_roles,
            ))
            if end == len(words):
                break
            start = end - overlap_words
            part += 1
    return chunks


class AcademicCorpus:
    """Thread-safe local corpus. Replace this boundary with Qdrant in production."""

    def __init__(self) -> None:
        self._documents: dict[str, AcademicDocument] = {}
        self._chunks: list[Chunk] = []
        self._lock = RLock()

    def add(self, document: AcademicDocument) -> AcademicDocument:
        new_chunks = chunk_document(document)
        with self._lock:
            self._documents[document.id] = document
            self._chunks = [chunk for chunk in self._chunks if chunk.document_id != document.id]
            self._chunks.extend(new_chunks)
        return document

    def load_seed_directory(self, directory: Path) -> None:
        for path in sorted(directory.glob("*.md")):
            self.add(load_markdown(path, source=f"seed:{path.name}"))

    def chunks_for(self, role: UserRole) -> list[Chunk]:
        with self._lock:
            return [chunk for chunk in self._chunks if role in chunk.allowed_roles]

    def documents(self) -> list[DocumentSummary]:
        with self._lock:
            counts: dict[str, int] = {}
            for chunk in self._chunks:
                counts[chunk.document_id] = counts.get(chunk.document_id, 0) + 1
            return [DocumentSummary(
                id=document.id,
                title=document.title,
                source=document.source,
                allowed_roles=list(document.allowed_roles),
                chunks=counts.get(document.id, 0),
            ) for document in self._documents.values()]

    @property
    def chunk_count(self) -> int:
        with self._lock:
            return len(self._chunks)

    @property
    def document_count(self) -> int:
        with self._lock:
            return len(self._documents)
