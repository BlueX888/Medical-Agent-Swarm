from __future__ import annotations

import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, List
from urllib.parse import urlsplit

from bs4 import BeautifulSoup


ALLOWED_EXTENSIONS = {".pdf", ".md", ".markdown", ".html", ".htm", ".txt"}
UNSAFE_SIGNATURES = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08", b"MZ", b"\x7fELF")


class DocumentValidationError(ValueError):
    pass


@dataclass
class TextSection:
    heading: str
    text: str


def safe_filename(filename: str) -> str:
    name = Path(filename or "").name
    name = re.sub(r"[^\w.\-\u4e00-\u9fff]", "_", name, flags=re.UNICODE)
    if not name or name in {".", ".."} or Path(name).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise DocumentValidationError("unsupported_file_type")
    return name[:180]


def safe_identifier(value: str, field: str) -> str:
    value = str(value or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", value):
        raise DocumentValidationError(f"invalid_{field}")
    return value


def validate_metadata(metadata: Any) -> dict:
    if not isinstance(metadata, dict):
        raise DocumentValidationError("metadata_must_be_object")
    result = {key: str(value or "").strip() for key, value in metadata.items()}
    title = result.get("title", "")
    if len(title) > 300:
        raise DocumentValidationError("title_too_long")
    for key in ("source_org", "published_at", "language"):
        if len(result.get(key, "")) > 200:
            raise DocumentValidationError(f"{key}_too_long")
    external_url = result.get("external_url", "")
    if external_url:
        parsed = urlsplit(external_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise DocumentValidationError("invalid_external_url")
    return result


def validate_file(filename: str, content: bytes, max_bytes: int) -> str:
    name = safe_filename(filename)
    if not content:
        raise DocumentValidationError("empty_document")
    if len(content) > max_bytes:
        raise DocumentValidationError("document_too_large")
    suffix = Path(name).suffix.lower()
    if content.startswith(UNSAFE_SIGNATURES):
        raise DocumentValidationError("unsafe_file_signature")
    actual_type = _sniff_document_type(content)
    expected_types = {
        ".pdf": {"pdf"},
        ".html": {"html"},
        ".htm": {"html"},
        ".md": {"text"},
        ".markdown": {"text"},
        ".txt": {"text"},
    }
    if actual_type not in expected_types[suffix]:
        raise DocumentValidationError("mime_extension_mismatch")
    if actual_type == "text":
        try:
            content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DocumentValidationError("text_must_be_utf8") from exc
    return name


def _sniff_document_type(content: bytes) -> str:
    if content.startswith(b"%PDF"):
        return "pdf"
    sample = content[:8192].lstrip().lower()
    if sample.startswith(b"<!doctype html") or sample.startswith(b"<html"):
        return "html"
    if re.search(br"<(?:head|body|h[1-6]|p|article|section)(?:\s|>)", sample):
        return "html"
    return "text"


def parse_document(filename: str, content: bytes) -> List[TextSection]:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise DocumentValidationError("pdf_support_not_installed") from exc
        reader = PdfReader(BytesIO(content))
        sections = [
            TextSection(heading=f"第 {index} 页", text=(page.extract_text() or "").strip())
            for index, page in enumerate(reader.pages, 1)
        ]
        sections = [section for section in sections if section.text]
        if not sections:
            raise DocumentValidationError("scanned_pdf_not_supported")
        return sections

    text = content.decode("utf-8").replace("\x00", "").strip()
    if suffix in {".html", ".htm"}:
        soup = BeautifulSoup(text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        for tag in soup.find_all(re.compile(r"^h[1-6]$")):
            level = int(tag.name[1])
            tag.replace_with(f"\n{'#' * level} {tag.get_text(' ', strip=True)}\n")
        text = soup.get_text("\n")
    return _heading_sections(text)


def _heading_sections(text: str) -> List[TextSection]:
    sections: List[TextSection] = []
    heading = "正文"
    lines: List[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if re.match(r"^#{1,6}\s+", line):
            if any(lines):
                sections.append(TextSection(heading, "\n".join(lines).strip()))
            heading = re.sub(r"^#{1,6}\s+", "", line).strip()
            lines = []
        elif line:
            lines.append(line)
    if any(lines):
        sections.append(TextSection(heading, "\n".join(lines).strip()))
    if not sections:
        raise DocumentValidationError("document_has_no_text")
    return sections


def chunk_sections(sections: List[TextSection], embedder: Any, size: int, overlap: int):
    output = []
    for section in sections:
        text = section.text
        if hasattr(embedder, "split_tokens"):
            pieces = embedder.split_tokens(text, size=size, overlap=overlap)
        else:
            step = max(1, size - overlap)
            pieces = [text[start : start + size] for start in range(0, len(text), step)]
        output.extend((section.heading, piece.strip()) for piece in pieces if piece.strip())
    return output
