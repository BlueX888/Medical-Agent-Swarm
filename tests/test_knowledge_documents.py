from __future__ import annotations

import pytest

from knowledge.documents import DocumentValidationError, parse_document, validate_file


def test_html_parser_preserves_heading_sections():
    sections = parse_document(
        "guide.html",
        b"<html><body><h1>Diagnosis</h1><p>Use clinical criteria.</p>"
        b"<h2>Treatment</h2><p>Individualize care.</p></body></html>",
    )

    assert [(section.heading, section.text) for section in sections] == [
        ("Diagnosis", "Use clinical criteria."),
        ("Treatment", "Individualize care."),
    ]


@pytest.mark.parametrize(
    ("filename", "content", "error"),
    [
        ("archive.txt", b"PK\x03\x04payload", "unsafe_file_signature"),
        ("program.md", b"MZpayload", "unsafe_file_signature"),
        ("fake.txt", b"%PDF-1.7 payload", "mime_extension_mismatch"),
        ("fake.html", b"plain text only", "mime_extension_mismatch"),
    ],
)
def test_file_validation_rejects_unsafe_or_mismatched_actual_types(filename, content, error):
    with pytest.raises(DocumentValidationError, match=error):
        validate_file(filename, content, 1024)
