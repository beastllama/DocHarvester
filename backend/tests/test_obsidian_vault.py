"""An Obsidian vault read through the local folder connector.

Builds a small vault inside INGEST_ROOT and reads it the way ingest does.
"""
import asyncio
import os
import shutil
from pathlib import Path

import pytest

from backend.connectors.local_folder import LocalFolderConnector, parse_markdown_note

NOTE = """---
tags: [policy, finance]
aliases: Money back
---
# Refunds

See the [[Refund policy|refund page]] and [[Payments#Cards]].
![[receipt.png]]
"""


@pytest.fixture
def vault():
    root = Path(os.environ["INGEST_ROOT"]) / "obsidian_vault_test"
    shutil.rmtree(root, ignore_errors=True)
    (root / "Finance").mkdir(parents=True)
    (root / ".obsidian").mkdir()
    (root / ".trash").mkdir()
    (root / "Finance" / "Refunds.md").write_text(NOTE, encoding="utf-8")
    (root / ".obsidian" / "app.json").write_text('{"theme": "dark"}', encoding="utf-8")
    (root / ".trash" / "Old.md").write_text("deleted note", encoding="utf-8")
    (root / "Broken.md").write_bytes(b"\xff\xfe not utf-8 \xff")
    yield root
    shutil.rmtree(root, ignore_errors=True)


def _read_all(root):
    return asyncio.run(LocalFolderConnector({"folder_path": str(root)}).search("", limit=5000))


def test_vault_settings_trash_and_broken_files_are_not_ingested(vault):
    titles = sorted(r.title for r in _read_all(vault))
    assert titles == ["Refunds"]


def test_note_text_metadata_and_links(vault):
    [note] = _read_all(vault)
    assert note.title == "Refunds"
    assert "tags:" not in note.raw_text and "---" not in note.raw_text
    assert "See the refund page and Payments." in note.raw_text
    assert "[[" not in note.raw_text and "receipt.png" not in note.raw_text
    assert note.source_meta["tags"] == ["policy", "finance"]
    assert note.source_meta["aliases"] == ["Money back"]
    assert note.source_meta["links"] == ["Refund policy", "Payments"]
    assert note.source_meta["relative_path"] == "Finance/Refunds.md"


def test_plain_markdown_without_frontmatter_is_unchanged():
    text, meta = parse_markdown_note("# Title\n\nJust text with --- dashes.")
    assert text.strip() == "Title\nJust text with --- dashes."
    assert meta == {}


def test_bad_frontmatter_is_dropped_not_fatal():
    text, meta = parse_markdown_note("---\ntags: [unclosed\n---\nBody")
    assert text.strip() == "Body"
    assert meta == {}
