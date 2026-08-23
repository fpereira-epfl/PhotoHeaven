"""Tests for library initialisation."""

from __future__ import annotations

from pathlib import Path

from photoheaven.adapters.integrity.hasher import Blake3Hasher
from photoheaven.adapters.persistence.sqlite import SqliteMediaRepository
from photoheaven.application.library_service import LibraryService


def test_init_library_creates_database(tmp_path: Path) -> None:
    service = LibraryService(Blake3Hasher())
    library = tmp_path / "MyLibrary.photoslibrary"

    db_path = service.init_library(library)

    assert db_path == library / "db" / "photoheaven.db"
    assert db_path.exists()
    assert (library / "files").is_dir()
    # Opening the repository should succeed and see no tables issues.
    repo = SqliteMediaRepository(str(db_path))
    assert repo.count_media() == 0
