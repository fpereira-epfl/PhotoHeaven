"""Tests for the ``ph rebase`` command."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

from typer.testing import CliRunner

from photoheaven.adapters.persistence.sqlite import SqliteMediaRepository
from photoheaven.cli.main import app
from photoheaven.domain.models import MediaFile, MediaType

runner = CliRunner()


def _media(path: str, checksum: str) -> MediaFile:
    return MediaFile(
        id=str(uuid4()),
        path=path,
        checksum=checksum,
        size_bytes=1,
        mtime=1.0,
        media_type=MediaType.IMAGE,
        capture_datetime=datetime.utcnow(),
    )


def _write_db_with_paths(db_path: Path, paths: list[str]) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    repo = SqliteMediaRepository(str(db_path))
    for index, path in enumerate(paths):
        repo.save_media(_media(path, checksum=f"checksum-{index}"))


def test_rebase_updates_media_paths(tmp_path: Path) -> None:
    db_path = tmp_path / "db" / "photoheaven.db"
    old_root = tmp_path / "Queue"
    new_root = tmp_path / "Library.photoslibrary"
    _write_db_with_paths(
        db_path,
        [
            str(old_root / "2008" / "12" / "img.jpg"),
            str(old_root / "2024" / "05" / "other.jpg"),
        ],
    )

    result = runner.invoke(
        app,
        [
            "rebase",
            str(new_root),
            "--db",
            str(db_path),
        ],
    )

    assert result.exit_code == 0
    repo = SqliteMediaRepository(str(db_path))
    media = repo.list_media(limit=10)
    paths = {m.path for m in media}
    assert paths == {
        str(new_root / "files" / "2008" / "12" / "img.jpg"),
        str(new_root / "files" / "2024" / "05" / "other.jpg"),
    }


def test_rebase_dry_run_does_not_modify(tmp_path: Path) -> None:
    db_path = tmp_path / "db" / "photoheaven.db"
    old_root = tmp_path / "Queue"
    new_root = tmp_path / "Library.photoslibrary"
    original_paths = [
        str(old_root / "2008" / "12" / "img.jpg"),
    ]
    _write_db_with_paths(db_path, original_paths)

    result = runner.invoke(
        app,
        [
            "rebase",
            str(new_root),
            "--db",
            str(db_path),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    repo = SqliteMediaRepository(str(db_path))
    media = repo.list_media(limit=10)
    assert [m.path for m in media] == original_paths
    assert "Dry run" in result.output


def test_rebase_leaves_paths_without_date_pattern_unchanged(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "db" / "photoheaven.db"
    old_root = tmp_path / "Queue"
    new_root = tmp_path / "Library.photoslibrary"
    no_date_path = str(old_root / "no_date_folder.jpg")
    _write_db_with_paths(
        db_path,
        [
            str(old_root / "2008" / "12" / "img.jpg"),
            no_date_path,
        ],
    )

    result = runner.invoke(
        app,
        [
            "rebase",
            str(new_root),
            "--db",
            str(db_path),
        ],
    )

    assert result.exit_code == 0
    repo = SqliteMediaRepository(str(db_path))
    media = repo.list_media(limit=10)
    paths = {m.path for m in media}
    assert str(new_root / "files" / "2008" / "12" / "img.jpg") in paths
    assert no_date_path in paths


def test_rebase_reports_no_media(tmp_path: Path) -> None:
    db_path = tmp_path / "db" / "photoheaven.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    SqliteMediaRepository(str(db_path))

    result = runner.invoke(
        app,
        [
            "rebase",
            str(tmp_path / "Library.photoslibrary"),
            "--db",
            str(db_path),
        ],
    )

    assert result.exit_code == 0
    assert "No media paths" in result.output


def test_rebase_debug_shows_unchanged_reasons(tmp_path: Path) -> None:
    db_path = tmp_path / "db" / "photoheaven.db"
    old_root = tmp_path / "Queue"
    new_root = tmp_path / "Library.photoslibrary"
    no_date_path = str(old_root / "screenshots" / "no_date.jpg")
    _write_db_with_paths(
        db_path,
        [
            str(old_root / "2008" / "12" / "img.jpg"),
            no_date_path,
        ],
    )

    result = runner.invoke(
        app,
        [
            "rebase",
            str(new_root),
            "--db",
            str(db_path),
            "--dry-run",
            "--debug",
        ],
    )

    assert result.exit_code == 0
    assert "no YYYY/MM folder segment found" in result.output
    assert "Sample unchanged paths" in result.output
