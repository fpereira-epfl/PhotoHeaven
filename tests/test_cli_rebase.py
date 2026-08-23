"""Tests for the ``ph rebase`` command."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

from typer.testing import CliRunner

from photoheaven.adapters.persistence.sqlite import SqliteMediaRepository
from photoheaven.cli import config as cli_config
from photoheaven.cli.main import app
from photoheaven.domain.models import MediaFile, MediaType

runner = CliRunner()


def _media(path: str, checksum: str = "abc") -> MediaFile:
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
    library = tmp_path / "Library.photoslibrary"
    db_path = library / "db" / "photoheaven.db"
    old_root = tmp_path / "Queue"
    _write_db_with_paths(
        db_path,
        [
            str(old_root / "2008" / "12" / "img.jpg"),
            str(old_root / "2024" / "05" / "other.jpg"),
        ],
    )

    cli_config.state["library"] = str(library)
    result = runner.invoke(app, ["rebase"])
    cli_config.state["library"] = None

    assert result.exit_code == 0
    repo = SqliteMediaRepository(str(db_path))
    media = repo.list_media(limit=10)
    paths = {m.path for m in media}
    assert paths == {
        str(library / "files" / "2008" / "12" / "img.jpg"),
        str(library / "files" / "2024" / "05" / "other.jpg"),
    }


def test_rebase_dry_run_does_not_modify(tmp_path: Path) -> None:
    library = tmp_path / "Library.photoslibrary"
    db_path = library / "db" / "photoheaven.db"
    old_root = tmp_path / "Queue"
    original_paths = [
        str(old_root / "2008" / "12" / "img.jpg"),
    ]
    _write_db_with_paths(db_path, original_paths)

    cli_config.state["library"] = str(library)
    result = runner.invoke(app, ["rebase", "--dry-run"])
    cli_config.state["library"] = None

    assert result.exit_code == 0
    repo = SqliteMediaRepository(str(db_path))
    media = repo.list_media(limit=10)
    assert [m.path for m in media] == original_paths
    assert "Dry run" in result.output


def test_rebase_leaves_paths_without_date_pattern_unchanged(
    tmp_path: Path,
) -> None:
    library = tmp_path / "Library.photoslibrary"
    db_path = library / "db" / "photoheaven.db"
    old_root = tmp_path / "Queue"
    no_date_path = str(old_root / "screenshots" / "no_date.jpg")
    _write_db_with_paths(
        db_path,
        [
            str(old_root / "2008" / "12" / "img.jpg"),
            no_date_path,
        ],
    )

    cli_config.state["library"] = str(library)
    result = runner.invoke(app, ["rebase"])
    cli_config.state["library"] = None

    assert result.exit_code == 0
    repo = SqliteMediaRepository(str(db_path))
    media = repo.list_media(limit=10)
    paths = {m.path for m in media}
    assert str(library / "files" / "2008" / "12" / "img.jpg") in paths
    assert no_date_path in paths


def test_rebase_reports_no_media(tmp_path: Path) -> None:
    library = tmp_path / "Library.photoslibrary"
    db_path = library / "db" / "photoheaven.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    SqliteMediaRepository(str(db_path))

    cli_config.state["library"] = str(library)
    result = runner.invoke(app, ["rebase"])
    cli_config.state["library"] = None

    assert result.exit_code == 0
    assert "No media paths" in result.output


def test_rebase_debug_shows_unchanged_reasons(tmp_path: Path) -> None:
    library = tmp_path / "Library.photoslibrary"
    db_path = library / "db" / "photoheaven.db"
    old_root = tmp_path / "Queue"
    no_date_path = str(old_root / "screenshots" / "no_date.jpg")
    _write_db_with_paths(
        db_path,
        [
            str(old_root / "2008" / "12" / "img.jpg"),
            no_date_path,
        ],
    )

    cli_config.state["library"] = str(library)
    result = runner.invoke(app, ["rebase", "--dry-run", "--debug"])
    cli_config.state["library"] = None

    assert result.exit_code == 0
    assert "no YYYY/MM folder segment found" in result.output
    assert "Sample unchanged paths" in result.output
