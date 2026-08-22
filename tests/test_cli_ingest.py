"""Tests for CLI ingestion helpers."""

from __future__ import annotations

from pathlib import Path

from photoheaven.cli.main import _target_path_for_corrupted


def test_target_path_preserves_relative_structure(tmp_path: Path) -> None:
    ingest_root = tmp_path / "source"
    target_root = tmp_path / "corrupted"
    file_path = ingest_root / "2020" / "04" / "image.jpg"
    file_path.parent.mkdir(parents=True)
    file_path.touch()

    target = _target_path_for_corrupted(file_path, ingest_root, target_root)

    assert target == target_root / "2020" / "04" / "image.jpg"


def test_target_path_handles_name_collision(tmp_path: Path) -> None:
    ingest_root = tmp_path / "source"
    target_root = tmp_path / "corrupted"
    file_path = ingest_root / "image.jpg"
    file_path.parent.mkdir(parents=True)
    file_path.touch()

    # Pre-create the first target to force a collision.
    (target_root / "image.jpg").parent.mkdir(parents=True, exist_ok=True)
    (target_root / "image.jpg").touch()

    target = _target_path_for_corrupted(file_path, ingest_root, target_root)

    assert target == target_root / "image_1.jpg"
