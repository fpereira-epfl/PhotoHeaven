"""Tests for lightweight SQLite schema migration."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine

from photoheaven.adapters.persistence.sqlite import SqliteMediaRepository


def _create_old_schema(db_path: Path) -> None:
    """Create a database file matching the schema before face-analysis columns."""
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        conn.exec_driver_sql(
            """
            CREATE TABLE media_files (
                id TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                checksum TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                mtime REAL NOT NULL,
                media_type TEXT NOT NULL,
                capture_datetime DATETIME,
                make TEXT,
                model TEXT,
                latitude REAL,
                longitude REAL,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL
            )
            """
        )
        conn.exec_driver_sql(
            """
            CREATE TABLE faces (
                id TEXT PRIMARY KEY,
                media_id TEXT NOT NULL,
                bbox_x1 INTEGER NOT NULL,
                bbox_y1 INTEGER NOT NULL,
                bbox_x2 INTEGER NOT NULL,
                bbox_y2 INTEGER NOT NULL,
                embedding_blob BLOB NOT NULL,
                embedding_version TEXT NOT NULL,
                detection_confidence REAL NOT NULL,
                cluster_label INTEGER,
                identity_name TEXT,
                created_at DATETIME NOT NULL
            )
            """
        )
    engine.dispose()


def test_repository_migrates_missing_face_analysis_columns(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "old.db"
    _create_old_schema(db_path)

    repository = SqliteMediaRepository(str(db_path))

    with repository.engine.begin() as conn:
        columns = {
            row[1] for row in conn.exec_driver_sql("PRAGMA table_info(media_files)")
        }

    assert "face_analysis_at" in columns
    assert "face_analysis_version" in columns
    assert "metadata_extracted" in columns
    assert "perceptual_hash" in columns
    assert "duration_seconds" in columns
    assert "video_frame_hashes" in columns
    assert repository.count_media() == 0
