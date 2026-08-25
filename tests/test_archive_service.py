"""Tests for the duplicates archive service."""

import hashlib
from pathlib import Path
from typing import Optional

import pytest

from photoheaven.application.archive_service import ArchiveService
from photoheaven.domain.models import MediaFile, MediaType


class FakeHasher:
    """Content-only hasher for tests."""

    def hash_file(self, path: Path) -> str:
        return hashlib.md5(path.read_bytes()).hexdigest()


class FakeRepository:
    """In-memory repository with just the methods ArchiveService needs."""

    def __init__(self) -> None:
        self.media: dict[str, MediaFile] = {}

    def save_media(self, media: MediaFile) -> None:
        self.media[media.id] = media

    def get_by_path(self, path: str) -> Optional[MediaFile]:
        for media in self.media.values():
            if media.path == path:
                return media
        return None

    def delete_media(self, media_id: str) -> None:
        self.media.pop(media_id, None)

    # Stub methods required by the abstract base class.
    def get_by_checksum(self, checksum: str) -> Optional[MediaFile]:
        return None

    def get_media_id_by_path(self, path: str) -> Optional[str]:
        media = self.get_by_path(path)
        return media.id if media else None

    def update_media_path(self, media_id: str, new_path: str) -> None:
        media = self.media.get(media_id)
        if media is not None:
            media.path = new_path

    def count_media(self) -> int:
        return len(self.media)

    def list_media(self, limit: int = 100, offset: int = 0) -> list[MediaFile]:
        return list(self.media.values())[offset : offset + limit]

    def get_all_media_paths(self) -> list[str]:
        return [m.path for m in self.media.values()]

    def update_media_perceptual_hash(
        self, media_id: str, perceptual_hash: str
    ) -> None:
        pass

    def update_media_video_frame_hashes(
        self, media_id: str, frame_hashes: list[str]
    ) -> None:
        pass

    def update_media_duration_seconds(
        self, media_id: str, duration_seconds: float
    ) -> None:
        pass

    def list_duplicate_groups(
        self,
        *,
        only_faces: bool = False,
        only_videos: bool = False,
    ) -> list[dict]:
        return []

    def save_duplicate_groups(
        self, groups: list[dict], *, progress_callback=None
    ) -> None:
        pass

    def get_media_paths_for_cluster(
        self,
        cluster_label: int,
        *,
        limit: int = 10,
        include_heic: bool = False,
    ) -> list[str]:
        return []

    def get_cluster_summary(
        self, limit: int = 100, offset: int = 0
    ) -> list[dict]:
        return []

    def get_identity_summary(
        self, limit: int = 100, offset: int = 0
    ) -> list[dict]:
        return []

    def get_identity_photo_counts(self) -> dict[str, int]:
        return {}

    def get_or_create_identity(self, name: str) -> tuple[str, bool]:
        return "", False

    def save_face(self, face) -> None:
        pass

    def get_unassigned_faces(self, embedding_version: str) -> list:
        return []

    def get_faces_for_identity(self, identity_id: str) -> list:
        return []

    def get_all_identities(self) -> list:
        return []

    def assign_face_to_identity(self, face_id: str, identity_id: str) -> None:
        pass

    def update_identity_name(self, identity_id: str, name: str) -> None:
        pass

    def merge_identities(self, target_id: str, source_id: str) -> None:
        pass


def _add_media(repo: FakeRepository, path: Path) -> None:
    repo.save_media(
        MediaFile(
            id=f"id-{path.name}",
            path=str(path),
            checksum=FakeHasher().hash_file(path),
            size_bytes=path.stat().st_size,
            mtime=path.stat().st_mtime,
            media_type=MediaType.IMAGE,
        )
    )


def _write_file(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_archives_files_and_removes_source():
    source = Path("/tmp/test_archive_source")
    archive = Path("/tmp/test_archive_dest")
    # Use unique paths to avoid cross-test collisions.
    source = Path(__file__).parent / "tmp_archive_source"
    archive = Path(__file__).parent / "tmp_archive_dest"
    try:
        for root in (source, archive):
            if root.exists():
                import shutil

                shutil.rmtree(root)

        files = {
            source / "2014" / "05" / "a.jpg": b"a",
            source / "2014" / "05" / "b.jpg": b"bb",
            source / "2024" / "08" / "c.mov": b"ccc",
        }
        repo = FakeRepository()
        for path, content in files.items():
            _write_file(path, content)
            _add_media(repo, path)

        service = ArchiveService(repository=repo, hasher=FakeHasher())
        result = service.archive_duplicates(source, archive)

        assert result.files_archived == 3
        assert result.files_skipped == 0
        assert result.files_failed == 0
        assert result.dirs_removed == 4  # 2014/05, 2014, 2024/08, 2024

        for path, content in files.items():
            archived = archive / path.relative_to(source)
            assert archived.read_bytes() == content
            assert not path.exists()

        assert repo.count_media() == 0
    finally:
        import shutil

        for root in (source, archive):
            if root.exists():
                shutil.rmtree(root)


def test_skips_already_archived_files():
    source = Path(__file__).parent / "tmp_archive_resume_source"
    archive = Path(__file__).parent / "tmp_archive_resume_dest"
    try:
        for root in (source, archive):
            if root.exists():
                import shutil

                shutil.rmtree(root)

        path = source / "file.txt"
        _write_file(path, b"hello")
        repo = FakeRepository()
        _add_media(repo, path)

        # Pre-populate archive with identical content.
        archived = archive / "file.txt"
        archived.parent.mkdir(parents=True, exist_ok=True)
        archived.write_bytes(b"hello")

        service = ArchiveService(repository=repo, hasher=FakeHasher())
        result = service.archive_duplicates(source, archive)

        assert result.files_archived == 0
        assert result.files_skipped == 1
        assert not path.exists()
        assert repo.count_media() == 0
    finally:
        import shutil

        for root in (source, archive):
            if root.exists():
                shutil.rmtree(root)


def test_archives_under_unique_name_when_destination_differs():
    source = Path(__file__).parent / "tmp_archive_unique_source"
    archive = Path(__file__).parent / "tmp_archive_unique_dest"
    try:
        for root in (source, archive):
            if root.exists():
                import shutil

                shutil.rmtree(root)

        path = source / "file.txt"
        _write_file(path, b"new")
        repo = FakeRepository()
        _add_media(repo, path)

        archived = archive / "file.txt"
        archived.parent.mkdir(parents=True, exist_ok=True)
        archived.write_bytes(b"old")

        service = ArchiveService(repository=repo, hasher=FakeHasher())
        result = service.archive_duplicates(source, archive)

        assert result.files_archived == 1
        assert (archive / "file_1.txt").read_bytes() == b"new"
        assert not path.exists()
    finally:
        import shutil

        for root in (source, archive):
            if root.exists():
                shutil.rmtree(root)


def test_dry_run_does_not_modify_filesystem_or_db():
    source = Path(__file__).parent / "tmp_archive_dryrun_source"
    archive = Path(__file__).parent / "tmp_archive_dryrun_dest"
    try:
        for root in (source, archive):
            if root.exists():
                import shutil

                shutil.rmtree(root)

        path = source / "file.txt"
        _write_file(path, b"data")
        repo = FakeRepository()
        _add_media(repo, path)

        service = ArchiveService(repository=repo, hasher=FakeHasher())
        result = service.archive_duplicates(source, archive, dry_run=True)

        assert result.files_archived == 1
        assert path.exists()
        assert not archive.exists()
        assert repo.count_media() == 1
    finally:
        import shutil

        for root in (source, archive):
            if root.exists():
                shutil.rmtree(root)


def test_retries_and_resumes_from_partial_temp_file():
    source = Path(__file__).parent / "tmp_archive_retry_source"
    archive = Path(__file__).parent / "tmp_archive_retry_dest"
    try:
        for root in (source, archive):
            if root.exists():
                import shutil

                shutil.rmtree(root)

        path = source / "file.txt"
        _write_file(path, b"content")
        repo = FakeRepository()
        _add_media(repo, path)

        service = ArchiveService(repository=repo, hasher=FakeHasher())
        original_copy = service._copy_file
        attempts = {"count": 0}

        def flaky_copy(src: Path, dst: Path) -> None:
            attempts["count"] += 1
            if attempts["count"] == 1:
                # Simulate a broken network copy: write partial data then fail.
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(b"par")
                raise OSError("network dropped")
            original_copy(src, dst)

        service._copy_file = flaky_copy  # type: ignore[method-assign]
        result = service.archive_duplicates(source, archive)

        assert result.files_archived == 1
        assert result.files_failed == 0
        assert attempts["count"] == 2
        assert (archive / "file.txt").read_bytes() == b"content"
        assert not path.exists()
    finally:
        import shutil

        for root in (source, archive):
            if root.exists():
                shutil.rmtree(root)
