"""Unit tests for the ingestion application service."""

from __future__ import annotations

from pathlib import Path

from photoheaven.application.ingestion_service import IngestionService
from photoheaven.application.ports import (
    Hasher,
    MediaMetadata,
    MediaRepository,
    MetadataExtractor,
)
from photoheaven.domain.models import MediaFile, MediaType


class FakeHasher(Hasher):
    def hash_file(self, path: Path) -> str:
        return "fake_checksum"


class FakeRepository(MediaRepository):
    def __init__(self) -> None:
        self.media: dict[str, MediaFile] = {}
        self.faces = {}

    def get_by_checksum(self, checksum: str) -> MediaFile | None:
        for media in self.media.values():
            if media.checksum == checksum:
                return media
        return None

    def save_media(self, media: MediaFile) -> None:
        self.media[media.id] = media

    def count_media(self) -> int:
        return len(self.media)

    def list_media(self, limit: int = 100, offset: int = 0) -> list[MediaFile]:
        return list(self.media.values())[offset : offset + limit]

    def get_all_media_paths(self) -> list[str]:
        return [media.path for media in self.media.values()]

    def save_face(self, face) -> None:
        pass

    def count_faces(self) -> int:
        return 0

    def media_has_faces(self, media_id: str) -> bool:
        return False

    def get_media_ids_with_faces(self) -> set[str]:
        return set()

    def get_unprocessed_faces_media(
        self, limit: int = 100, offset: int = 0
    ) -> list[MediaFile]:
        return []

    def update_media_face_analysis(
        self, media_id: str, analyzed_at, version: str
    ) -> None:
        pass

    def list_faces_for_media(self, media_id: str) -> list:
        return []

    def get_face_by_id(self, face_id: str):
        return None

    def list_faces(self, limit: int = 100, offset: int = 0) -> list:
        return []

    def get_all_faces(self) -> list:
        return []

    def get_embedding_versions(self) -> set[str]:
        return set()

    def update_face_cluster_label(
        self, face_id: str, cluster_label: int | None
    ) -> None:
        pass

    def update_face_identity_name_for_cluster(
        self, cluster_label: int, identity_name: str | None
    ) -> None:
        pass

    def save_identity(self, identity) -> None:
        pass

    def get_identity_by_name(self, name: str):
        return None

    def get_identity_by_id(self, identity_id: str):
        return None

    def list_identities(self, limit: int = 100, offset: int = 0):
        return []

    def get_faces_for_cluster(self, cluster_label: int):
        return []

    def get_faces_for_identity(self, identity_id: str):
        return []

    def get_faces_without_identity(self, limit: int = 100, offset: int = 0):
        return []

    def update_face_identity(
        self,
        face_id: str,
        *,
        identity_id: str | None,
        identity_name: str | None,
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

    def delete_media(self, media_id: str) -> None:
        self.media.pop(media_id, None)

    def get_identity_photo_counts(self) -> dict[str, int]:
        return {}

    def update_media_perceptual_hash(
        self, media_id: str, perceptual_hash: str
    ) -> None:
        media = self.media.get(media_id)
        if media is not None:
            media.perceptual_hash = perceptual_hash

    def update_media_video_frame_hashes(
        self, media_id: str, frame_hashes: list[str]
    ) -> None:
        media = self.media.get(media_id)
        if media is not None:
            media.video_frame_hashes = frame_hashes


    def update_media_duration_seconds(
        self, media_id: str, duration_seconds: float
    ) -> None:
        pass
    def update_media_path(self, media_id: str, new_path: str) -> None:
        media = self.media.get(media_id)
        if media is not None:
            media.path = new_path

    def get_media_id_by_path(self, path: str) -> str | None:
        for media in self.media.values():
            if media.path == path:
                return media.id
        return None


    def get_by_path(self, path: str) -> Optional[MediaFile]:
        for media in self.media.values():
            if media.path == path:
                return media
        return None

    def search_media(self, query):
        return []
    def clear_duplicate_groups(self) -> None:
        pass

    def save_duplicate_groups(
        self, groups: list[tuple[str, list[dict]]]
    ) -> None:
        pass

    def list_duplicate_groups(self) -> list[dict]:
        return []


class FailingMetadataExtractor(MetadataExtractor):
    """Always raises, simulating a corrupt/unreadable image."""

    def extract(self, path: Path, media_type: MediaType) -> MediaMetadata:
        raise RuntimeError("cannot identify image file")


class SuccessMetadataExtractor(MetadataExtractor):
    """Returns basic metadata successfully."""

    def extract(self, path: Path, media_type: MediaType) -> MediaMetadata:
        return MediaMetadata(media_type=media_type)


class SwallowingFailureExtractor(MetadataExtractor):
    """Simulates FallbackMetadataExtractor: swallows exception, returns flag."""

    def extract(self, path: Path, media_type: MediaType) -> MediaMetadata:
        return MediaMetadata(media_type=media_type, extracted=False)


def test_metadata_extraction_failure_sets_flag(tmp_path: Path) -> None:
    repo = FakeRepository()
    service = IngestionService(
        hasher=FakeHasher(),
        metadata_extractor=FailingMetadataExtractor(),
        repository=repo,
    )

    file_path = tmp_path / "corrupt.jpg"
    file_path.write_bytes(b"not an image")

    result = service.ingest_file(file_path)

    assert result.status in {"added", "updated"}
    assert result.metadata_extracted is False
    assert result.media is not None
    assert result.media.media_type is MediaType.IMAGE


def test_successful_metadata_extraction_sets_flag(tmp_path: Path) -> None:
    repo = FakeRepository()
    service = IngestionService(
        hasher=FakeHasher(),
        metadata_extractor=SuccessMetadataExtractor(),
        repository=repo,
    )

    file_path = tmp_path / "valid.jpg"
    file_path.write_bytes(b"not an image")

    result = service.ingest_file(file_path)

    assert result.status in {"added", "updated"}
    assert result.metadata_extracted is True
    assert result.media is not None
    assert result.media.metadata_extracted is True


def test_metadata_extracted_is_persisted(tmp_path: Path) -> None:
    repo = FakeRepository()
    service = IngestionService(
        hasher=FakeHasher(),
        metadata_extractor=FailingMetadataExtractor(),
        repository=repo,
    )

    file_path = tmp_path / "corrupt.jpg"
    file_path.write_bytes(b"not an image")

    service.ingest_file(file_path)
    saved = next(iter(repo.media.values()))

    assert saved.metadata_extracted is False


def test_check_metadata_returns_false_on_failure(tmp_path: Path) -> None:
    service = IngestionService(
        hasher=FakeHasher(),
        metadata_extractor=FailingMetadataExtractor(),
        repository=FakeRepository(),
    )

    file_path = tmp_path / "corrupt.jpg"
    file_path.write_bytes(b"not an image")

    assert service.check_metadata(file_path) is False


def test_check_metadata_returns_true_on_success(tmp_path: Path) -> None:
    service = IngestionService(
        hasher=FakeHasher(),
        metadata_extractor=SuccessMetadataExtractor(),
        repository=FakeRepository(),
    )

    file_path = tmp_path / "valid.jpg"
    file_path.write_bytes(b"not an image")

    assert service.check_metadata(file_path) is True


def test_swallowed_extraction_failure_sets_flag(tmp_path: Path) -> None:
    """Fallback-style extractors that swallow errors must still report failure."""
    repo = FakeRepository()
    service = IngestionService(
        hasher=FakeHasher(),
        metadata_extractor=SwallowingFailureExtractor(),
        repository=repo,
    )

    file_path = tmp_path / "corrupt.jpg"
    file_path.write_bytes(b"not an image")

    result = service.ingest_file(file_path)

    assert result.status in {"added", "updated"}
    assert result.metadata_extracted is False
    assert result.media is not None
    assert result.media.metadata_extracted is False
