"""Unit tests for the face detection application service."""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from photoheaven.application.face_detection_service import FaceDetectionService
from photoheaven.application.ports import FaceAnalyzer, MediaRepository
from photoheaven.domain.models import Face, MediaFile, MediaType


class FakeRepository(MediaRepository):
    """In-memory repository stub for testing."""

    def __init__(self) -> None:
        self.media: dict[str, MediaFile] = {}
        self.faces: dict[str, Face] = {}

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

    def save_face(self, face: Face) -> None:
        self.faces[face.id] = face

    def count_faces(self) -> int:
        return len(self.faces)

    def media_has_faces(self, media_id: str) -> bool:
        return any(face.media_id == media_id for face in self.faces.values())

    def get_unprocessed_faces_media(
        self, limit: int = 100, offset: int = 0
    ) -> list[MediaFile]:
        unprocessed = [
            media
            for media in self.media.values()
            if media.face_analysis_at is None
        ]
        return unprocessed[offset : offset + limit]

    def update_media_face_analysis(
        self, media_id: str, analyzed_at: datetime, version: str
    ) -> None:
        media = self.media.get(media_id)
        if media is not None:
            media.face_analysis_at = analyzed_at
            media.face_analysis_version = version

    def list_faces_for_media(self, media_id: str) -> list[Face]:
        return [
            face for face in self.faces.values() if face.media_id == media_id
        ]

    def get_face_by_id(self, face_id: str) -> Face | None:
        return self.faces.get(face_id)

    def list_faces(self, limit: int = 100, offset: int = 0) -> list[Face]:
        return list(self.faces.values())[offset : offset + limit]

    def get_all_faces(self) -> list[Face]:
        return list(self.faces.values())

    def get_embedding_versions(self) -> set[str]:
        versions = {
            face.embedding_version
            for face in self.faces.values()
            if face.embedding_version
        }
        return versions if versions else set()

    def update_face_cluster_label(
        self, face_id: str, cluster_label: int | None
    ) -> None:
        face = self.faces.get(face_id)
        if face is not None:
            face.cluster_label = cluster_label

    def update_face_identity_name_for_cluster(
        self, cluster_label: int, identity_name: str | None
    ) -> None:
        for face in self.faces.values():
            if face.cluster_label == cluster_label:
                face.identity_name = identity_name

    def save_identity(self, identity) -> None:
        pass

    def get_identity_by_name(self, name: str):
        return None

    def get_identity_by_id(self, identity_id: str):
        return None

    def list_identities(self, limit: int = 100, offset: int = 0):
        return []

    def get_faces_for_cluster(self, cluster_label: int):
        return [
            face
            for face in self.faces.values()
            if face.cluster_label == cluster_label
        ]

    def get_faces_for_identity(self, identity_id: str):
        return [
            face
            for face in self.faces.values()
            if face.identity_id == identity_id
        ]

    def get_faces_without_identity(self, limit: int = 100, offset: int = 0):
        return [
            face
            for face in self.faces.values()
            if face.identity_id is None
        ][offset : offset + limit]

    def update_face_identity(
        self,
        face_id: str,
        *,
        identity_id: str | None,
        identity_name: str | None,
    ) -> None:
        face = self.faces.get(face_id)
        if face is not None:
            face.identity_id = identity_id
            face.identity_name = identity_name

    def get_media_paths_for_cluster(
        self,
        cluster_label: int,
        *,
        limit: int = 10,
        include_heic: bool = False,
    ) -> list[str]:
        paths = {
            self.media[face.media_id].path
            for face in self.faces.values()
            if face.cluster_label == cluster_label and face.media_id in self.media
        }
        return list(paths)[:limit]

    def get_cluster_summary(
        self, limit: int = 100, offset: int = 0
    ) -> list[dict]:
        return []

    def get_identity_summary(
        self, limit: int = 100, offset: int = 0
    ) -> list[dict]:
        return []


class FakeAnalyzer(FaceAnalyzer):
    """Stub face analyzer that returns configurable faces."""

    def __init__(self, faces_by_media: dict[str, list[Face]] | None = None):
        self.faces_by_media = faces_by_media or {}
        self.fail_for: set[str] = set()
        self._version = "fake_analyzer_v1"

    @property
    def version(self) -> str:
        return self._version

    def analyze(self, media: MediaFile) -> list[Face]:
        if media.path in self.fail_for:
            raise RuntimeError("analysis failed")
        return list(self.faces_by_media.get(media.id, []))


def _image_media(path: str = "/fake/IMG_1.jpg") -> MediaFile:
    return MediaFile(
        id=str(uuid4()),
        path=path,
        checksum="abc123",
        size_bytes=1234,
        mtime=0.0,
        media_type=MediaType.IMAGE,
    )


def _video_media(path: str = "/fake/clip.mp4") -> MediaFile:
    return MediaFile(
        id=str(uuid4()),
        path=path,
        checksum="def456",
        size_bytes=5678,
        mtime=0.0,
        media_type=MediaType.VIDEO,
    )


def _face(media_id: str, confidence: float = 0.9) -> Face:
    return Face(
        id=str(uuid4()),
        media_id=media_id,
        bbox=(10, 20, 30, 40),
        embedding=[0.0] * 512,
        embedding_version="fake_v1",
        detection_confidence=confidence,
    )


def test_detect_saves_faces_for_unprocessed_image() -> None:
    repo = FakeRepository()
    media = _image_media()
    repo.save_media(media)

    face = _face(media.id)
    analyzer = FakeAnalyzer(faces_by_media={media.id: [face]})
    service = FaceDetectionService(analyzer, repo)

    result = service.detect()

    assert result.processed == 1
    assert result.faces_detected == 1
    assert result.skipped == 0
    assert result.errors == 0
    assert len(repo.faces) == 1
    assert media.face_analysis_at is not None
    assert media.face_analysis_version == analyzer.version


def test_detect_skips_already_analysed_media() -> None:
    repo = FakeRepository()
    media = _image_media()
    media.face_analysis_at = datetime.utcnow()
    media.face_analysis_version = "old_version"
    repo.save_media(media)

    analyzer = FakeAnalyzer(faces_by_media={media.id: [_face(media.id)]})
    service = FaceDetectionService(analyzer, repo)

    result = service.detect()

    assert result.processed == 0
    assert result.skipped == 1
    assert result.faces_detected == 0
    assert media.face_analysis_version == "old_version"


def test_detect_force_reanalyses_media() -> None:
    repo = FakeRepository()
    media = _image_media()
    media.face_analysis_at = datetime.utcnow()
    media.face_analysis_version = "old_version"
    repo.save_media(media)

    face = _face(media.id)
    analyzer = FakeAnalyzer(faces_by_media={media.id: [face]})
    service = FaceDetectionService(analyzer, repo)

    result = service.detect(force=True)

    assert result.processed == 1
    assert result.faces_detected == 1
    assert result.skipped == 0
    assert media.face_analysis_version == analyzer.version


def test_detect_skips_non_image_media() -> None:
    repo = FakeRepository()
    video = _video_media()
    repo.save_media(video)

    analyzer = FakeAnalyzer(faces_by_media={video.id: [_face(video.id)]})
    service = FaceDetectionService(analyzer, repo)

    result = service.detect()

    assert result.processed == 0
    assert result.skipped == 1
    assert result.faces_detected == 0
    assert video.face_analysis_at is None


def test_detect_applies_min_confidence() -> None:
    repo = FakeRepository()
    media = _image_media()
    repo.save_media(media)

    high = _face(media.id, confidence=0.9)
    low = _face(media.id, confidence=0.3)
    analyzer = FakeAnalyzer(faces_by_media={media.id: [high, low]})
    service = FaceDetectionService(analyzer, repo)

    result = service.detect(min_confidence=0.5)

    assert result.faces_detected == 1
    assert len(repo.faces) == 1


def test_detect_handles_analyzer_error_gracefully() -> None:
    repo = FakeRepository()
    media = _image_media(path="/fake/broken.jpg")
    repo.save_media(media)

    analyzer = FakeAnalyzer()
    analyzer.fail_for.add(media.path)
    service = FaceDetectionService(analyzer, repo)

    result = service.detect()

    assert result.processed == 0
    assert result.errors == 1
    assert media.face_analysis_at is None


def test_detect_continues_after_analyzer_error() -> None:
    repo = FakeRepository()
    broken = _image_media(path="/fake/broken.jpg")
    good = _image_media(path="/fake/good.jpg")
    repo.save_media(broken)
    repo.save_media(good)

    face = _face(good.id)
    analyzer = FakeAnalyzer(faces_by_media={good.id: [face]})
    analyzer.fail_for.add(broken.path)
    service = FaceDetectionService(analyzer, repo)

    result = service.detect()

    assert result.processed == 1
    assert result.errors == 1
    assert result.faces_detected == 1
    assert len(repo.faces) == 1
    assert broken.face_analysis_at is None
    assert good.face_analysis_at is not None
