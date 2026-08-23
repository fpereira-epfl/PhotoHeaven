"""Unit tests for the face clustering application service."""

from __future__ import annotations

from uuid import uuid4

from photoheaven.application.face_clustering_service import FaceClusteringService
from photoheaven.application.ports import MediaRepository
from photoheaven.domain.models import Face


class FakeRepository(MediaRepository):
    """In-memory repository stub with just enough for clustering tests."""

    def __init__(self) -> None:
        self.media = {}
        self.faces: dict[str, Face] = {}
        self.cluster_labels: dict[str, int | None] = {}

    def get_by_checksum(self, checksum: str):
        return None

    def save_media(self, media) -> None:
        self.media[media.id] = media

    def count_media(self) -> int:
        return len(self.media)

    def list_media(self, limit: int = 100, offset: int = 0):
        return []

    def save_face(self, face: Face) -> None:
        self.faces[face.id] = face

    def count_faces(self) -> int:
        return len(self.faces)

    def media_has_faces(self, media_id: str) -> bool:
        return False

    def get_unprocessed_faces_media(self, limit: int = 100, offset: int = 0):
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

    def get_all_faces(self) -> list[Face]:
        return list(self.faces.values())

    def update_face_cluster_label(
        self, face_id: str, cluster_label: int | None
    ) -> None:
        self.cluster_labels[face_id] = cluster_label
        if face_id in self.faces:
            self.faces[face_id].cluster_label = cluster_label

    def update_face_identity_name_for_cluster(
        self, cluster_label: int, identity_name: str | None
    ) -> None:
        pass


def _face(embedding: list[float]) -> Face:
    return Face(
        id=str(uuid4()),
        media_id=str(uuid4()),
        bbox=(0, 0, 10, 10),
        embedding=embedding,
        embedding_version="test",
        detection_confidence=0.9,
    )


def test_cluster_groups_similar_faces() -> None:
    repo = FakeRepository()
    # Three identical embeddings -> one cluster.
    for _ in range(3):
        repo.save_face(_face([1.0, 0.0, 0.0]))
    # Two identical embeddings -> second cluster.
    for _ in range(2):
        repo.save_face(_face([0.0, 1.0, 0.0]))
    # One outlier -> noise.
    repo.save_face(_face([0.0, 0.0, 1.0]))

    service = FaceClusteringService(repo)
    result = service.cluster(eps=0.1, min_samples=2)

    assert result.total_faces == 6
    assert result.num_clusters == 2
    assert result.clustered_faces == 5
    assert result.noise_faces == 1


def test_empty_library_returns_zero_summary() -> None:
    repo = FakeRepository()
    service = FaceClusteringService(repo)
    result = service.cluster()

    assert result.total_faces == 0
    assert result.num_clusters == 0
    assert result.clustered_faces == 0
    assert result.noise_faces == 0


def test_unsupported_algorithm_raises() -> None:
    repo = FakeRepository()
    service = FaceClusteringService(repo)

    try:
        service.cluster(algorithm="hdbscan")
        raise AssertionError("Expected ValueError")
    except ValueError as exc:
        assert "hdbscan" in str(exc)
