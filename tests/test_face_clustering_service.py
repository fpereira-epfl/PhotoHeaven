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

    def get_all_media_paths(self) -> list[str]:
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
        self.cluster_labels[face_id] = cluster_label
        if face_id in self.faces:
            self.faces[face_id].cluster_label = cluster_label

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
        pass

    def get_identity_photo_counts(self) -> dict[str, int]:
        return {}


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


def test_cluster_propagates_identity_to_new_cluster_label() -> None:
    repo = FakeRepository()
    # Existing named cluster with three faces.
    for _ in range(3):
        face = _face([1.0, 0.0, 0.0])
        face.identity_id = "identity-alice"
        face.identity_name = "Alice"
        repo.save_face(face)
    # New, similar face without a name.
    repo.save_face(_face([1.0, 0.0, 0.0]))

    service = FaceClusteringService(repo)
    result = service.cluster(eps=0.1, min_samples=2)

    assert result.num_clusters == 1
    for face in repo.faces.values():
        assert face.cluster_label == 0
        assert face.identity_id == "identity-alice"
        assert face.identity_name == "Alice"


def test_cluster_migrates_legacy_name_only_faces() -> None:
    repo = FakeRepository()
    for _ in range(2):
        face = _face([1.0, 0.0, 0.0])
        # Simulate pre-identity-table rows: name set, identity_id absent.
        face.identity_name = "Alice"
        face.identity_id = None
        repo.save_face(face)

    service = FaceClusteringService(repo)
    result = service.cluster(eps=0.1, min_samples=2)

    assert result.num_clusters == 1
    identities = {
        (face.identity_id, face.identity_name) for face in repo.faces.values()
    }
    assert len(identities) == 1
    identity_id, identity_name = identities.pop()
    assert identity_name == "Alice"
    assert identity_id is not None


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


def test_cluster_rejects_mixed_embedding_versions() -> None:
    repo = FakeRepository()
    face_a = _face([1.0, 0.0, 0.0])
    face_a.embedding_version = "v1"
    face_b = _face([1.0, 0.0, 0.0])
    face_b.embedding_version = "v2"
    repo.save_face(face_a)
    repo.save_face(face_b)

    service = FaceClusteringService(repo)

    try:
        service.cluster()
        raise AssertionError("Expected ValueError")
    except ValueError as exc:
        assert "mixed embedding versions" in str(exc)
