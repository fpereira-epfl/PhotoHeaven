"""Unit tests for the incremental face assignment service."""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from photoheaven.application.face_assignment_service import FaceAssignmentService
from photoheaven.application.ports import MediaRepository
from photoheaven.domain.models import Face, Identity


class FakeRepository(MediaRepository):
    """In-memory repository stub for assignment tests."""

    def __init__(self) -> None:
        self.media = {}
        self.faces: dict[str, Face] = {}
        self.identities: dict[str, Identity] = {}

    def get_by_checksum(self, checksum: str):
        return None

    def save_media(self, media) -> None:
        self.media[media.id] = media

    def count_media(self) -> int:
        return len(self.media)

    def list_media(self, limit: int = 100, offset: int = 0):
        return []

    def get_all_media_paths(self) -> list[str]:
        return [media.path for media in self.media.values()]

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
        return self.faces.get(face_id)

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
        face = self.faces.get(face_id)
        if face is not None:
            face.cluster_label = cluster_label

    def update_face_identity_name_for_cluster(
        self, cluster_label: int, identity_name: str | None
    ) -> None:
        pass

    def save_identity(self, identity: Identity) -> None:
        self.identities[identity.id] = identity

    def get_identity_by_name(self, name: str) -> Identity | None:
        for identity in self.identities.values():
            if identity.name == name:
                return identity
        return None

    def get_identity_by_id(self, identity_id: str) -> Identity | None:
        return self.identities.get(identity_id)

    def list_identities(self, limit: int = 100, offset: int = 0) -> list[Identity]:
        return sorted(
            self.identities.values(), key=lambda i: i.name
        )[offset : offset + limit]

    def get_faces_for_cluster(self, cluster_label: int):
        return []

    def get_faces_for_identity(self, identity_id: str) -> list[Face]:
        return [
            face
            for face in self.faces.values()
            if face.identity_id == identity_id
        ]

    def get_faces_without_identity(
        self, limit: int = 100, offset: int = 0
    ) -> list[Face]:
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

    def update_media_perceptual_hash(
        self, media_id: str, perceptual_hash: str
    ) -> None:
        pass

    def clear_duplicate_groups(self) -> None:
        pass

    def save_duplicate_group(
        self, group_id: str, members: list[dict]
    ) -> None:
        pass

    def list_duplicate_groups(self) -> list[dict]:
        return []


def _face(embedding: list[float]) -> Face:
    return Face(
        id=str(uuid4()),
        media_id=str(uuid4()),
        bbox=(0, 0, 10, 10),
        embedding=embedding,
        embedding_version="test",
        detection_confidence=0.9,
        created_at=datetime.utcnow(),
    )


def test_assign_links_new_face_to_named_identity() -> None:
    repo = FakeRepository()

    identity = Identity(id="identity-1", name="Alice")
    repo.save_identity(identity)

    # Two faces already named as Alice -> centroid is [1, 0, 0].
    for _ in range(2):
        face = _face([1.0, 0.0, 0.0])
        face.identity_id = identity.id
        face.identity_name = identity.name
        repo.save_face(face)

    # A new, similar face without an identity.
    new_face = _face([1.0, 0.0, 0.0])
    repo.save_face(new_face)

    service = FaceAssignmentService(repo)
    result = service.assign(eps=0.1)

    assert result.assigned_faces == 1
    assert result.unassigned_faces == 0
    assert result.identities_used == 1
    assert new_face.identity_id == "identity-1"
    assert new_face.identity_name == "Alice"


def test_assign_leaves_distant_faces_unassigned() -> None:
    repo = FakeRepository()

    identity = Identity(id="identity-1", name="Alice")
    repo.save_identity(identity)

    face = _face([1.0, 0.0, 0.0])
    face.identity_id = identity.id
    face.identity_name = identity.name
    repo.save_face(face)

    new_face = _face([0.0, 1.0, 0.0])
    repo.save_face(new_face)

    service = FaceAssignmentService(repo)
    result = service.assign(eps=0.1)

    assert result.assigned_faces == 0
    assert result.unassigned_faces == 1
    assert new_face.identity_id is None


def test_assign_returns_zero_when_no_identities_exist() -> None:
    repo = FakeRepository()
    repo.save_face(_face([1.0, 0.0, 0.0]))

    service = FaceAssignmentService(repo)
    result = service.assign()

    assert result.identities_used == 0
    assert result.assigned_faces == 0
    assert result.unassigned_faces == 0


def test_assign_rejects_mixed_embedding_versions() -> None:
    repo = FakeRepository()
    identity = Identity(id="identity-1", name="Alice")
    repo.save_identity(identity)

    face_a = _face([1.0, 0.0, 0.0])
    face_a.embedding_version = "v1"
    face_a.identity_id = identity.id
    face_b = _face([1.0, 0.0, 0.0])
    face_b.embedding_version = "v2"
    repo.save_face(face_a)
    repo.save_face(face_b)

    service = FaceAssignmentService(repo)

    try:
        service.assign()
        raise AssertionError("Expected ValueError")
    except ValueError as exc:
        assert "mixed embedding versions" in str(exc)
