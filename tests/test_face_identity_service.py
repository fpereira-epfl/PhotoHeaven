"""Unit tests for the face identity service."""

from __future__ import annotations

from datetime import datetime

from photoheaven.application.face_identity_service import FaceIdentityService
from photoheaven.application.ports import MediaRepository
from photoheaven.domain.models import Face, Identity


class FakeRepository(MediaRepository):
    """In-memory repository stub for identity service tests."""

    def __init__(
        self,
        summaries: list[dict] | None = None,
        sample_paths: dict[int, list[str]] | None = None,
        faces: dict[str, Face] | None = None,
    ) -> None:
        self.summaries = summaries or []
        self.sample_paths = sample_paths or {}
        self.faces = faces or {}
        self.identities: dict[str, Identity] = {}
        self.named: dict[int, str | None] = {}

    def get_by_checksum(self, checksum: str):
        return None

    def save_media(self, media) -> None:
        pass

    def count_media(self) -> int:
        return 0

    def list_media(self, limit: int = 100, offset: int = 0):
        return []

    def save_face(self, face) -> None:
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

    def get_all_faces(self) -> list:
        return list(self.faces.values())

    def update_face_cluster_label(
        self, face_id: str, cluster_label: int | None
    ) -> None:
        face = self.faces.get(face_id)
        if face is not None:
            face.cluster_label = cluster_label

    def update_face_identity_name_for_cluster(
        self, cluster_label: int, identity_name: str | None
    ) -> None:
        self.named[cluster_label] = identity_name

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

    def get_faces_for_cluster(self, cluster_label: int) -> list[Face]:
        return [
            face
            for face in self.faces.values()
            if face.cluster_label == cluster_label
        ]

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
        paths = self.sample_paths.get(cluster_label, [])
        if not include_heic:
            paths = [
                p for p in paths if p.lower().endswith((".jpg", ".jpeg"))
            ]
        return paths[:limit]

    def get_cluster_summary(
        self, limit: int = 100, offset: int = 0
    ) -> list[dict]:
        return self.summaries[offset : offset + limit]


def test_list_clusters_returns_ordered_summaries() -> None:
    summaries = [
        {
            "cluster_label": 1,
            "face_count": 142,
            "photo_count": 100,
            "identity_name": None,
            "sample_path": "/photos/a.jpg",
        },
        {
            "cluster_label": 2,
            "face_count": 89,
            "photo_count": 80,
            "identity_name": "Alice",
            "sample_path": "/photos/b.jpg",
        },
    ]
    service = FaceIdentityService(FakeRepository(summaries))
    clusters = service.list_clusters()

    assert len(clusters) == 2
    assert clusters[0].cluster_label == 1
    assert clusters[0].face_count == 142
    assert clusters[0].photo_count == 100
    assert clusters[0].identity_name is None
    assert clusters[0].sample_path == "/photos/a.jpg"
    assert clusters[1].identity_name == "Alice"


def _face(
    face_id: str,
    cluster_label: int | None = None,
    identity_id: str | None = None,
    identity_name: str | None = None,
) -> Face:
    return Face(
        id=face_id,
        media_id="media-1",
        bbox=(0, 0, 1, 1),
        embedding=[0.0] * 512,
        cluster_label=cluster_label,
        identity_id=identity_id,
        identity_name=identity_name,
        created_at=datetime.utcnow(),
    )


def test_name_cluster_creates_identity_and_links_faces() -> None:
    face = _face("face-1", cluster_label=5)
    repo = FakeRepository(faces={"face-1": face})
    service = FaceIdentityService(repo)

    updated = service.name_cluster(5, "Bob")

    assert updated == 1
    assert face.identity_name == "Bob"
    assert face.identity_id is not None
    identity = repo.get_identity_by_id(face.identity_id)
    assert identity is not None
    assert identity.name == "Bob"


def test_name_cluster_reuses_existing_identity() -> None:
    existing = Identity(id="identity-1", name="Bob")
    face_a = _face("face-a", cluster_label=5)
    face_b = _face("face-b", cluster_label=7)
    repo = FakeRepository(
        faces={"face-a": face_a, "face-b": face_b},
    )
    repo.save_identity(existing)
    service = FaceIdentityService(repo)

    service.name_cluster(7, "Bob")

    assert face_b.identity_id == "identity-1"
    assert face_b.identity_name == "Bob"
    assert face_a.identity_id is None


def test_get_sample_photos_returns_paths() -> None:
    repo = FakeRepository(sample_paths={3: ["/a.jpg", "/b.jpg", "/c.jpg"]})
    service = FaceIdentityService(repo)

    paths = service.get_sample_photos(3, limit=2)

    assert paths == ["/a.jpg", "/b.jpg"]
